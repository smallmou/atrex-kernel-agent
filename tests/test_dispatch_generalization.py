from __future__ import annotations

import unittest

from orchestrator import optimization_policy, optimize

BF16 = "torch.bfloat16"
STRIDED = "torch.strided"


def _tensor(shape: list[int], dtype: str = BF16) -> list:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.insert(0, running)
        running *= dim
    return ["tensor", list(shape), stride, dtype, STRIDED, False]


def _record(index: int, call_kwargs: list, init_kwargs: list | None = None) -> dict:
    return {
        "index": index,
        "id": str(index),
        "init": ["invocation", [], init_kwargs or []],
        "call": ["invocation", [], call_kwargs],
    }


def _rows(index: int, rows: int, dtype: str = BF16) -> dict:
    """One lm_head-style workload whose only varying axis is x's leading dimension."""
    return _record(
        index,
        [
            ["x", _tensor([rows, 5120], dtype)],
            ["weight", _tensor([248320, 5120])],
        ],
    )


class DispatchIntervalDerivationTest(unittest.TestCase):
    def test_single_varying_dim_becomes_the_routing_axis(self) -> None:
        records = [_rows(0, 1), _rows(1, 16), _rows(2, 20), _rows(3, 32), _rows(4, 48)]
        buckets = {0: "le16", 1: "le16", 2: "m17_32", 3: "m17_32", 4: "m33_64"}
        derived = optimize.derive_dispatch_generalization(records, buckets)

        self.assertEqual(derived.status, "interval_v1")
        self.assertTrue(derived.enabled)
        self.assertEqual(derived.route_path, ("call", "kwargs", "x", "shape", 0))
        self.assertEqual(derived.route_links, ())
        self.assertEqual(
            derived.bucket_intervals,
            (("le16", 1, 16), ("m17_32", 20, 32), ("m33_64", 48, 48)),
        )
        # The invariant weight tensor and dtypes must land in the compatibility family.
        family = dict(derived.family)
        self.assertEqual(family[("call", "kwargs", "x", "dtype")], BF16)
        self.assertEqual(family[("call", "kwargs", "weight", "shape", 0)], 248320)
        self.assertNotIn(("call", "kwargs", "x", "shape", 0), family)

    def test_covarying_axes_collapse_to_one_representative_plus_links(self) -> None:
        # A contiguous 2D tensor makes stride[0] track shape[1], so both vary together.
        records = [
            _record(index, [["x", _tensor([4, cols])]])
            for index, cols in enumerate((64, 128, 512, 1024))
        ]
        buckets = {0: "small", 1: "small", 2: "big", 3: "big"}
        derived = optimize.derive_dispatch_generalization(records, buckets)

        self.assertEqual(derived.status, "interval_v1")
        self.assertEqual(derived.route_path, ("call", "kwargs", "x", "shape", 1))
        self.assertEqual(derived.route_links, (("call", "kwargs", "x", "stride", 0),))

    def test_separable_axis_wins_over_interleaved_one(self) -> None:
        records = [
            _record(0, [["x", _tensor([4, 100])]]),
            _record(1, [["x", _tensor([8, 900])]]),
            _record(2, [["x", _tensor([40, 200])]]),
            _record(3, [["x", _tensor([80, 800])]]),
        ]
        buckets = {0: "a", 1: "a", 2: "b", 3: "b"}
        derived = optimize.derive_dispatch_generalization(records, buckets)

        self.assertEqual(derived.status, "interval_v1")
        self.assertEqual(derived.route_path, ("call", "kwargs", "x", "shape", 0))
        self.assertEqual(derived.bucket_intervals, (("a", 4, 8), ("b", 40, 80)))

    def test_varying_dtype_falls_back_to_exact_dispatch(self) -> None:
        records = [_rows(0, 4), _rows(1, 8, dtype="torch.float16")]
        derived = optimize.derive_dispatch_generalization(records, {0: "a", 1: "b"})
        self.assertEqual(derived.status, "structure_mismatch")
        self.assertFalse(derived.enabled)

    def test_differing_argument_sets_fall_back_to_exact_dispatch(self) -> None:
        records = [
            _record(0, [["x", _tensor([4, 8])]]),
            _record(1, [["x", _tensor([8, 8])], ["extra", ["int", 3]]]),
        ]
        derived = optimize.derive_dispatch_generalization(records, {0: "a", 1: "b"})
        self.assertEqual(derived.status, "structure_mismatch")

    def test_interleaved_buckets_are_not_separable(self) -> None:
        records = [_rows(index, rows) for index, rows in enumerate((10, 30, 20, 40))]
        buckets = {0: "a", 1: "a", 2: "b", 3: "b"}
        derived = optimize.derive_dispatch_generalization(records, buckets)
        self.assertEqual(derived.status, "not_separable")
        self.assertFalse(derived.enabled)
        # The family is still reported so the rejection reason is inspectable.
        self.assertTrue(derived.family)

    def test_no_varying_axis_reports_no_axis(self) -> None:
        records = [_rows(0, 4), _rows(1, 4)]
        derived = optimize.derive_dispatch_generalization(records, {0: "a", 1: "a"})
        self.assertEqual(derived.status, "no_axis")

    def test_scalar_int_argument_can_be_the_routing_axis(self) -> None:
        records = [
            _record(index, [["x", _tensor([4, 8])], ["seqlen", ["int", value]]])
            for index, value in enumerate((16, 32, 4096, 8192))
        ]
        buckets = {0: "short", 1: "short", 2: "long", 3: "long"}
        derived = optimize.derive_dispatch_generalization(records, buckets)
        self.assertEqual(derived.status, "interval_v1")
        self.assertEqual(derived.route_path, ("call", "kwargs", "seqlen", "value"))


class RoutingConsistencyTest(unittest.TestCase):
    def test_endpoints_are_probed_and_a_clean_split_passes(self) -> None:
        records = [_rows(index, rows) for index, rows in enumerate((4, 6, 8, 64, 96, 128))]
        buckets = {0: "small", 1: "small", 2: "small", 3: "big", 4: "big", 5: "big"}
        report = optimize.dispatch_routing_consistency(records, buckets)

        self.assertEqual(report["non_adjacent_misroutes"], [])
        self.assertEqual(report["axis_flips"], [])
        # Four endpoints (4, 8, 64, 128) are probed; interior 6 and 96 are not, because
        # removing them cannot move their bucket's min/max.
        self.assertEqual(report["endpoint_checks"], 4)

    def test_removing_an_interior_workload_is_a_provable_no_op(self) -> None:
        records = [_rows(index, rows) for index, rows in enumerate((4, 6, 8, 64, 96, 128))]
        buckets = {0: "small", 1: "small", 2: "small", 3: "big", 4: "big", 5: "big"}
        baseline = optimize.derive_dispatch_generalization(records, buckets)
        without_interior = optimize.derive_dispatch_generalization(
            [record for record in records if int(record["index"]) != 1], buckets
        )
        self.assertEqual(baseline.bucket_intervals, without_interior.bucket_intervals)

    def test_gap_ratio_reports_the_widest_unmeasured_span(self) -> None:
        records = [_rows(index, rows) for index, rows in enumerate((10, 20, 90, 100))]
        buckets = {0: "low", 1: "low", 2: "high", 3: "high"}
        report = optimize.dispatch_routing_consistency(records, buckets)
        # Span 10..100 is 90 wide; the gap between 20 and 90 is 70.
        self.assertAlmostEqual(report["max_gap_ratio"], 70 / 90, places=3)

    def test_axis_instability_is_reported(self) -> None:
        # q wins the axis contest only because one workload holds its minimum; dropping that
        # workload collapses q's dynamic range and the best axis flips to p.
        def two_axis(index: int, p: int, q: int) -> dict:
            return _record(index, [["p", ["int", p]], ["q", ["int", q]]])

        records = [
            two_axis(0, 100, 1),
            two_axis(1, 200, 3000),
            two_axis(2, 300, 3100),
            two_axis(3, 400, 3200),
        ]
        buckets = {0: "a", 1: "a", 2: "b", 3: "b"}
        baseline = optimize.derive_dispatch_generalization(records, buckets)
        self.assertEqual(baseline.route_path, ("call", "kwargs", "q", "value"))

        report = optimize.dispatch_routing_consistency(records, buckets)
        self.assertEqual(len(report["axis_flips"]), 1)
        self.assertEqual(report["axis_flips"][0]["index"], 0)
        self.assertIn("axis moved to", report["axis_flips"][0]["reason"])

    def test_endpoint_removal_cannot_misroute_past_a_neighbour(self) -> None:
        # Intervals are ordered, so distance grows monotonically with position: a dropped
        # endpoint always lands in its own or an adjacent regime. The probe therefore
        # asserts an invariant rather than discriminating between good and bad splits.
        records = [_rows(index, rows) for index, rows in enumerate((1, 2, 400, 800, 810))]
        buckets = {0: "low", 1: "low", 2: "mid", 3: "high", 4: "high"}
        report = optimize.dispatch_routing_consistency(records, buckets)
        self.assertEqual(report["non_adjacent_misroutes"], [])
        self.assertGreater(report["endpoint_checks"], 0)

    def test_exact_only_derivation_reports_an_empty_probe(self) -> None:
        records = [_rows(index, rows) for index, rows in enumerate((10, 30, 20, 40))]
        buckets = {0: "a", 1: "a", 2: "b", 3: "b"}
        report = optimize.dispatch_routing_consistency(records, buckets)
        self.assertEqual(report["endpoint_checks"], 0)
        self.assertEqual(report["non_adjacent_misroutes"], [])
        self.assertEqual(report["axis_flips"], [])


class IntervalRoutingTest(unittest.TestCase):
    INTERVALS = ((1, 16, 0), (20, 32, 1), (48, 64, 2))

    def test_values_inside_an_interval_route_to_its_bucket(self) -> None:
        for key, expected in ((1, 0), (9, 0), (16, 0), (20, 1), (32, 1), (48, 2), (64, 2)):
            self.assertEqual(optimize._route_interval(key, self.INTERVALS), expected)

    def test_gaps_route_to_the_nearest_interval(self) -> None:
        self.assertEqual(optimize._route_interval(17, self.INTERVALS), 0)
        self.assertEqual(optimize._route_interval(18, self.INTERVALS), 0)
        self.assertEqual(optimize._route_interval(19, self.INTERVALS), 1)
        self.assertEqual(optimize._route_interval(40, self.INTERVALS), 1)
        self.assertEqual(optimize._route_interval(41, self.INTERVALS), 2)

    def test_out_of_range_values_clamp_monotonically(self) -> None:
        self.assertEqual(optimize._route_interval(0, self.INTERVALS), 0)
        self.assertEqual(optimize._route_interval(-99, self.INTERVALS), 0)
        self.assertEqual(optimize._route_interval(65, self.INTERVALS), 2)
        self.assertEqual(optimize._route_interval(10 ** 9, self.INTERVALS), 2)

    def test_equidistant_gap_prefers_the_lower_bucket_index(self) -> None:
        # 18 is 2 away from both interval 0's high and interval 1's low.
        self.assertEqual(optimize._route_interval(18, ((1, 16, 0), (20, 32, 1))), 0)


class FamilyGuardTest(unittest.TestCase):
    def _signature(self, rows: int, dtype: str = BF16) -> tuple:
        record = _rows(0, rows, dtype)
        return (
            optimize._freeze_dispatch_signature(record["init"]),
            optimize._freeze_dispatch_signature(record["call"]),
        )

    def setUp(self) -> None:
        records = [_rows(0, 4), _rows(1, 64)]
        self.derived = optimize.derive_dispatch_generalization(records, {0: "a", 1: "b"})
        self.assertEqual(self.derived.status, "interval_v1")

    def test_unseen_size_inside_the_family_is_accepted(self) -> None:
        signature = self._signature(37)
        self.assertIsNone(optimize._family_violation(signature, self.derived.family))
        key = optimize._signature_leaf(signature, self.derived.route_path)
        self.assertEqual(key, 37)

    def test_foreign_dtype_violates_the_family(self) -> None:
        signature = self._signature(37, dtype="torch.float64")
        violation = optimize._family_violation(signature, self.derived.family)
        self.assertEqual(violation, ("call", "kwargs", "x", "dtype"))

    def test_missing_keyword_violates_the_family(self) -> None:
        init = optimize._freeze_dispatch_signature(["invocation", [], []])
        call = optimize._freeze_dispatch_signature(
            ["invocation", [], [["x", _tensor([37, 5120])]]]
        )
        violation = optimize._family_violation((init, call), self.derived.family)
        self.assertIsNotNone(violation)


class GeneratedDispatcherTest(unittest.TestCase):
    """Exercise the generated source itself, using a torch stub.

    The real torch is unavailable here (which is why tests/test_aggregate_dispatch.py
    cannot even be collected), so the stub keeps the generated dispatcher executable.
    """

    BUCKET_SOURCES = {
        "small": (
            "import torch\n\n\nBIAS = 1.0\n\n\n"
            "class Model(torch.nn.Module):\n"
            "    def __init__(self, **kwargs):\n        super().__init__()\n\n"
            "    def forward(self, **kwargs):\n        return BIAS\n"
        ),
        "large": (
            "import torch\n\n\nBIAS = 10.0\n\n\n"
            "class Model(torch.nn.Module):\n"
            "    def __init__(self, **kwargs):\n        super().__init__()\n\n"
            "    def forward(self, **kwargs):\n        return BIAS\n"
        ),
    }

    class _StubTensor:
        def __init__(self, shape: tuple[int, ...], dtype: str = BF16) -> None:
            self._shape = tuple(shape)
            self._dtype = dtype
            self.requires_grad = False

        @property
        def shape(self) -> tuple[int, ...]:
            return self._shape

        def stride(self) -> tuple[int, ...]:
            strides: list[int] = []
            running = 1
            for dim in reversed(self._shape):
                strides.insert(0, running)
                running *= dim
            return tuple(strides)

        @property
        def dtype(self) -> str:
            return self._dtype

        @property
        def layout(self) -> str:
            return STRIDED

    def setUp(self) -> None:
        import sys
        import types

        records = [
            _record(index, [["x", _tensor([rows, 64])]])
            for index, rows in enumerate((4, 8, 64, 128))
        ]
        buckets = {0: "small", 1: "small", 2: "large", 3: "large"}
        self.derived = optimize.derive_dispatch_generalization(records, buckets)
        self.assertEqual(self.derived.status, "interval_v1")

        source = optimize.build_deterministic_dispatcher(
            kind="shapes",
            signature_records=records,
            bucket_by_index=buckets,
            module_records={"small": {"kernel_blob": "a"}, "large": {"kernel_blob": "b"}},
            module_sources=self.BUCKET_SOURCES,
            generalization=self.derived,
        )
        self.source = source

        stub = types.ModuleType("torch")
        stub.Tensor = self._StubTensor
        stub.nn = types.SimpleNamespace(
            Module=type("Module", (object,), {"__init__": lambda self: None}),
            ModuleDict=dict,
        )
        self._saved_torch = sys.modules.get("torch")
        sys.modules["torch"] = stub
        self.addCleanup(self._restore_torch)
        self.namespace: dict = {}
        exec(compile(source, "<generated>", "exec"), self.namespace)

    def _restore_torch(self) -> None:
        import sys

        if self._saved_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self._saved_torch

    def _route(self, rows: int, cols: int = 64, dtype: str = BF16) -> int:
        empty_init = self.namespace["_invocation_signature"]((), {})
        tensor = self._StubTensor((rows, cols), dtype)
        return self.namespace["_select_bucket"](empty_init, (), {"x": tensor})

    def test_generated_source_embeds_the_routing_rule(self) -> None:
        for marker in (
            "_DISPATCH_FAMILY",
            "_DISPATCH_ROUTE_PATH",
            "_DISPATCH_INTERVALS",
            "_route_interval",
            "_family_violation",
        ):
            self.assertIn(marker, self.source)

    def test_benchmarked_shapes_route_through_the_exact_table(self) -> None:
        small = self.namespace["_SIGNATURE_TO_BUCKET"]
        self.assertEqual(len(small), 4)
        self.assertEqual(self._route(4), self._route(8))
        self.assertEqual(self._route(64), self._route(128))
        self.assertNotEqual(self._route(8), self._route(64))

    def test_unseen_sizes_route_to_the_nearest_regime(self) -> None:
        small_bucket = self._route(8)
        large_bucket = self._route(64)
        self.assertEqual(self._route(1), small_bucket)
        self.assertEqual(self._route(6), small_bucket)
        self.assertEqual(self._route(20), small_bucket)
        self.assertEqual(self._route(300), large_bucket)

    def test_out_of_family_input_still_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            self._route(6, cols=999)
        with self.assertRaises(RuntimeError):
            self._route(6, dtype="torch.float64")

    def test_exact_only_generation_keeps_raising_on_unseen_shapes(self) -> None:
        records = [
            _record(index, [["x", _tensor([rows, 64])]])
            for index, rows in enumerate((4, 8, 64, 128))
        ]
        buckets = {0: "small", 1: "small", 2: "large", 3: "large"}
        source = optimize.build_deterministic_dispatcher(
            kind="shapes",
            signature_records=records,
            bucket_by_index=buckets,
            module_records={"small": {"kernel_blob": "a"}, "large": {"kernel_blob": "b"}},
            module_sources=self.BUCKET_SOURCES,
            generalization=None,
        )
        self.assertNotIn("_DISPATCH_INTERVALS", source)
        namespace: dict = {}
        exec(compile(source, "<generated-exact>", "exec"), namespace)
        empty_init = namespace["_invocation_signature"]((), {})
        with self.assertRaises(RuntimeError):
            namespace["_select_bucket"](
                empty_init, (), {"x": self._StubTensor((6, 64))}
            )


class SyntheticShapeTest(unittest.TestCase):
    def test_scalar_only_entries_are_interpolation_safe(self) -> None:
        self.assertTrue(
            optimize._interpolation_safe(
                {"init_kwargs": None, "input_kwargs": {"m": 4, "n": 248320, "k": 5120}}
            )
        )
        self.assertTrue(
            optimize._interpolation_safe(
                {
                    "init_kwargs": {"beta": 1.0, "threshold": 20.0},
                    "input_kwargs": {"num_tokens": 450, "num_heads": 48},
                }
            )
        )

    def test_coupled_entries_are_refused(self) -> None:
        # causal_conv1d encodes offsets derived from x's second dimension as a string.
        self.assertFalse(
            optimize._interpolation_safe(
                {
                    "init_kwargs": {"activation": "silu"},
                    "input_kwargs": {"x": [8192, 446], "query_start_loc": "0,446"},
                }
            )
        )
        # flash_attention couples block_table and seq_lens to the query length.
        self.assertFalse(
            optimize._interpolation_safe(
                {
                    "init_kwargs": {"max_seqlen_q": 446},
                    "input_kwargs": {"q": [450, 24, 256], "seq_lens": "4323,446"},
                }
            )
        )
        # A value repeated from init_kwargs is probably derived from it.
        self.assertFalse(
            optimize._interpolation_safe(
                {"init_kwargs": {"seqlen": 512}, "input_kwargs": {"tokens": 512, "heads": 8}}
            )
        )

    def _source(self, values: list[int]) -> tuple:
        entries = tuple({"init_kwargs": None, "input_kwargs": {"m": value, "n": 9}} for value in values)
        ids = tuple(str(index) for index in range(len(values)))
        return optimize.WorkloadSource(
            kind="shapes", filename="shapes.json", ids=ids, entries=entries
        )

    def test_synthesis_interpolates_inside_the_widest_bucket(self) -> None:
        values = [4, 8, 40, 80]
        records = [
            _record(index, [["x", _tensor([value, 9])]]) for index, value in enumerate(values)
        ]
        buckets = {0: "small", 1: "small", 2: "big", 3: "big"}
        derived = optimize.derive_dispatch_generalization(records, buckets)
        self.assertEqual(derived.status, "interval_v1")

        result = optimize.synthesize_unseen_shape(
            self._source(values), records, buckets, derived
        )
        self.assertIsNotNone(result)
        shape_id, entry, key = result
        # Widest interval is big (40..80), so the midpoint is 60 and was never benchmarked.
        self.assertEqual(key, 60)
        self.assertEqual(entry["input_kwargs"]["m"], 60)
        self.assertNotIn(key, values)
        # The evaluator seeds its RNG with int(shape_id) and rejects non-numeric ids; the id
        # must also outrank every existing one so recorded workload indices stay valid.
        self.assertTrue(shape_id.isdigit())
        self.assertGreater(int(shape_id), max(int(existing) for existing in ("0", "1", "2", "3")))

    def test_synthesis_declines_when_entries_are_unsafe(self) -> None:
        values = [4, 8, 40, 80]
        records = [
            _record(index, [["x", _tensor([value, 9])]]) for index, value in enumerate(values)
        ]
        buckets = {0: "small", 1: "small", 2: "big", 3: "big"}
        derived = optimize.derive_dispatch_generalization(records, buckets)
        unsafe = optimize.WorkloadSource(
            kind="shapes",
            filename="shapes.json",
            ids=tuple(str(index) for index in range(4)),
            entries=tuple(
                {"init_kwargs": None, "input_kwargs": {"x": [value, 9], "loc": f"0,{value}"}}
                for value in values
            ),
        )
        self.assertIsNone(
            optimize.synthesize_unseen_shape(unsafe, records, buckets, derived)
        )

    def test_synthesis_declines_without_a_generalization(self) -> None:
        values = [10, 30, 20, 40]
        records = [
            _record(index, [["x", _tensor([value, 9])]]) for index, value in enumerate(values)
        ]
        buckets = {0: "a", 1: "a", 2: "b", 3: "b"}
        derived = optimize.derive_dispatch_generalization(records, buckets)
        self.assertFalse(derived.enabled)
        self.assertIsNone(
            optimize.synthesize_unseen_shape(self._source(values), records, buckets, derived)
        )


class ManifestGeneralizationValidationTest(unittest.TestCase):
    MODULES = {"low": {"embedded": True}, "high": {"embedded": True}}

    def _valid(self, **overrides) -> dict:
        block = {
            "status": "interval_v1",
            "family": [[["call", "kwargs", "x", "dtype"], BF16]],
            "route_path": ["call", "kwargs", "x", "shape", 0],
            "intervals": {"low": [1, 16], "high": [20, 32]},
        }
        block.update(overrides)
        return block

    def test_valid_interval_block_is_accepted(self) -> None:
        self.assertEqual(
            optimization_policy._generalization_errors(self._valid(), self.MODULES), []
        )

    def test_degraded_statuses_need_no_routing_data(self) -> None:
        for status in ("structure_mismatch", "not_separable", "no_axis", "loo_failed", "disabled"):
            self.assertEqual(
                optimization_policy._generalization_errors({"status": status}, self.MODULES),
                [],
                status,
            )

    def test_missing_block_and_unknown_status_are_rejected(self) -> None:
        self.assertTrue(optimization_policy._generalization_errors(None, self.MODULES))
        self.assertTrue(
            optimization_policy._generalization_errors({"status": "bogus"}, self.MODULES)
        )

    def test_overlapping_intervals_are_rejected(self) -> None:
        errors = optimization_policy._generalization_errors(
            self._valid(intervals={"low": [1, 25], "high": [20, 32]}), self.MODULES
        )
        self.assertTrue(any("overlap" in error for error in errors))

    def test_intervals_must_cover_exactly_the_declared_buckets(self) -> None:
        errors = optimization_policy._generalization_errors(
            self._valid(intervals={"low": [1, 16]}), self.MODULES
        )
        self.assertTrue(any("exactly the declared buckets" in error for error in errors))

    def test_inverted_bounds_are_rejected(self) -> None:
        errors = optimization_policy._generalization_errors(
            self._valid(intervals={"low": [16, 1], "high": [20, 32]}), self.MODULES
        )
        self.assertTrue(any("invalid interval" in error for error in errors))

    def test_missing_family_or_route_path_is_rejected(self) -> None:
        self.assertTrue(
            optimization_policy._generalization_errors(
                self._valid(family=[]), self.MODULES
            )
        )
        self.assertTrue(
            optimization_policy._generalization_errors(
                self._valid(route_path=[]), self.MODULES
            )
        )


if __name__ == "__main__":
    unittest.main()
