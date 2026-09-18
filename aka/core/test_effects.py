"""Behaviour tests for :mod:`aka.core.effects`.

``EffectStack`` is the teardown contract for every plugin contribution, so these tests pin
the observable guarantees its docstrings make: LIFO unwind order, one bad disposer never
starving the rest, an aggregate ``DisposalErrors`` that names ``owner:label``, idempotent
early disposers, immediate disposal of anything registered while an unwind is in flight,
and ``BaseException`` passing through untouched instead of being collected.
"""

from __future__ import annotations

import unittest

from aka.core.effects import EffectStack
from aka.core.errors import CoreError, DisposalErrors


class _Abort(BaseException):
    """Stand-in for ``EnvironmentUnavailable``: a ``BaseException``, not an ``Exception``."""


class EffectStackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.order: list[str] = []
        self.stack = EffectStack("fiber-a")

    # -- helpers ---------------------------------------------------------

    def record(self, name: str):
        """A disposer that appends its name when it runs."""

        def dispose() -> None:
            self.order.append(name)

        return dispose

    def failing(self, name: str, error: BaseException):
        """A disposer that records that it ran and then fails."""

        def dispose() -> None:
            self.order.append(name)
            raise error

        return dispose

    # -- order -----------------------------------------------------------

    def test_disposers_run_in_lifo_order(self) -> None:
        for name in ("alpha", "beta", "gamma"):
            self.stack.add(self.record(name), label=name)

        self.stack.unwind()

        self.assertEqual(self.order, ["gamma", "beta", "alpha"])
        self.assertEqual(len(self.stack), 0)
        self.assertEqual(self.stack.live, ())

    def test_live_reports_labels_in_registration_order(self) -> None:
        self.stack.add(self.record("alpha"), label="alpha")
        self.stack.add(self.record("beta"), label="beta")

        # `live` is a registration-order roster; the reversal happens at unwind time.
        self.assertEqual(self.stack.live, ("alpha", "beta"))

    def test_blank_label_falls_back_to_effect(self) -> None:
        self.stack.add(self.record("alpha"))

        self.assertEqual(self.stack.live, ("effect",))

    def test_unwind_of_an_empty_stack_is_a_noop(self) -> None:
        self.stack.unwind()
        self.stack.unwind()

        self.assertEqual(self.order, [])
        self.assertEqual(len(self.stack), 0)

    # -- failure collection ----------------------------------------------

    def test_failing_disposer_does_not_starve_the_rest(self) -> None:
        error = RuntimeError("boom")
        self.stack.add(self.record("alpha"), label="alpha")
        self.stack.add(self.failing("beta", error), label="beta")
        self.stack.add(self.record("gamma"), label="gamma")

        with self.assertRaises(DisposalErrors) as caught:
            self.stack.unwind()

        # The disposer registered *before* the failing one still ran.
        self.assertEqual(self.order, ["gamma", "beta", "alpha"])
        self.assertIsInstance(caught.exception, CoreError)
        self.assertEqual(caught.exception.failures, (("fiber-a:beta", error),))
        self.assertIn("fiber-a:beta", str(caught.exception))
        self.assertIn("boom", str(caught.exception))
        # The aggregate is raised only after the stack has been fully drained.
        self.assertEqual(len(self.stack), 0)
        self.assertEqual(self.stack.live, ())

    def test_every_failure_is_collected_in_unwind_order(self) -> None:
        first = RuntimeError("first")
        last = ValueError("last")
        self.stack.add(self.failing("alpha", first), label="alpha")
        self.stack.add(self.record("beta"), label="beta")
        self.stack.add(self.failing("gamma", last), label="gamma")

        with self.assertRaises(DisposalErrors) as caught:
            self.stack.unwind()

        self.assertEqual(self.order, ["gamma", "beta", "alpha"])
        self.assertEqual(
            tuple(label for label, _ in caught.exception.failures),
            ("fiber-a:gamma", "fiber-a:alpha"),
        )
        self.assertEqual(
            tuple(exc for _, exc in caught.exception.failures), (last, first)
        )

    def test_unnamed_failing_effect_is_still_attributed_to_its_owner(self) -> None:
        self.stack.add(self.failing("alpha", RuntimeError("boom")))

        with self.assertRaises(DisposalErrors) as caught:
            self.stack.unwind()

        self.assertEqual(
            tuple(label for label, _ in caught.exception.failures), ("fiber-a:effect",)
        )

    # -- early disposers -------------------------------------------------

    def test_early_disposer_removes_only_that_effect(self) -> None:
        self.stack.add(self.record("alpha"), label="alpha")
        remove_beta = self.stack.add(self.record("beta"), label="beta")
        self.stack.add(self.record("gamma"), label="gamma")

        remove_beta()

        self.assertEqual(self.order, ["beta"])
        self.assertEqual(self.stack.live, ("alpha", "gamma"))
        self.assertEqual(len(self.stack), 2)

        self.stack.unwind()

        # "beta" is not disposed a second time by the unwind.
        self.assertEqual(self.order, ["beta", "gamma", "alpha"])

    def test_early_disposer_is_idempotent(self) -> None:
        calls: list[int] = []
        remove = self.stack.add(lambda: calls.append(1), label="alpha")

        remove()
        remove()
        remove()

        self.assertEqual(calls, [1])
        self.assertEqual(len(self.stack), 0)
        self.assertEqual(self.stack.live, ())

    def test_early_disposer_after_unwind_does_not_dispose_twice(self) -> None:
        remove = self.stack.add(self.record("alpha"), label="alpha")

        self.stack.unwind()
        remove()

        self.assertEqual(self.order, ["alpha"])
        self.assertEqual(len(self.stack), 0)

    def test_early_disposer_called_during_unwind_disposes_once(self) -> None:
        remove_inner = self.stack.add(self.record("inner"), label="inner")

        def outer() -> None:
            self.order.append("outer")
            remove_inner()

        self.stack.add(outer, label="outer")

        self.stack.unwind()

        # The nested early disposal removed "inner" from the stack, so the unwind loop
        # did not run it a second time.
        self.assertEqual(self.order, ["outer", "inner"])
        self.assertEqual(len(self.stack), 0)

    def test_early_disposer_failure_propagates_unwrapped(self) -> None:
        error = RuntimeError("boom")
        remove = self.stack.add(self.failing("alpha", error), label="alpha")
        self.stack.add(self.record("beta"), label="beta")

        with self.assertRaises(RuntimeError) as caught:
            remove()

        # A direct disposal is not an unwind: nothing is aggregated.
        self.assertIs(caught.exception, error)
        self.assertEqual(self.stack.live, ("beta",))

        self.stack.unwind()

        # The already-attempted effect is gone, so the unwind does not retry it.
        self.assertEqual(self.order, ["alpha", "beta"])

    def test_early_disposer_targets_its_own_effect_when_effects_look_alike(self) -> None:
        """Two effects can share a callable and a label without aliasing each other."""
        calls: list[int] = []

        def dispose() -> None:
            calls.append(1)

        self.stack.add(dispose, label="same")
        remove_second = self.stack.add(dispose, label="same")

        remove_second()
        self.assertEqual(calls, [1])
        self.assertEqual(len(self.stack), 1)

        self.stack.unwind()

        # If the early disposer had dropped the *first* effect from the stack, the
        # remaining one would be flagged disposed and skipped, leaving only one call.
        self.assertEqual(calls, [1, 1])

    # -- registering while unwinding -------------------------------------

    def test_add_during_unwind_disposes_immediately_and_returns_a_noop(self) -> None:
        late_disposer: list = []
        self.stack.add(self.record("early"), label="early")

        def adder() -> None:
            self.order.append("adder")
            late_disposer.append(self.stack.add(self.record("late"), label="late"))

        self.stack.add(adder, label="adder")

        self.stack.unwind()

        # "late" ran the moment it was filed -- before the older "early" effect -- and was
        # never tracked on the dying stack.
        self.assertEqual(self.order, ["adder", "late", "early"])
        self.assertEqual(len(self.stack), 0)
        self.assertEqual(self.stack.live, ())

        self.assertIsNone(late_disposer[0]())
        self.assertEqual(self.order, ["adder", "late", "early"])

    def test_failure_of_an_effect_added_during_unwind_is_collected(self) -> None:
        error = RuntimeError("late boom")
        self.stack.add(self.record("early"), label="early")

        def adder() -> None:
            self.order.append("adder")
            self.stack.add(self.failing("late", error), label="late")

        self.stack.add(adder, label="adder")

        with self.assertRaises(DisposalErrors) as caught:
            self.stack.unwind()

        # The immediate disposal raises inside the running effect, so it is attributed to
        # that effect -- and the rest of the stack still unwinds.
        self.assertEqual(self.order, ["adder", "late", "early"])
        self.assertEqual(caught.exception.failures, (("fiber-a:adder", error),))

    def test_stack_is_reusable_after_unwind(self) -> None:
        self.stack.add(self.failing("alpha", RuntimeError("boom")), label="alpha")
        with self.assertRaises(DisposalErrors):
            self.stack.unwind()

        # The unwinding flag is cleared, so a fresh effect is tracked rather than being
        # disposed on the spot.
        remove = self.stack.add(self.record("beta"), label="beta")
        self.assertEqual(self.stack.live, ("beta",))
        self.assertEqual(self.order, ["alpha"])

        remove()
        self.assertEqual(self.order, ["alpha", "beta"])
        self.assertEqual(len(self.stack), 0)

    # -- BaseException ---------------------------------------------------

    def test_base_exception_propagates_out_of_unwind(self) -> None:
        abort = _Abort("environment gone")
        self.stack.add(self.record("alpha"), label="alpha")
        self.stack.add(self.failing("beta", abort), label="beta")
        self.stack.add(self.record("gamma"), label="gamma")

        with self.assertRaises(_Abort) as caught:
            self.stack.unwind()

        self.assertIs(caught.exception, abort)
        self.assertNotIsInstance(caught.exception, DisposalErrors)
        # The abort reaches the caller untouched, but the stack is drained first: stopping at the
        # failure would leave "alpha" installed with nothing left to remove it, and a leaked
        # registration -- a listener, or a sandbox process group -- outliving its plugin is worse
        # than a delayed abort.
        self.assertEqual(self.order, ["gamma", "beta", "alpha"])
        self.assertEqual(self.stack.live, ())
        self.assertEqual(len(self.stack), 0)

    def test_base_exception_is_not_converted_into_disposal_errors(self) -> None:
        self.stack.add(self.record("alpha"), label="alpha")
        self.stack.add(self.failing("beta", _Abort("environment gone")), label="beta")
        self.stack.add(self.failing("gamma", RuntimeError("boom")), label="gamma")

        # "gamma" fails with an ordinary Exception and is collected, then "beta" aborts. The abort
        # wins over the collected DisposalErrors and is never wrapped, and "alpha" still runs.
        with self.assertRaises(_Abort):
            self.stack.unwind()

        self.assertEqual(self.order, ["gamma", "beta", "alpha"])
        self.assertEqual(len(self.stack), 0)

    def test_only_the_first_abort_wins_and_the_rest_still_unwind(self) -> None:
        first = _Abort("environment gone")
        self.stack.add(self.record("alpha"), label="alpha")
        self.stack.add(self.failing("beta", first), label="beta")
        self.stack.add(self.failing("gamma", _Abort("second abort")), label="gamma")

        with self.assertRaises(_Abort) as caught:
            self.stack.unwind()

        # "gamma" aborts first in LIFO order, so its abort is the one that escapes.
        self.assertEqual(str(caught.exception), "second abort")
        self.assertEqual(self.order, ["gamma", "beta", "alpha"])

    def test_stack_is_reusable_after_a_base_exception(self) -> None:
        self.stack.add(self.failing("alpha", _Abort("environment gone")), label="alpha")

        with self.assertRaises(_Abort):
            self.stack.unwind()

        # The `finally` cleared the unwinding flag, so the stack still tracks new effects
        # instead of disposing them immediately.
        self.stack.add(self.record("beta"), label="beta")
        self.assertEqual(self.stack.live, ("beta",))
        self.assertEqual(self.order, ["alpha"])

        self.stack.unwind()
        self.assertEqual(self.order, ["alpha", "beta"])


if __name__ == "__main__":
    unittest.main()
