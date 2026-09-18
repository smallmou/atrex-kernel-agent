"""Behavioural tests for fiber epochs and the settle loop.

Load order is never declared anywhere in this container: it falls out of each fiber's **epoch**,
the tuple of live registration serials behind its declared injections. These tests pin the
observable consequences of that claim rather than the mechanism itself:

* a row whose required service is absent waits in ``PENDING`` -- a steady state, not a
  failure -- and says which service it is waiting for;
* the row activates when the provider arrives, in whatever order the rows were mounted,
  even three levels deep, inside a single :meth:`Root.settle` call;
* losing the provider returns the row to ``PENDING`` and unwinds its effects;
* a *different* fiber providing the same name is a new epoch, so the dependent reloads and
  sees the new implementation -- the same mechanism an optional provider appearing uses
  (a live serial versus ``0`` in ``compute_epoch``);
* a registration made inside another fiber's ``apply`` is queued instead of recursing, so a
  plugin never observes a half-loaded neighbour;
* two rows undoing each other exhaust the round budget and raise instead of spinning.

Plugins here are :class:`PluginDeclaration` objects built in-test: the declaration is a
frozen dataclass, so a fiber does not need a module on disk. Every plugin records what it
did in ``self.receipt`` so lifecycle order is asserted, not assumed.
"""

from __future__ import annotations

import io
import unittest
from typing import Any, Callable, Mapping, Sequence

from aka.core.declaration import PluginDeclaration
from aka.core.errors import ConfigError, SettleDivergence
from aka.core.fiber import INACTIVE, FiberState, Root
from aka.core.internal import INTERNAL_FIBER
from aka.core.keys import ServiceKey

# Seams under test. These are deliberately not registered in ``aka.seams.SEAMS``: a fiber
# provides a ``ServiceKey`` token directly, and the core never consults the seam table.
ALPHA = ServiceKey(name="alpha_svc", definition="Alpha")
BETA = ServiceKey(name="beta_svc", definition="Beta")


def _declaration(
    name: str,
    apply: Callable[..., Any],
    *,
    inject: tuple[str, ...] = (),
    optional_inject: tuple[str, ...] = (),
    provide: tuple[str, ...] = (),
) -> PluginDeclaration:
    """One validated declaration, assembled without a module on disk."""
    return PluginDeclaration(
        name=name,
        module=f"test.fiber_epoch.{name.replace('-', '_')}",
        apply=apply,
        config_schema=None,
        defaults={},
        inject=inject,
        optional_inject=optional_inject,
        provide=provide,
        interpolate=(),
    )


class _FiberCase(unittest.TestCase):
    """Shared root, receipt, and the row builder every test composes."""

    settle_rounds = 16

    def setUp(self) -> None:
        self.stderr = io.StringIO()
        self.receipt: list[str] = []
        self.observed: list[tuple[Any, ...]] = []
        self.root = Root(settle_rounds=self.settle_rounds, stderr=self.stderr)

    def tearDown(self) -> None:
        self.root.dispose()

    # -- observation -----------------------------------------------------

    def transitions_for(self, entry_id: str) -> list[str]:
        """The observed state path of one row, as ``previous->current`` steps."""
        return [
            f"{transition.previous}->{transition.current}"
            for transition in self.root.transitions
            if transition.entry_id == entry_id
        ]

    def epoch_for(self, *names: str) -> tuple[int, ...]:
        """The epoch fragment a dependent should carry for ``names``, currently resolved.

        Spelled as a helper rather than inline so the tests assert the *relationship* -- "this
        row's epoch is the live registration of what it injects" -- instead of restating how an
        epoch is composed.
        """
        fragment: list[int] = []
        for name in names:
            impl = self.root.realm.resolve(name)
            assert impl is not None, f"no live provider for {name}"
            fragment.append(impl.serial)
        return tuple(fragment)

    # -- row builder -----------------------------------------------------

    def row(
        self,
        name: str,
        *,
        inject: tuple[str, ...] = (),
        optional_inject: tuple[str, ...] = (),
        provides: tuple[ServiceKey, Any] | None = None,
        reads: Sequence[str] = (),
    ) -> PluginDeclaration:
        """A plugin that records its load, reads ``reads``, and publishes ``provides``.

        The returned teardown callable is what ``apply`` hands back, so every row also
        exercises "apply's return value is the last effect".
        """

        def apply(ctx: Any, config: Mapping[str, Any]) -> Callable[[], None]:
            self.receipt.append(f"apply:{name}")
            if reads:
                self.observed.append(tuple(getattr(ctx, read) for read in reads))
            if provides is not None:
                ctx.provide(provides[0], provides[1])

            def unload() -> None:
                self.receipt.append(f"unload:{name}")

            return unload

        return _declaration(
            name,
            apply,
            inject=inject,
            optional_inject=optional_inject,
            provide=() if provides is None else (provides[0].name,),
        )


class PendingWhileUnsatisfiedTest(_FiberCase):
    def test_absent_required_injection_stays_pending_and_names_the_service(self) -> None:
        consumer = self.root.mount(
            self.row("consumer", inject=("alpha_svc",)), entry_id="consumer"
        )

        self.root.settle()

        self.assertIs(consumer.state, FiberState.PENDING)
        self.assertIs(consumer.epoch, INACTIVE)
        self.assertEqual(consumer.unsatisfied, ("alpha_svc",))
        self.assertEqual(self.root.pending, (("consumer", ("alpha_svc",)),))
        # Waiting is a steady state: no failure, no error report, and apply never ran.
        self.assertEqual(self.root.active, ())
        self.assertEqual(self.root.failed, ())
        self.assertEqual(self.root.errors, [])
        self.assertEqual(self.receipt, [])
        self.assertEqual(self.stderr.getvalue(), "")

    def test_settling_again_does_not_disturb_a_waiting_row(self) -> None:
        consumer = self.root.mount(
            self.row("consumer", inject=("alpha_svc", "beta_svc")), entry_id="consumer"
        )

        self.root.settle()
        self.root.settle()

        self.assertIs(consumer.state, FiberState.PENDING)
        self.assertEqual(consumer.unsatisfied, ("alpha_svc", "beta_svc"))
        self.assertEqual(self.receipt, [])

    def test_provider_arriving_activates_the_dependent(self) -> None:
        consumer = self.root.mount(
            self.row("consumer", inject=("alpha_svc",), reads=("alpha_svc",)),
            entry_id="consumer",
        )
        self.root.settle()
        self.assertIs(consumer.state, FiberState.PENDING)

        provider = self.root.mount(
            self.row("provider", provides=(ALPHA, "alpha-1")), entry_id="provider"
        )
        self.root.settle()

        self.assertIs(provider.state, FiberState.ACTIVE)
        self.assertIs(consumer.state, FiberState.ACTIVE)
        self.assertEqual(self.root.active, ("consumer", "provider"))
        self.assertEqual(self.root.pending, ())
        self.assertEqual(consumer.epoch, self.epoch_for("alpha_svc"))
        self.assertEqual(consumer.unsatisfied, ())
        self.assertEqual(self.observed, [("alpha-1",)])
        self.assertEqual(self.receipt, ["apply:provider", "apply:consumer"])

    def test_losing_the_provider_returns_the_dependent_to_pending(self) -> None:
        provider = self.root.mount(
            self.row("provider", provides=(ALPHA, "alpha-1")), entry_id="provider"
        )
        consumer = self.root.mount(
            self.row("consumer", inject=("alpha_svc",), reads=("alpha_svc",)),
            entry_id="consumer",
        )
        self.root.settle()
        self.assertEqual(self.root.active, ("consumer", "provider"))

        provider.dispose()
        self.root.settle()

        self.assertIs(provider.state, FiberState.DISPOSED)
        self.assertNotIn("provider", self.root.fibers)
        # A vanished dependency is not the dependent's fault.
        self.assertIs(consumer.state, FiberState.PENDING)
        self.assertIsNone(consumer.error)
        self.assertIs(consumer.epoch, INACTIVE)
        self.assertEqual(self.root.failed, ())
        self.assertEqual(self.root.errors, [])
        self.assertEqual(self.root.pending, (("consumer", ("alpha_svc",)),))
        # ...and its effects are gone, so nothing it published outlives the epoch.
        self.assertEqual(len(consumer.effects), 0)
        self.assertEqual(self.root.live_effects, 0)
        self.assertEqual(
            self.receipt,
            [
                "apply:provider",
                "apply:consumer",
                "unload:provider",
                "unload:consumer",
            ],
        )


class EpochChangeTest(_FiberCase):
    def test_a_different_fiber_for_the_same_name_reloads_the_dependent(self) -> None:
        first = self.root.mount(
            self.row("provider-a", provides=(ALPHA, "alpha-1")), entry_id="provider-a"
        )
        consumer = self.root.mount(
            self.row("consumer", inject=("alpha_svc",), reads=("alpha_svc",)),
            entry_id="consumer",
        )
        self.root.settle()
        self.assertEqual(self.observed, [("alpha-1",)])
        first_epoch = self.epoch_for("alpha_svc")
        self.assertEqual(consumer.epoch, first_epoch)

        first.dispose()
        second = self.root.mount(
            self.row("provider-b", provides=(ALPHA, "alpha-2")), entry_id="provider-b"
        )
        self.root.settle()

        self.assertNotEqual(second.uid, first.uid)
        self.assertIs(consumer.state, FiberState.ACTIVE)
        self.assertEqual(consumer.epoch, self.epoch_for("alpha_svc"))
        self.assertNotEqual(consumer.epoch, first_epoch)
        # apply ran twice, and the second run saw the new implementation.
        self.assertEqual(self.receipt.count("apply:consumer"), 2)
        self.assertEqual(self.receipt.count("unload:consumer"), 1)
        self.assertEqual(self.observed, [("alpha-1",), ("alpha-2",)])
        self.assertEqual(
            self.receipt,
            [
                "apply:provider-a",
                "apply:consumer",
                "unload:provider-a",
                "unload:consumer",
                "apply:provider-b",
                "apply:consumer",
            ],
        )
        self.assertEqual(
            self.transitions_for("consumer"),
            [
                "pending->loading",
                "loading->active",
                "active->unloading",
                "unloading->pending",
                "pending->loading",
                "loading->active",
            ],
        )

    def test_optional_provider_appearing_changes_the_epoch_and_reloads(self) -> None:
        provider = self.root.mount(
            self.row("provider", provides=(ALPHA, "alpha-1")), entry_id="provider"
        )
        consumer = self.root.mount(
            self.row(
                "consumer",
                inject=("alpha_svc",),
                optional_inject=("beta_svc",),
                reads=("alpha_svc", "beta_svc"),
            ),
            entry_id="consumer",
        )
        self.root.settle()

        # An absent optional provider contributes 0, not nothing: the row is active.
        self.assertIs(consumer.state, FiberState.ACTIVE)
        self.assertEqual(consumer.epoch, (provider.uid, 0))
        self.assertEqual(consumer.unsatisfied, ())
        self.assertEqual(self.observed, [("alpha-1", None)])

        extra = self.root.mount(
            self.row("extra", provides=(BETA, "beta-1")), entry_id="extra"
        )
        self.root.settle()

        self.assertIs(consumer.state, FiberState.ACTIVE)
        self.assertEqual(
            consumer.epoch,
            self.epoch_for("alpha_svc") + self.epoch_for("beta_svc"),
        )
        # The optional swap reloaded the row instead of leaving a stale None behind.
        self.assertEqual(self.observed, [("alpha-1", None), ("alpha-1", "beta-1")])
        self.assertEqual(self.receipt.count("apply:consumer"), 2)
        self.assertEqual(self.receipt.count("unload:consumer"), 1)
        # One refresh unloaded and reloaded the row against the new epoch.
        self.assertEqual(
            self.receipt[-3:], ["apply:extra", "unload:consumer", "apply:consumer"]
        )

    def test_optional_provider_vanishing_reloads_without_failing(self) -> None:
        extra = self.root.mount(
            self.row("extra", provides=(BETA, "beta-1")), entry_id="extra"
        )
        consumer = self.root.mount(
            self.row("consumer", optional_inject=("beta_svc",), reads=("beta_svc",)),
            entry_id="consumer",
        )
        self.root.settle()
        self.assertEqual(consumer.epoch, self.epoch_for("beta_svc"))

        extra.dispose()
        self.root.settle()

        self.assertIs(consumer.state, FiberState.ACTIVE)
        self.assertEqual(consumer.epoch, (0,))
        self.assertEqual(self.observed, [("beta-1",), (None,)])


class ChainTest(_FiberCase):
    def test_three_deep_chain_activates_within_one_settle(self) -> None:
        # Mounted leaf-first, so nothing but the epoch can produce a working load order.
        third = self.root.mount(
            self.row("c", inject=("beta_svc",), reads=("beta_svc",)), entry_id="c"
        )
        second = self.root.mount(
            self.row(
                "b",
                inject=("alpha_svc",),
                reads=("alpha_svc",),
                provides=(BETA, "beta-1"),
            ),
            entry_id="b",
        )
        first = self.root.mount(
            self.row("a", provides=(ALPHA, "alpha-1")), entry_id="a"
        )
        self.assertEqual(
            self.root.pending,
            (("a", ()), ("b", ("alpha_svc",)), ("c", ("beta_svc",))),
        )

        self.root.settle()

        self.assertEqual(self.root.active, ("a", "b", "c"))
        self.assertEqual(self.root.pending, ())
        self.assertEqual(self.receipt, ["apply:a", "apply:b", "apply:c"])
        self.assertEqual(first.epoch, ())
        self.assertEqual(second.epoch, self.epoch_for("alpha_svc"))
        self.assertEqual(third.epoch, self.epoch_for("beta_svc"))
        self.assertEqual(self.observed, [("alpha-1",), ("beta-1",)])

    def test_provide_inside_apply_is_queued_rather_than_recursing(self) -> None:
        witness: dict[str, Any] = {}

        def provider_apply(ctx: Any, config: Mapping[str, Any]) -> None:
            self.receipt.append("apply:provider")
            ctx.provide(ALPHA, "alpha-1")
            # The registration is visible immediately...
            witness["resolved"] = ctx.get("alpha_svc")
            # ...but the dependent must not have been loaded re-entrantly.
            witness["state"] = self.root.state_of("consumer")
            witness["receipt"] = tuple(self.receipt)

        consumer = self.root.mount(
            self.row("consumer", inject=("alpha_svc",), reads=("alpha_svc",)),
            entry_id="consumer",
        )
        self.root.mount(
            _declaration("provider", provider_apply, provide=("alpha_svc",)),
            entry_id="provider",
        )

        self.root.settle()

        self.assertEqual(witness["resolved"], "alpha-1")
        self.assertIs(witness["state"], FiberState.PENDING)
        self.assertEqual(witness["receipt"], ("apply:provider",))
        self.assertIs(consumer.state, FiberState.ACTIVE)
        self.assertEqual(self.receipt, ["apply:provider", "apply:consumer"])

    def test_child_mounted_inside_apply_settles_in_the_same_settle(self) -> None:
        witness: dict[str, Any] = {}
        child_declaration = self.row("child", provides=(BETA, "beta-1"))

        def parent_apply(ctx: Any, config: Mapping[str, Any]) -> Callable[[], None]:
            self.receipt.append("apply:parent")
            child = ctx.plugin(child_declaration, entry_id="child")
            # The nested settle() is a no-op while the outer one is draining.
            witness["child_state"] = child.state
            witness["receipt"] = tuple(self.receipt)
            return lambda: self.receipt.append("unload:parent")

        parent = self.root.mount(_declaration("parent", parent_apply), entry_id="parent")
        consumer = self.root.mount(
            self.row("consumer", inject=("beta_svc",), reads=("beta_svc",)),
            entry_id="consumer",
        )

        self.root.settle()

        self.assertIs(witness["child_state"], FiberState.PENDING)
        self.assertEqual(witness["receipt"], ("apply:parent",))
        self.assertEqual(self.root.active, ("child", "consumer", "parent"))
        self.assertEqual(self.observed, [("beta-1",)])

        # The child belongs to the parent's subtree, so it goes when the parent goes. Withdrawing
        # the child's service wakes its dependent immediately, so the consumer is unloaded as part
        # of that disposal rather than surviving with a provider that is already gone.
        parent.dispose()
        self.root.settle()

        self.assertNotIn("child", self.root.fibers)
        self.assertIs(consumer.state, FiberState.PENDING)
        self.assertEqual(
            self.receipt,
            [
                "apply:parent",
                "apply:child",
                "apply:consumer",
                "unload:child",
                "unload:consumer",
                "unload:parent",
            ],
        )
        self.assertEqual(self.root.live_effects, 0)


class SettleDivergenceTest(_FiberCase):
    def test_provide_dispose_pingpong_exhausts_the_round_budget(self) -> None:
        root = Root(settle_rounds=4, stderr=self.stderr)
        self.addCleanup(root.dispose)

        def reaper_apply(ctx: Any, config: Mapping[str, Any]) -> None:
            self.receipt.append("apply:reaper")
            if ctx.get("alpha_svc") is not None:
                # Undo the provider's registration by forcing it through another cycle:
                # it re-provides on load, this row tears it down again.
                root.fibers["provider"].restart()

        root.mount(
            self.row("provider", provides=(ALPHA, "alpha-1")), entry_id="provider"
        )
        root.mount(
            _declaration("reaper", reaper_apply, optional_inject=("alpha_svc",)),
            entry_id="reaper",
        )

        with self.assertRaises(SettleDivergence) as caught:
            root.settle()

        diverged = caught.exception
        self.assertEqual(diverged.rounds, 4)
        self.assertIn("reaper", diverged.entry_ids)
        self.assertIn("reaper", str(diverged))
        self.assertIn("did not settle within 4 rounds", str(diverged))
        # It really was a fight, not a queue that merely failed to drain.
        self.assertGreaterEqual(self.receipt.count("apply:provider"), 2)
        self.assertGreaterEqual(self.receipt.count("apply:reaper"), 2)


class EffectLifecycleTest(_FiberCase):
    def test_apply_return_value_is_the_final_effect_and_runs_on_unload(self) -> None:
        def apply(ctx: Any, config: Mapping[str, Any]) -> Callable[[], None]:
            ctx.effect(lambda: self.receipt.append("inner"), label="inner")
            return lambda: self.receipt.append("returned")

        fiber = self.root.mount(_declaration("row", apply), entry_id="row")
        self.root.settle()

        self.assertIs(fiber.state, FiberState.ACTIVE)
        self.assertEqual(fiber.effects.live, ("inner", "apply() return"))
        self.assertEqual(self.receipt, [])

        fiber.dispose()

        # Registered last, so LIFO unwind runs it first.
        self.assertEqual(self.receipt, ["returned", "inner"])
        self.assertEqual(self.root.live_effects, 0)

    def test_row_that_declares_provide_without_registering_it_fails(self) -> None:
        def apply(ctx: Any, config: Mapping[str, Any]) -> None:
            self.receipt.append("apply:liar")
            ctx.effect(lambda: self.receipt.append("unload:liar"), label="stray")

        fiber = self.root.mount(
            _declaration("liar", apply, provide=("alpha_svc",)), entry_id="liar"
        )
        self.root.settle()

        self.assertIs(fiber.state, FiberState.FAILED)
        self.assertIsInstance(fiber.error, ConfigError)
        self.assertIn(
            "declared provide alpha_svc but never registered it", str(fiber.error)
        )
        self.assertEqual(self.root.failed, (("liar", f"ConfigError: {fiber.error}"),))
        self.assertEqual(self.root.active, ())
        # A failed load leaves nothing behind, and is reported once.
        self.assertEqual(self.receipt, ["apply:liar", "unload:liar"])
        self.assertEqual(len(fiber.effects), 0)
        self.assertEqual([report.source for report in self.root.errors], ["entry liar"])
        self.assertIn("liar", self.stderr.getvalue())
        self.assertEqual(
            self.transitions_for("liar"), ["pending->loading", "loading->failed"]
        )

        # A failed row is terminal: settling again does not retry it.
        self.root.settle()
        self.assertEqual(self.receipt, ["apply:liar", "unload:liar"])


class RestartTest(_FiberCase):
    def test_restart_forces_one_unload_load_cycle(self) -> None:
        fiber = self.root.mount(self.row("row"), entry_id="row")
        self.root.settle()
        self.assertEqual(self.receipt, ["apply:row"])
        uid = fiber.uid

        fiber.restart()

        self.assertIs(fiber.state, FiberState.PENDING)
        self.assertIs(fiber.epoch, INACTIVE)
        self.assertEqual(self.receipt, ["apply:row", "unload:row"])

        self.root.settle()

        self.assertIs(fiber.state, FiberState.ACTIVE)
        self.assertEqual(fiber.epoch, ())
        self.assertEqual(fiber.uid, uid)
        self.assertEqual(self.receipt, ["apply:row", "unload:row", "apply:row"])

    def test_restarting_a_provider_reloads_its_dependents(self) -> None:
        published: list[str] = []

        def provider_apply(ctx: Any, config: Mapping[str, Any]) -> None:
            published.append(f"value-{len(published) + 1}")
            ctx.provide(ALPHA, published[-1])

        provider = self.root.mount(
            _declaration("provider", provider_apply, provide=("alpha_svc",)),
            entry_id="provider",
        )
        consumer = self.root.mount(
            self.row("consumer", inject=("alpha_svc",), reads=("alpha_svc",)),
            entry_id="consumer",
        )
        self.root.settle()
        self.assertEqual(self.observed, [("value-1",)])

        provider.restart()
        self.root.settle()

        # The provider fiber is the same one, so the reload rides on the registration serial:
        # the dependent must not keep the withdrawn first value.
        self.assertIs(consumer.state, FiberState.ACTIVE)
        self.assertEqual(consumer.epoch, self.epoch_for("alpha_svc"))
        self.assertEqual(self.observed, [("value-1",), ("value-2",)])
        self.assertEqual(self.receipt.count("apply:consumer"), 2)
        self.assertEqual(self.receipt.count("unload:consumer"), 1)

    def test_restart_requested_during_load_still_unloads_before_reloading(self) -> None:
        """``restart()`` promises "one unload/load cycle" whatever the current state is.

        A row that re-arms itself from inside its own ``apply`` is mid-transition (``LOADING``).
        Unloading right then would tear down the load that is still running, so the request is
        honoured once that load settles -- never by loading a second time over the first load's
        still-live effects.
        """
        log: list[str] = []

        def apply(ctx: Any, config: Mapping[str, Any]) -> None:
            log.append("apply")
            ctx.effect(lambda: log.append("teardown"), label="t")
            if log.count("apply") == 1:
                ctx.fiber.restart()

        fiber = self.root.mount(_declaration("row", apply), entry_id="row")
        self.root.settle()

        self.assertIs(fiber.state, FiberState.ACTIVE)
        self.assertEqual(log, ["apply", "teardown", "apply"])
        self.assertEqual(fiber.effects.live, ("t",))
        self.assertNotIn("active->loading", self.transitions_for("row"))

    def test_restart_retries_a_failed_row(self) -> None:
        """A config patch is the documented reason to restart, and it follows a failure.

        ``refresh()`` returns early for ``FAILED``, so ``restart()`` clears that state before
        re-enqueuing; otherwise the retry its docstring invites could never happen.
        """
        attempts: list[int] = []

        def apply(ctx: Any, config: Mapping[str, Any]) -> None:
            attempts.append(len(attempts) + 1)
            if len(attempts) == 1:
                raise RuntimeError("first load fails")

        fiber = self.root.mount(_declaration("flaky", apply), entry_id="flaky")
        self.root.settle()
        self.assertIs(fiber.state, FiberState.FAILED)
        self.assertEqual(attempts, [1])

        fiber.restart()
        self.root.settle()

        self.assertEqual(attempts, [1, 2])
        self.assertIs(fiber.state, FiberState.ACTIVE)
        self.assertIsNone(fiber.error)


class RootTeardownTest(_FiberCase):
    def test_dispose_tears_rows_down_in_reverse_mount_order(self) -> None:
        for name in ("a", "b", "c"):
            self.root.mount(self.row(name), entry_id=name)
        self.root.settle()
        self.root.bus.on(INTERNAL_FIBER, lambda transition: None, owner="observer")
        self.assertEqual(self.receipt, ["apply:a", "apply:b", "apply:c"])
        self.assertEqual(self.root.live_effects, 3)

        self.root.dispose()

        self.assertEqual(
            self.receipt,
            ["apply:a", "apply:b", "apply:c", "unload:c", "unload:b", "unload:a"],
        )
        self.assertEqual(self.root.live_effects, 0)
        self.assertEqual(self.root.fibers, {})
        self.assertEqual(self.root.active, ())
        self.assertTrue(self.root.bus.closed)
        self.assertFalse(self.root.bus.has_listeners(INTERNAL_FIBER))
        self.assertEqual(self.root.errors, [])

    def test_dispose_unwinds_a_dependent_before_its_provider(self) -> None:
        self.root.mount(
            self.row("provider", provides=(ALPHA, "alpha-1")), entry_id="provider"
        )
        self.root.mount(
            self.row("consumer", inject=("alpha_svc",), reads=("alpha_svc",)),
            entry_id="consumer",
        )
        self.root.settle()

        self.root.dispose()

        # Reverse mount order happens to be dependency order here, and the dependent's
        # teardown must not observe a provider that is already gone.
        self.assertEqual(self.receipt[-2:], ["unload:consumer", "unload:provider"])
        self.assertEqual(self.root.live_effects, 0)

    def test_duplicate_composition_row_id_at_mount_raises_config_error(self) -> None:
        self.root.mount(self.row("row"), entry_id="row")

        with self.assertRaises(ConfigError) as caught:
            self.root.mount(self.row("row"), entry_id="row")

        self.assertEqual(caught.exception.entry_id, "row")
        self.assertIn("duplicate composition row id", str(caught.exception))
        # The rejected mount left no fiber behind, so the tree still settles cleanly.
        self.assertEqual(len(self.root.fibers), 1)
        self.root.settle()
        self.assertEqual(self.receipt, ["apply:row"])


if __name__ == "__main__":
    unittest.main()
