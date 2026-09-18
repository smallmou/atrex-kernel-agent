"""Behavioural tests for isolation realms, named provider registries, and scopes.

Three merge rules live in ``registry.py`` and ``scope.py``, and every seam in the product
inherits one of them. These tests pin the observable consequences rather than the mechanism:

* **Realm** -- a single-cardinality service has exactly one provider *per realm*. Two rows
  claiming it in one realm is an attributable failure naming both rows; the same two rows
  claiming it in realms that isolate the name both succeed and neither can see the other.
  A realm that does *not* isolate a name delegates upward, so a subtree reads its parent's
  implementation and cannot shadow it by accident.
* **ProviderRegistry** -- the "many implementations coexist, keyed by name, consulted in a
  declared order" shape that roughly ten seams embed: ordering is ``(order, name)``, a
  duplicate name is refused, and disposing one registration leaves the rest alone.
* **Scope** -- the "global layer plus a scope chain" shape shared by tools, prompt sections,
  skills, and gates: nearest layer wins a duplicate name, and ``restrict`` filters what a
  layer *inherits* without ever touching what that layer contributed itself.

Rows here are :class:`PluginDeclaration` objects built in-test -- the declaration is a frozen
dataclass, so a fiber does not need a module on disk. Realm writes go through
``Fiber.provide`` (the path ``ctx.provide`` uses) so failures surface to the caller instead of
being swallowed into a failed load.
"""

from __future__ import annotations

import io
import unittest
from typing import Any, Callable, Mapping

from aka.core.declaration import PluginDeclaration
from aka.core.errors import DuplicateProvide
from aka.core.fiber import Fiber, FiberState, Root
from aka.core.keys import ServiceKey
from aka.core.registry import Impl, ProviderRegistry, Realm
from aka.core.scope import Scope, ScopedEntry

# Seams under test. Deliberately not registered in ``aka.seams.SEAMS``: a fiber provides a
# ``ServiceKey`` token directly and the realm never consults the seam table.
DRIVER = ServiceKey(name="driver_svc", definition="CampaignDriver")
NUMERICS = ServiceKey(name="numerics_svc", definition="Numerics")
GATES = ServiceKey(
    name="gates_svc", definition="GateRegistry", cardinality="registry"
)


def _declaration(name: str, apply: Callable[..., Any]) -> PluginDeclaration:
    """One declaration, assembled without a module on disk."""
    return PluginDeclaration(
        name=name,
        module=f"test.isolation.{name.replace('-', '_')}",
        apply=apply,
        config_schema=None,
        defaults={},
        inject=(),
        optional_inject=(),
        provide=(),
        interpolate=(),
    )


class _RealmCase(unittest.TestCase):
    """A live root plus idle rows whose only job is to own realm registrations."""

    def setUp(self) -> None:
        self.stderr = io.StringIO()
        self.root = Root(stderr=self.stderr)

    def tearDown(self) -> None:
        self.root.dispose()

    def row(self, entry_id: str, *, isolate: tuple[str, ...] = ()) -> Fiber:
        """An active row that publishes nothing on its own.

        The test body calls ``fiber.provide`` afterwards, which is exactly what
        ``ctx.provide`` does, so a rejected registration raises here instead of being
        recorded as a failed load.
        """

        def apply(ctx: Any, config: Mapping[str, Any]) -> None:
            return None

        fiber = self.root.mount(
            _declaration(entry_id, apply), entry_id=entry_id, isolate=isolate
        )
        self.root.settle()
        self.assertIs(fiber.state, FiberState.ACTIVE)
        return fiber


class RealmSingleProviderTest(_RealmCase):
    def test_two_rows_providing_one_service_in_one_realm_name_both_entries(self) -> None:
        alpha = self.row("alpha")
        beta = self.row("beta")
        self.assertIs(alpha.realm, self.root.realm)
        self.assertIs(beta.realm, self.root.realm)

        alpha.provide(DRIVER, "alpha-driver")

        with self.assertRaises(DuplicateProvide) as caught:
            beta.provide(DRIVER, "beta-driver")

        collision = caught.exception
        self.assertEqual(collision.name, "driver_svc")
        self.assertEqual(collision.owner, "alpha")
        self.assertEqual(collision.claimant, "beta")
        # Attributable without already knowing the plugin tree: both rows are in the text.
        message = str(collision)
        self.assertIn('entry "alpha"', message)
        self.assertIn('entry "beta"', message)
        self.assertIn('service "driver_svc"', message)
        self.assertIn("same isolation realm", message)
        # The incumbent is untouched and the loser filed nothing.
        resolved = self.root.realm.resolve("driver_svc")
        assert resolved is not None
        self.assertEqual(resolved.value, "alpha-driver")
        self.assertEqual(resolved.entry_id, "alpha")
        self.assertEqual(beta.provided, set())
        self.assertEqual(len(beta.effects), 0)

    def test_disposing_the_incumbent_frees_the_name_for_the_next_row(self) -> None:
        alpha = self.row("alpha")
        beta = self.row("beta")
        undo = alpha.provide(DRIVER, "alpha-driver")

        undo()

        self.assertIsNone(self.root.realm.resolve("driver_svc"))
        self.assertEqual(alpha.provided, set())
        beta.provide(DRIVER, "beta-driver")
        resolved = self.root.realm.resolve("driver_svc")
        assert resolved is not None
        self.assertEqual(resolved.value, "beta-driver")
        self.assertIs(resolved.fiber, beta)

    def test_distinct_names_coexist_in_one_realm(self) -> None:
        alpha = self.row("alpha")
        alpha.provide(DRIVER, "alpha-driver")
        alpha.provide(NUMERICS, "alpha-numerics")

        self.assertEqual(alpha.provided, {"driver_svc", "numerics_svc"})
        entries = self.root.realm.entries()
        self.assertEqual(sorted(entries), ["driver_svc", "numerics_svc"])
        self.assertIsInstance(entries["driver_svc"], Impl)


class RealmIsolationTest(_RealmCase):
    def test_isolated_realms_each_resolve_their_own_provider(self) -> None:
        left = self.row("left", isolate=("driver_svc",))
        right = self.row("right", isolate=("driver_svc",))
        self.assertIsNot(left.realm, right.realm)
        self.assertIs(left.realm.parent, self.root.realm)
        self.assertEqual(left.realm.label, "left")
        self.assertEqual(left.realm.isolated, frozenset({"driver_svc"}))

        # The same single-cardinality seam, twice, with no collision.
        left.provide(DRIVER, "left-driver")
        right.provide(DRIVER, "right-driver")

        left_impl = left.realm.resolve("driver_svc")
        right_impl = right.realm.resolve("driver_svc")
        assert left_impl is not None and right_impl is not None
        self.assertEqual(left_impl.value, "left-driver")
        self.assertEqual(left_impl.entry_id, "left")
        self.assertEqual(right_impl.value, "right-driver")
        self.assertEqual(right_impl.entry_id, "right")
        # Neither publication escaped into the shared realm.
        self.assertIsNone(self.root.realm.resolve("driver_svc"))
        self.assertEqual(dict(self.root.realm.entries()), {})
        self.assertEqual(sorted(left.realm.entries()), ["driver_svc"])
        self.assertEqual(self.root.errors, [])

    def test_isolated_realm_still_collides_with_itself(self) -> None:
        left = self.row("left", isolate=("driver_svc",))
        other = self.root.mount(
            _declaration("left-twin", lambda ctx, config: None),
            entry_id="left-twin",
            parent=left,
        )
        self.root.settle()
        self.assertIs(other.realm, left.realm)
        left.provide(DRIVER, "left-driver")

        with self.assertRaises(DuplicateProvide) as caught:
            other.provide(DRIVER, "twin-driver")

        self.assertEqual(caught.exception.owner, "left")
        self.assertEqual(caught.exception.claimant, "left-twin")

    def test_child_realm_that_does_not_isolate_a_name_resolves_the_parents(self) -> None:
        host = self.row("host")
        # The child owns ``numerics_svc`` privately and delegates everything else upward.
        child = self.row("child", isolate=("numerics_svc",))
        host.provide(DRIVER, "host-driver")

        inherited = child.realm.resolve("driver_svc")
        assert inherited is not None
        self.assertEqual(inherited.value, "host-driver")
        self.assertIs(inherited.fiber, host)
        self.assertEqual(sorted(child.realm.entries()), ["driver_svc"])

        # A name it does isolate is private, and invisible upward.
        child.provide(NUMERICS, "child-numerics")
        private = child.realm.resolve("numerics_svc")
        assert private is not None
        self.assertEqual(private.value, "child-numerics")
        self.assertIsNone(self.root.realm.resolve("numerics_svc"))

        # Delegating upward means it cannot shadow the parent's service by accident.
        with self.assertRaises(DuplicateProvide) as caught:
            child.provide(DRIVER, "child-driver")
        self.assertEqual(caught.exception.owner, "host")
        self.assertEqual(caught.exception.claimant, "child")
        still = self.root.realm.resolve("driver_svc")
        assert still is not None
        self.assertEqual(still.value, "host-driver")

    def test_entries_agrees_with_resolve_about_an_isolated_name(self) -> None:
        """``entries()`` is the enumeration counterpart of ``resolve()``.

        A realm that isolates a name owns it privately, so an enumeration of what that realm can
        see must not list an implementation it deliberately cannot reach -- reporting one would
        attribute the parent's service to a realm that resolves it as absent.
        """
        host = self.row("host")
        child = self.row("child", isolate=("driver_svc",))
        host.provide(DRIVER, "host-driver")

        # The child isolates the name and nothing provided it there, so it is unreachable.
        self.assertIsNone(child.realm.resolve("driver_svc"))
        self.assertNotIn("driver_svc", child.realm.entries())
        # The parent still sees its own.
        self.assertEqual(self.root.realm.entries()["driver_svc"].value, "host-driver")

        # Once the child publishes its own, the two read paths agree again.
        child.provide(DRIVER, "child-driver")
        resolved = child.realm.resolve("driver_svc")
        assert resolved is not None
        self.assertEqual(resolved.value, "child-driver")
        self.assertEqual(child.realm.entries()["driver_svc"].value, "child-driver")

    def test_on_mutate_fires_for_a_provide_and_for_its_disposal(self) -> None:
        seen: list[str] = []
        stop = self.root.realm.on_mutate(seen.append)
        host = self.row("host")

        undo = host.provide(DRIVER, "host-driver")
        self.assertEqual(seen, ["driver_svc"])

        undo()
        self.assertEqual(seen, ["driver_svc", "driver_svc"])
        self.assertIsNone(self.root.realm.resolve("driver_svc"))

        # A publication inside an isolated child realm still reaches the root's listeners,
        # which is what makes a nested provide wake a waiting row.
        child = self.row("child", isolate=("numerics_svc",))
        child.provide(NUMERICS, "child-numerics")
        self.assertEqual(seen[-1], "numerics_svc")

        stop()
        host.provide(DRIVER, "host-driver-2")
        self.assertEqual(len(seen), 3)

    def test_a_rejected_provide_notifies_nobody(self) -> None:
        seen: list[str] = []
        self.addCleanup(self.root.realm.on_mutate(seen.append))
        alpha = self.row("alpha")
        beta = self.row("beta")
        alpha.provide(DRIVER, "alpha-driver")

        with self.assertRaises(DuplicateProvide):
            beta.provide(DRIVER, "beta-driver")

        self.assertEqual(seen, ["driver_svc"])


class RegistryCardinalityGuardTest(_RealmCase):
    def test_registry_seam_rejects_a_value_missing_both_methods(self) -> None:
        row = self.row("gate-host")

        with self.assertRaises(TypeError) as caught:
            row.provide(GATES, object())

        message = str(caught.exception)
        self.assertIn('entry "gate-host"', message)
        self.assertIn("registry seam gates_svc", message)
        self.assertIn("missing register, ids()", message)
        self.assertIn("embed a ProviderRegistry or declare the seam single", message)
        # Rejected outright: nothing registered, nothing to unwind.
        self.assertIsNone(self.root.realm.resolve("gates_svc"))
        self.assertEqual(row.provided, set())
        self.assertEqual(len(row.effects), 0)

    def test_registry_seam_names_only_the_missing_method(self) -> None:
        class HalfRegistry:
            def register(self, name: str, value: Any) -> None:
                return None

        row = self.row("gate-host")

        with self.assertRaises(TypeError) as caught:
            row.provide(GATES, HalfRegistry())

        message = str(caught.exception)
        self.assertIn("missing ids()", message)
        self.assertNotIn("register,", message)

    def test_registry_seam_rejects_non_callable_attributes(self) -> None:
        class Decoy:
            register = "not callable"
            ids = ()

        row = self.row("gate-host")

        with self.assertRaises(TypeError) as caught:
            row.provide(GATES, Decoy())

        self.assertIn("missing register, ids()", str(caught.exception))

    def test_registry_seam_accepts_a_provider_registry(self) -> None:
        row = self.row("gate-host")
        registry = ProviderRegistry("gates_svc")

        row.provide(GATES, registry)

        resolved = self.root.realm.resolve("gates_svc")
        assert resolved is not None
        self.assertIs(resolved.value, registry)
        self.assertEqual(resolved.key.cardinality, "registry")
        self.assertEqual(row.provided, {"gates_svc"})

    def test_single_cardinality_seam_accepts_any_value(self) -> None:
        row = self.row("plain")
        row.provide(DRIVER, object())
        self.assertIsNotNone(self.root.realm.resolve("driver_svc"))


class ProviderRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry: ProviderRegistry = ProviderRegistry("gates")

    def test_register_returns_a_disposer_that_removes_only_that_entry(self) -> None:
        undo_fast = self.registry.register("fast", "fast-gate", owner="row-a")
        self.registry.register("slow", "slow-gate", owner="row-b")
        self.assertEqual(len(self.registry), 2)
        self.assertEqual(self.registry.ids(), ("fast", "slow"))

        undo_fast()

        self.assertEqual(self.registry.ids(), ("slow",))
        self.assertEqual(len(self.registry), 1)
        self.assertNotIn("fast", self.registry)
        self.assertIsNone(self.registry.get("fast"))
        with self.assertRaises(KeyError):
            self.registry["fast"]
        # The survivor is untouched.
        self.assertEqual(self.registry["slow"], "slow-gate")

    def test_disposing_twice_is_a_no_op(self) -> None:
        changes: list[tuple[str, ...]] = []
        self.registry.on_change(lambda: changes.append(self.registry.ids()))
        undo = self.registry.register("fast", "fast-gate")
        undo()
        undo()

        self.assertEqual(changes, [("fast",), ()])

    def test_re_registering_a_disposed_name_is_allowed(self) -> None:
        undo = self.registry.register("fast", "first", owner="row-a")
        undo()
        self.registry.register("fast", "second", owner="row-b")

        self.assertEqual(self.registry["fast"], "second")
        # The stale disposer must not delete the replacement.
        undo()
        self.assertEqual(self.registry["fast"], "second")

    def test_duplicate_name_raises_duplicate_provide_naming_both_owners(self) -> None:
        self.registry.register("fast", "first", owner="row-a")

        with self.assertRaises(DuplicateProvide) as caught:
            self.registry.register("fast", "second", owner="row-b")

        collision = caught.exception
        self.assertEqual(collision.name, "gates.fast")
        self.assertEqual(collision.owner, "row-a")
        self.assertEqual(collision.claimant, "row-b")
        self.assertIn('service "gates.fast"', str(collision))
        self.assertIn('entry "row-a"', str(collision))
        self.assertIn('entry "row-b"', str(collision))
        # The incumbent survives the rejected registration.
        self.assertEqual(self.registry["fast"], "first")
        self.assertEqual(len(self.registry), 1)

    def test_ordered_sorts_by_order_then_name(self) -> None:
        # Registered in an order that neither insertion nor name alone would reproduce.
        self.registry.register("zulu", "z", order=100, owner="row-z")
        self.registry.register("alpha", "a", order=900, owner="row-a")
        self.registry.register("mike", "m", order=100, owner="row-m")
        self.registry.register("bravo", "b", owner="row-b")  # default order 500

        self.assertEqual(self.registry.ids(), ("mike", "zulu", "bravo", "alpha"))
        self.assertEqual(self.registry.values(), ("m", "z", "b", "a"))
        self.assertEqual(tuple(self.registry), self.registry.ids())
        self.assertEqual(len(self.registry), 4)
        self.assertEqual(
            [(entry.name, entry.order, entry.owner) for entry in self.registry.ordered()],
            [
                ("mike", 100, "row-m"),
                ("zulu", 100, "row-z"),
                ("bravo", 500, "row-b"),
                ("alpha", 900, "row-a"),
            ],
        )

    def test_reads_of_an_empty_registry(self) -> None:
        self.assertEqual(self.registry.ids(), ())
        self.assertEqual(self.registry.ordered(), ())
        self.assertEqual(self.registry.values(), ())
        self.assertEqual(len(self.registry), 0)
        self.assertNotIn("fast", self.registry)
        self.assertIsNone(self.registry.get("fast"))
        self.assertEqual(list(self.registry), [])

    def test_on_change_fires_after_register_and_after_dispose(self) -> None:
        snapshots: list[tuple[str, ...]] = []
        stop = self.registry.on_change(lambda: snapshots.append(self.registry.ids()))

        undo = self.registry.register("fast", "fast-gate", order=100)
        # The listener observes the mutation already applied, not the state before it.
        self.assertEqual(snapshots, [("fast",)])

        self.registry.register("slow", "slow-gate", order=200)
        self.assertEqual(snapshots, [("fast",), ("fast", "slow")])

        undo()
        self.assertEqual(snapshots[-1], ("slow",))

        # A rejected duplicate is not a change.
        with self.assertRaises(DuplicateProvide):
            self.registry.register("slow", "other")
        self.assertEqual(len(snapshots), 3)

        stop()
        self.registry.register("third", "third-gate")
        self.assertEqual(len(snapshots), 3)

    def test_two_listeners_both_fire_and_unsubscribe_independently(self) -> None:
        first: list[int] = []
        second: list[int] = []
        stop_first = self.registry.on_change(lambda: first.append(len(self.registry)))
        self.registry.on_change(lambda: second.append(len(self.registry)))

        self.registry.register("fast", "fast-gate")
        stop_first()
        self.registry.register("slow", "slow-gate")

        self.assertEqual(first, [1])
        self.assertEqual(second, [1, 2])


class ScopeMergeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scope = Scope("global")

    def entry(self, name: str, value: str, *, owner: str = "", order: int = 500) -> ScopedEntry:
        return ScopedEntry(name=name, value=value, owner=owner, order=order)

    def names(self, scope: Scope, kind: str = "tool") -> tuple[str, ...]:
        return tuple(entry.name for entry in scope.resolve(kind))

    def test_resolve_orders_by_order_then_name(self) -> None:
        self.scope.contribute("tool", self.entry("zulu", "z", order=100))
        self.scope.contribute("tool", self.entry("alpha", "a", order=900))
        self.scope.contribute("tool", self.entry("mike", "m", order=100))
        self.scope.contribute("tool", self.entry("bravo", "b"))
        self.scope.contribute("prompt", self.entry("header", "h"))

        self.assertEqual(self.names(self.scope), ("mike", "zulu", "bravo", "alpha"))
        self.assertEqual(
            tuple(entry.value for entry in self.scope.resolve("tool")),
            ("m", "z", "b", "a"),
        )
        # Kinds are independent namespaces.
        self.assertEqual(self.names(self.scope, "prompt"), ("header",))
        self.assertEqual(self.scope.kinds(), ("prompt", "tool"))
        self.assertEqual(self.scope.resolve("skill"), ())

    def test_disposing_a_contribution_removes_only_that_entry(self) -> None:
        undo = self.scope.contribute("tool", self.entry("grep", "g"))
        self.scope.contribute("tool", self.entry("bash", "b"))

        undo()

        self.assertEqual(self.names(self.scope), ("bash",))
        undo()
        self.assertEqual(self.names(self.scope), ("bash",))

    def test_duplicate_name_in_one_layer_raises_duplicate_provide(self) -> None:
        self.scope.contribute("tool", self.entry("grep", "first", owner="row-a"))

        with self.assertRaises(DuplicateProvide) as caught:
            self.scope.contribute("tool", self.entry("grep", "second", owner="row-b"))

        collision = caught.exception
        self.assertEqual(collision.name, "tool.grep")
        self.assertEqual(collision.owner, "row-a")
        self.assertEqual(collision.claimant, "row-b")
        self.assertIn('service "tool.grep"', str(collision))
        self.assertIn('entry "row-b"', str(collision))
        # The incumbent survives, and the same name is free in another kind.
        self.assertEqual(
            tuple(entry.value for entry in self.scope.resolve("tool")), ("first",)
        )
        self.scope.contribute("prompt", self.entry("grep", "prompt-grep"))

    def test_ownerless_duplicate_is_attributed_to_the_layer(self) -> None:
        self.scope.contribute("tool", self.entry("grep", "first"))

        with self.assertRaises(DuplicateProvide) as caught:
            self.scope.contribute("tool", self.entry("grep", "second", owner="row-b"))

        self.assertEqual(caught.exception.owner, "global")
        self.assertIn('entry "global"', str(caught.exception))

    def test_child_entry_shadows_the_parents_same_name(self) -> None:
        child = self.scope.child("session")
        self.scope.contribute("tool", self.entry("grep", "global-grep", order=100))
        self.scope.contribute("tool", self.entry("bash", "global-bash"))
        # Same name, one layer down: not a duplicate, a shadow.
        child.contribute("tool", self.entry("grep", "session-grep", order=900))

        merged = child.resolve("tool")
        self.assertEqual(tuple(entry.name for entry in merged), ("bash", "grep"))
        self.assertEqual(
            {entry.name: entry.value for entry in merged},
            {"bash": "global-bash", "grep": "session-grep"},
        )
        # Nearest wins outright, so the winner's own order decides placement.
        self.assertEqual(merged[-1].order, 900)
        # The parent layer is untouched by what its child did.
        self.assertEqual(
            {entry.name: entry.value for entry in self.scope.resolve("tool")},
            {"bash": "global-bash", "grep": "global-grep"},
        )
        self.assertEqual(
            tuple(entry.value for entry in child.local("tool")), ("session-grep",)
        )

    def test_nearest_wins_three_layers_deep(self) -> None:
        session = self.scope.child("session")
        task = session.child("task")
        self.scope.contribute("tool", self.entry("grep", "global-grep"))
        session.contribute("tool", self.entry("grep", "session-grep"))
        task.contribute("tool", self.entry("grep", "task-grep"))

        self.assertEqual(
            tuple(entry.value for entry in task.resolve("tool")), ("task-grep",)
        )
        self.assertEqual(
            tuple(entry.value for entry in session.resolve("tool")), ("session-grep",)
        )
        self.assertEqual(
            tuple(entry.value for entry in self.scope.resolve("tool")), ("global-grep",)
        )

    def test_chain_runs_root_first(self) -> None:
        session = self.scope.child("session")
        task = session.child("task")

        self.assertEqual(
            tuple(scope.name for scope in task.chain()), ("global", "session", "task")
        )
        self.assertIs(task.chain()[0], self.scope)
        self.assertIs(task.chain()[-1], task)
        self.assertEqual(self.scope.chain(), (self.scope,))
        self.assertIs(task.parent, session)
        self.assertIsNone(self.scope.parent)


class ScopeRestrictTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scope = Scope("global")
        self.child = self.scope.child("session")
        for name in ("bash", "edit", "grep"):
            self.scope.contribute(
                "tool", ScopedEntry(name=name, value=f"global-{name}", owner="host")
            )

    def names(self, scope: Scope) -> tuple[str, ...]:
        return tuple(entry.name for entry in scope.resolve("tool"))

    def test_deny_filters_inherited_entries_only(self) -> None:
        # The child denies a name it also contributes itself, plus one it only inherits.
        self.child.contribute(
            "tool", ScopedEntry(name="grep", value="session-grep", owner="row")
        )
        self.child.restrict("tool", deny=("grep", "edit"))

        merged = self.child.resolve("tool")
        self.assertEqual(tuple(entry.name for entry in merged), ("bash", "grep"))
        # Its own contribution is never filtered; only the inherited copy went away.
        self.assertEqual(
            {entry.name: entry.value for entry in merged},
            {"bash": "global-bash", "grep": "session-grep"},
        )
        # The owning layer is unaffected: restriction is a view, not a mutation.
        self.assertEqual(self.names(self.scope), ("bash", "edit", "grep"))

    def test_allow_filters_inherited_entries_only(self) -> None:
        self.child.contribute(
            "tool", ScopedEntry(name="write", value="session-write", owner="row")
        )
        self.child.restrict("tool", allow=("bash",))

        self.assertEqual(self.names(self.child), ("bash", "write"))
        self.assertEqual(self.names(self.scope), ("bash", "edit", "grep"))

    def test_allow_and_deny_together_intersect(self) -> None:
        self.child.restrict("tool", allow=("bash", "edit"), deny=("edit",))
        self.assertEqual(self.names(self.child), ("bash",))

    def test_restriction_is_per_kind(self) -> None:
        self.scope.contribute(
            "prompt", ScopedEntry(name="grep", value="global-prompt", owner="host")
        )
        self.child.restrict("tool", deny=("grep",))

        self.assertEqual(self.names(self.child), ("bash", "edit"))
        self.assertEqual(
            tuple(entry.name for entry in self.child.resolve("prompt")), ("grep",)
        )

    def test_a_deeper_layer_restricts_what_the_middle_layer_contributed(self) -> None:
        self.child.contribute(
            "tool", ScopedEntry(name="write", value="session-write", owner="row")
        )
        grandchild = self.child.child("task")
        grandchild.contribute(
            "tool", ScopedEntry(name="write", value="task-write", owner="row")
        )
        grandchild.restrict("tool", allow=("write",))

        # ``write`` survives because the grandchild contributed it, not because the
        # allowlist saved the inherited one.
        merged = grandchild.resolve("tool")
        self.assertEqual(tuple(entry.name for entry in merged), ("write",))
        self.assertEqual(merged[0].value, "task-write")
        self.assertEqual(self.names(self.child), ("bash", "edit", "grep", "write"))

    def test_restrict_disposer_puts_the_previous_filter_back(self) -> None:
        undo_deny = self.child.restrict("tool", deny=("bash",))
        self.assertEqual(self.names(self.child), ("edit", "grep"))

        undo_allow = self.child.restrict("tool", allow=("edit",))
        self.assertEqual(self.names(self.child), ("edit",))

        undo_allow()

        # Back to the deny, not back to unrestricted.
        self.assertEqual(self.names(self.child), ("edit", "grep"))

        undo_deny()

        self.assertEqual(self.names(self.child), ("bash", "edit", "grep"))

    def test_restricting_the_layer_that_owns_the_entries_filters_nothing(self) -> None:
        """A restriction belongs to the layer that inherits, not the layer that owns.

        ``bash`` is contributed *at* the global layer, so the global layer's own
        restriction cannot revoke it -- not for that layer and not for its children. The
        way to keep a tool out of a subtree is to restrict in the subtree.
        """
        self.scope.restrict("tool", deny=("bash",))

        self.assertEqual(self.names(self.scope), ("bash", "edit", "grep"))
        self.assertEqual(self.names(self.child), ("bash", "edit", "grep"))

    def test_a_restriction_keeps_a_filtered_entry_out_of_deeper_layers(self) -> None:
        grandchild = self.child.child("task")
        self.child.restrict("tool", deny=("bash",))

        # Denying an inherited entry removes it from the accumulation the deeper layers
        # build on, so the subtree stays consistent without restricting again.
        self.assertEqual(self.names(self.child), ("edit", "grep"))
        self.assertEqual(self.names(grandchild), ("edit", "grep"))
        self.assertEqual(self.names(self.scope), ("bash", "edit", "grep"))


class ScopeThroughContextTest(_RealmCase):
    """Contributions made through ``ctx`` are effects, so they unwind with the row."""

    def test_contributions_are_effect_scoped_to_the_contributing_row(self) -> None:
        def apply(ctx: Any, config: Mapping[str, Any]) -> None:
            ctx.contribute("tool", "grep", "grep-tool", order=100)
            ctx.contribute("tool", "bash", "bash-tool", order=200)

        fiber = self.root.mount(_declaration("tools", apply), entry_id="tools")
        self.root.settle()

        self.assertIs(fiber.state, FiberState.ACTIVE)
        resolved = self.root.scope.resolve("tool")
        self.assertEqual(
            [(entry.name, entry.value, entry.owner) for entry in resolved],
            [("grep", "grep-tool", "tools"), ("bash", "bash-tool", "tools")],
        )
        self.assertEqual(len(fiber.effects), 2)

        fiber.dispose()

        self.assertEqual(self.root.scope.resolve("tool"), ())
        self.assertEqual(self.root.live_effects, 0)

    def test_two_rows_contributing_one_name_is_an_attributable_load_failure(self) -> None:
        def apply(ctx: Any, config: Mapping[str, Any]) -> None:
            ctx.contribute("tool", "grep", f"{ctx.entry_id}-grep")

        self.root.mount(_declaration("first", apply), entry_id="first")
        self.root.mount(_declaration("second", apply), entry_id="second")
        self.root.settle()

        self.assertEqual(self.root.active, ("first",))
        failed = dict(self.root.failed)
        self.assertIn("second", failed)
        self.assertIn("DuplicateProvide", failed["second"])
        self.assertIn('service "tool.grep"', failed["second"])
        self.assertIn('entry "first"', failed["second"])
        # The winner keeps the name and the loser left nothing behind.
        self.assertEqual(
            [entry.value for entry in self.root.scope.resolve("tool")], ["first-grep"]
        )
        self.assertEqual(self.root.live_effects, 1)


if __name__ == "__main__":
    unittest.main()
