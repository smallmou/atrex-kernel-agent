from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from orchestrator import optimize


def _manifest(names: tuple[str, ...]) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "dispatch_visibility_policy": optimize.DISPATCH_VISIBILITY_POLICY,
            "workload_count": 2,
            "buckets": [
                {"name": name, "workload_indices": [index], "rationale": "test"}
                for index, name in enumerate(names)
            ],
        },
        indent=2,
    )


class SharedBucketManifestTest(unittest.TestCase):
    def test_adoption_is_byte_identical(self) -> None:
        with TemporaryDirectory(prefix="shared-bucket-") as temp_dir:
            root = Path(temp_dir)
            shared = root / "shared" / "op.json"
            shared.parent.mkdir(parents=True)
            payload = _manifest(("low", "high"))
            shared.write_text(payload, encoding="utf-8")

            local = root / "campaign" / "workload_buckets.json"
            self.assertTrue(optimize.adopt_shared_bucket_manifest(shared, local))
            self.assertEqual(local.read_text(encoding="utf-8"), payload)

    def test_adoption_reports_absence_without_creating_anything(self) -> None:
        with TemporaryDirectory(prefix="shared-bucket-") as temp_dir:
            root = Path(temp_dir)
            shared = root / "shared" / "missing.json"
            local = root / "campaign" / "workload_buckets.json"
            self.assertFalse(optimize.adopt_shared_bucket_manifest(shared, local))
            self.assertFalse(local.exists())

    def test_publication_is_atomic(self) -> None:
        with TemporaryDirectory(prefix="shared-bucket-") as temp_dir:
            shared = Path(temp_dir) / "shared" / "op.json"
            payload = _manifest(("low", "high"))
            optimize.publish_shared_bucket_manifest(shared, payload)
            self.assertEqual(shared.read_text(encoding="utf-8"), payload)
            # os.replace leaves no staging file behind, so no child can read a partial one.
            self.assertEqual(
                sorted(entry.name for entry in shared.parent.iterdir()), ["op.json"]
            )

    def test_lock_serializes_two_holders(self) -> None:
        with TemporaryDirectory(prefix="shared-bucket-") as temp_dir:
            shared = Path(temp_dir) / "shared" / "op.json"
            order: list[str] = []
            first_inside = threading.Event()
            release_first = threading.Event()

            def hold_first() -> None:
                with optimize.shared_partition_lock(shared):
                    order.append("first-enter")
                    first_inside.set()
                    release_first.wait(timeout=5)
                    order.append("first-exit")

            def hold_second() -> None:
                first_inside.wait(timeout=5)
                with optimize.shared_partition_lock(shared):
                    order.append("second-enter")

            threads = [threading.Thread(target=hold_first), threading.Thread(target=hold_second)]
            for thread in threads:
                thread.start()
            first_inside.wait(timeout=5)
            release_first.set()
            for thread in threads:
                thread.join(timeout=10)

            # flock is per-file-descriptor, so both holders are serialized: the second may
            # only enter after the first has left.
            self.assertEqual(order[0], "first-enter")
            self.assertLess(order.index("first-exit"), order.index("second-enter"))

    def test_lock_is_a_no_op_without_a_shared_manifest(self) -> None:
        with optimize.shared_partition_lock(None) as shared:
            self.assertIsNone(shared)

    def test_manifest_path_is_deterministic_and_filesystem_safe(self) -> None:
        base = Path("/tmp/work")
        first = optimize.shared_bucket_manifest_path(base, "gdn_gating", "pro5000", "production")
        second = optimize.shared_bucket_manifest_path(base, "gdn_gating", "pro5000", "production")
        self.assertEqual(first, second)
        self.assertEqual(first.parent, base / "shared_workload_buckets")
        self.assertEqual(first.name, "gdn_gating_pro5000_production.json")
        # Different operators, platforms or modes must not collide.
        self.assertNotEqual(
            first, optimize.shared_bucket_manifest_path(base, "gdn_gating", "h20", "production")
        )
        self.assertNotEqual(
            first, optimize.shared_bucket_manifest_path(base, "lm_head", "pro5000", "production")
        )
        self.assertNotEqual(
            first, optimize.shared_bucket_manifest_path(base, "gdn_gating", "pro5000", "leaderboard")
        )


class SharedManifestDispatchWiringTest(unittest.TestCase):
    def test_shared_flag_is_stripped_before_being_reinjected(self) -> None:
        # The dispatcher re-adds its own value, so a child must never inherit a stale one.
        argv = [
            "--op-dir", "/tmp/op",
            "--platform", "pro5000",
            "--shared-bucket-manifest", "/tmp/stale.json",
        ]
        stripped = optimize._without_cli_options(
            argv,
            (
                "--framework",
                "--workspace",
                "--arch",
                "--workspace-suffix",
                "--shared-bucket-manifest",
            ),
        )
        self.assertNotIn("--shared-bucket-manifest", stripped)
        self.assertNotIn("/tmp/stale.json", stripped)
        self.assertEqual(stripped, ["--op-dir", "/tmp/op", "--platform", "pro5000"])


if __name__ == "__main__":
    unittest.main()
