"""Explicit stopped-workspace migration; default invocation only prints a plan."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from aka.bootstrap.continuation import restore
from aka.contracts.workflow_launch import decode_launch
from aka.legacy.task.problem.snapshot import inspect
from aka.legacy.workflow.checkpoint import CONTEXT
from aka.legacy.workflow.public_state import campaign_lease
from aka.legacy.task.persistence import durable_write_json
from aka.light.runtime.plugins.registry import PluginRegistry
from aka.light.runtime.paths import supervisor_campaign_root


def migration_plan(workspace, context, plugin_dir):
    workspace = Path(workspace).resolve()
    path = workspace / CONTEXT
    old_bytes = path.read_bytes()
    old = json.loads(old_bytes)
    new = json.loads(Path(context).read_text())
    if old["argv"] != new["argv"] or old["problem"] != new["problem"]:
        raise ValueError("Migration must preserve every launch argument and the frozen problem")
    request = decode_launch(tuple(new["argv"])).request
    if request.workspace.resolve() != workspace:
        raise ValueError("Context belongs to another workspace")
    inspect(Path(new["problem"]["directory"]), identity=new["problem"]["identity"])
    restore(new["selection"])  # Validate the target installed runtime before changing any state.
    lock = supervisor_campaign_root(workspace) / "plugins/.atrex_plugins/lock.json"
    changes = [(path, old_bytes, (json.dumps(new, sort_keys=True) + "\n").encode())]
    if lock.exists():
        before = lock.read_bytes()
        previous = json.loads(before)
        target = PluginRegistry(Path(plugin_dir)).snapshot()
        # Relocation of a nonempty plugin tree needs a separate content migration.
        if previous["plugins"] or target["plugins"]:
            if previous != target:
                raise ValueError("Nonempty plugin locks must remain identical")
        elif {k: v for k, v in previous.items() if k != "plugin_dir"} != {
            k: v for k, v in target.items() if k != "plugin_dir"
        }:
            raise ValueError("Plugin lock semantics changed")
        changes.append((lock, before, (json.dumps(target, sort_keys=True) + "\n").encode()))
    rows = [{"path": str(p), "before": hashlib.sha256(a).hexdigest(),
             "after": hashlib.sha256(b).hexdigest()} for p, a, b in changes]
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return {"workspace": str(workspace), "problem_identity": new["problem"]["identity"],
            "changes": rows, "plan_digest": digest}, changes


def migrate(workspace, context, plugin_dir, expected_digest=None):
    workspace = Path(workspace).resolve()
    with campaign_lease(workspace):
        plan, changes = migration_plan(workspace, context, plugin_dir)
        if expected_digest is None:
            return plan
        if expected_digest != plan["plan_digest"]:
            raise ValueError("Migration inputs changed after review; generate a new plan")
        backup = supervisor_campaign_root(workspace) / "migrations" / str(time.time_ns())
        backup.mkdir(parents=True)
        for index, (path, before, after) in enumerate(changes):
            (backup / f"{index}.before").write_bytes(before)
            (backup / f"{index}.after").write_bytes(after)
        durable_write_json(backup / "manifest.json", plan)
        try:
            for path, _, after in changes:
                durable_write_json(path, json.loads(after))
        except BaseException:
            for path, before, _ in changes:
                durable_write_json(path, json.loads(before))
            raise
        return dict(plan, applied=True, backup=str(backup))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--context", type=Path, help="Complete context captured with the target runtime")
    parser.add_argument("--plugin-dir", type=Path)
    parser.add_argument("--apply-plan", help="Exact plan_digest from the read-only preview")
    parser.add_argument("--rollback", type=Path, help="Backup directory returned by a completed migration")
    args = parser.parse_args()
    if args.rollback:
        print(json.dumps(rollback(args.workspace, args.rollback), indent=2))
        return
    if args.context is None or args.plugin_dir is None:
        parser.error("--context and --plugin-dir are required for a migration plan")
    print(json.dumps(migrate(args.workspace, args.context, args.plugin_dir, args.apply_plan), indent=2))


def rollback(workspace, backup):
    workspace, backup = Path(workspace).resolve(), Path(backup).resolve()
    with campaign_lease(workspace):
        if not backup.is_relative_to(supervisor_campaign_root(workspace) / "migrations"):
            raise ValueError("Backup does not belong to this campaign")
        plan = json.loads((backup / "manifest.json").read_text())
        if plan["workspace"] != str(workspace):
            raise ValueError("Migration belongs to another workspace")
        allowed = {workspace / CONTEXT, supervisor_campaign_root(workspace) / "plugins/.atrex_plugins/lock.json"}
        changes = []
        for index, row in enumerate(plan["changes"]):
            path = Path(row["path"])
            before, after = (backup / f"{index}.before").read_bytes(), (backup / f"{index}.after").read_bytes()
            if path not in allowed or hashlib.sha256(before).hexdigest() != row["before"]:
                raise ValueError("Invalid migration backup")
            if json.loads(path.read_bytes()) != json.loads(after):
                raise ValueError("Context changed after migration; refusing to overwrite it")
            changes.append((path, before))
        for path, before in changes:
            durable_write_json(path, json.loads(before))
        return {"rolled_back": True, "workspace": str(workspace)}


if __name__ == "__main__":
    main()
