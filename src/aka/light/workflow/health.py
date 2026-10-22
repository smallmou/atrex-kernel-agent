"""Evidence-based local Campaign status for command schedulers and operators."""
import argparse
import json
from pathlib import Path
import time

from aka.light.runtime.paths import supervisor_campaign_root


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def inspect_health(workspace, *, command_phase=None, stale_seconds=1800, now=None):
    workspace = Path(workspace).resolve()
    now = time.time() if now is None else now
    root = workspace / ".atrex_long_horizon"
    private = supervisor_campaign_root(workspace)
    active, state, outcome = (read(root / name) for name in ("active_episode.json", "state.json", "outcome.json"))
    waits = [read(path) for category in ("infrastructure", "invalid_reviews", "baseline_repairs")
             for path in (private / category).glob("*.json")]
    waits = [row for row in waits if row.get("status") in {"blocked", "waiting_for_infrastructure"}]
    jobs = [read(path) for path in private.glob("**/jobs/*/state.json")]
    pending = [row for row in jobs if row.get("phase") not in {"terminal", "rejected"}]
    for row in pending:
        try:
            job = json.loads(row.get("process", {}).get("stdout", "{}"))
        except json.JSONDecodeError:
            job = {}
        row.update({key: job[key] for key in ("job_id", "status", "updated_at") if key in job})
    last_progress = active.get("last_progress_at")
    phase = active.get("phase", "unknown")
    if command_phase in {"Cancelled", "Failed"}:
        status = command_phase.lower()
    elif command_phase in {"Queued", "Starting"}:
        status = "queued" if command_phase == "Queued" else "starting"
    elif phase in {"paused_protocol", "paused_model_service", "validation_blocked"}:
        status = phase
    elif not active and outcome:
        status = ("budget_exhausted" if outcome.get("budget_exhausted") else
                  "finished" if outcome.get("search_finished") else "invocation_limited")
    elif any(row["status"] == "blocked" for row in waits):
        status = "validation_blocked"
    elif phase == "waiting_model_service":
        status = "waiting_model_service"
    elif waits:
        status = "waiting_validation"
    elif pending:
        status = "waiting_gpu"
    elif active:
        status = "stalled" if last_progress and now - last_progress > stale_seconds else "in_progress_unverified"
    else:
        status = "unverified"  # A scheduler exit code alone establishes no optimization result.
    return {"workspace": str(workspace), "status": status, "phase": phase,
            "episode": active.get("episode"), "completed_episodes": state.get("episodes", 0),
            "accepted": state.get("accepted", 0), "tokens": state.get("tokens", 0),
            "last_progress_at": last_progress,
            "protocol_repair_count": active.get("protocol_repair_count", 0),
            "model_service": active.get("model_service", {}), "validation_waits": waits,
            "pending_jobs": [{k: row[k] for k in ("job_id", "phase", "status", "updated_at") if k in row} for row in pending],
            "outcome": outcome}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--command-phase")
    args = parser.parse_args()
    print(json.dumps(inspect_health(args.workspace, command_phase=args.command_phase), indent=2))


if __name__ == "__main__":
    main()
