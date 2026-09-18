"""Independent, evidence-bound precision gate for production promotion.

The policy lives in the `precision-validation` plugin: which probes must run, and
whether the returned Atrex-Bench evidence and the independent review discharge them.
This module owns only what a plugin tool cannot -- the GPU allocation and its queue
wait, the isolated agent sessions, and the evidence digest that binds a pass to the
exact bytes that produced it.

Kept out of `campaign.py`'s import graph on purpose: it reaches `long_horizon.store`
and `tools.sandbox`, so it must stay a deferred import from its call sites.
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event
from urllib.parse import urlparse
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

from .infrastructure_retry import (
    InfrastructureUnavailable,
    check_review_service,
    check_transport,
    retry_infrastructure,
    retry_review,
)
from long_horizon.store import VERIFY_DIR

PLUGIN_ID = "precision-validation"
RESULT_PREFIX = "__ATREX_NUMERICAL_RESULT__="
HARNESS = Path(__file__).resolve().parents[1] / "reference" / "atrex_bench_test_kernel.py"
SUITE_FILENAME = "numerical_suite.json"


class PrecisionPluginMissing(RuntimeError):
    """The gate cannot be discharged without its policy plugin."""


def _plugin(campaign, workspace):
    """Resolve the policy plugin from a freshly fingerprinted registry.

    Deliberately not ``campaign.plugin_registry``: that is a cached_property whose
    fingerprints were computed when the campaign started, so it would keep validating a
    lock the plugin no longer matches. The gate's accept/reject policy is now a
    subprocess re-read from disk on every call, and the repository is reachable from an
    agent workspace, so the bytes about to run must be re-hashed and re-checked against
    the workspace's lock here. A tampered adapter becomes ``plugin_changed``, which the
    caller converts into a blocking violation.
    """
    from .plugins import PluginRegistry

    registry = PluginRegistry()
    registry.check_lock(workspace)
    plugin = next((p for p in registry.plugins if p.id == PLUGIN_ID), None)
    if plugin is None:
        raise PrecisionPluginMissing(
            f"the {PLUGIN_ID} plugin is not enabled; production promotion has no precision gate"
        )
    # The vendored evaluator is declared optional so a non-recursive clone still discovers
    # every plugin. The gate itself cannot run without it, so check here rather than
    # letting a missing runtime surface as an opaque failure inside the allocation.
    for name in ("atrex-bench-runtime", "atrex-bench-runner"):
        path, _mount = plugin.resources[name]
        if not path.exists():
            raise PrecisionPluginMissing(
                f"the vendored Atrex-Bench evaluator is missing at {path}; "
                "run `git submodule update --init 3rdparty/atrex-bench`"
            )
    return registry, plugin


def remote_agate(campaign):
    if campaign.sandbox_ssh:
        return False
    if campaign.sandbox_profile:
        return True
    endpoint = campaign.sandbox_url or os.environ.get("AGATE_URL", "")
    if not endpoint:
        # Respect agate's configured loopback endpoint without exposing credentials.
        config = Path.home() / ".atrex" / "config.json"
        if config.is_file():
            endpoint = json.loads(config.read_text()).get("url", "")
    return urlparse(endpoint).hostname not in {"localhost", "127.0.0.1", "::1"}


def gate_mode(campaign):
    selected = getattr(campaign, "numerical_gate", "auto")
    if selected == "auto":
        return "thorough" if remote_agate(campaign) else "light"
    if selected not in {"light", "thorough"}:
        raise ValueError("numerical gate must be auto, light or thorough")
    return selected


def plan_and_load(registry, workspace, suite_path, shapes, framework):
    """Load a suite and prove it schedulable, so an unusable one fails before any GPU work."""
    suite = json.loads(suite_path.read_text())
    plan(registry, workspace, suite, shapes, "light", "", framework)
    return suite


def check_authored_suite(suite):
    """Bounds that apply to a supervisor-authored suite but not to an operator's own."""
    if not 3 <= len(suite["cases"]) <= 6 or suite.get("coverage", "compact") != "compact":
        raise ValueError("automatically authored suites require 3..6 compact risk cases")
    if suite.get("evaluator_command") or suite.get("evaluator_files"):
        raise ValueError(
            "custom evaluator adapters must be operator-supplied, not invented by the suite author"
        )
    return suite


def resolve_suite(campaign, workspace, private, registry, plugin, shapes):
    """Return an operator-owned suite, or author and cache one from the trusted contract."""
    supplied = private / SUITE_FILENAME
    if supplied.is_file():
        # `private` falls back to the agent's own workspace when the campaign has no
        # private reference directory, so a suite found there is not evidence of operator
        # intent. A custom evaluator adapter is the one field that would let such a suite
        # replace the comparator outright, so it is only honoured from a real private dir.
        if campaign.private_reference_dir is None:
            claimed = json.loads(supplied.read_text())
            if claimed.get("evaluator_command") or claimed.get("evaluator_files"):
                raise ValueError(
                    "a numerical suite carrying a custom evaluator adapter must come from an "
                    "operator-owned private reference directory, not from the candidate workspace"
                )
        return supplied
    from .session_io import run_session

    prompt = plugin.root / "prompts" / "suite.md"
    sources = {}
    for name in ("input.py", "reference.py", "shapes.json", "agent_problem.json"):
        path = private / name if (private / name).is_file() else workspace / name
        if path.is_file():
            sources[name] = path
    if not {"input.py", "reference.py", "shapes.json"} <= set(sources):
        raise ValueError("precision suite generation needs trusted input.py, reference.py and shapes.json")
    sources["instructions.md"] = prompt
    for name in ("attention", "gemm", "norm"):
        sources[f"examples/{name}.json"] = plugin.root / "examples" / f"{name}.json"
    digest = hashlib.sha256()
    for name, path in sorted(sources.items()):
        digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    cached = private / ".atrex_numerical" / digest.hexdigest() / SUITE_FILENAME
    if cached.is_file():
        # A cache hit re-enters through the same bounds as a fresh authoring run. The cache
        # is keyed by a digest the candidate can compute from the contract it holds, and it
        # lives in the workspace whenever there is no private reference directory, so it is
        # not more trusted than the suite an authoring session would have produced.
        check_authored_suite(
            plan_and_load(registry, workspace, cached, shapes, campaign.framework))
        return cached

    def author_once():
        with tempfile.TemporaryDirectory(prefix="atrex-numerical-contract-") as temporary:
            root = Path(temporary)
            for name, path in sources.items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            result = run_session(root, prompt.read_text(),
                                 timeout=getattr(campaign, "numerical_review_timeout", 600),
                                 agent_cli=campaign.agent_cli, reasoning_effort="high", agent_plugins=False)
            campaign._account(result, "operator precision suite construction")
            check_review_service(result)
            for name, path in sources.items():
                if (root / name).read_bytes() != path.read_bytes():
                    raise ValueError("precision suite author modified its contract evidence")
            suite = check_authored_suite(
                plan_and_load(registry, workspace, root / SUITE_FILENAME, shapes, campaign.framework))
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(suite, indent=2) + "\n")

    retry_review(
        workspace,
        f"numerical-suite:{digest.hexdigest()}:{campaign.agent_cli}:{getattr(campaign, 'numerical_review_timeout', 600)}",
        author_once,
    )
    return cached


def plan(registry, workspace, suite, shapes, mode, rotation, framework):
    """Validate the suite and derive the probe schedule through the policy plugin.

    ``framework`` is normalized here, with the same function that decides what the
    workspace policy directive promises the agent, so the guard profile and that promise
    cannot disagree about which campaigns count as CUDA.
    """
    from .optimization_policy import _framework_key

    return registry.call(
        f"{PLUGIN_ID}.plan",
        {"schema_version": 1, "suite": suite, "shapes": shapes, "mode": mode,
         "rotation": rotation, "framework_key": _framework_key(framework),
         "trust_mode": "untrusted"},
        workspace,
    )


def evidence_files(workspace, private, suite_path, driver_path, review_prompt):
    files = {"candidate/kernel.py": workspace / "kernel.py", "evaluator.py": workspace / "test_kernel.py",
             SUITE_FILENAME: suite_path, "driver.py": driver_path, "transport.py": HARNESS,
             "numerical_review.md": review_prompt}
    for name in ("input.py", "reference.py", "shapes.json", "metadata.json"):
        path = private / name if (private / name).is_file() else workspace / name
        if name != "metadata.json" or path.is_file():
            files["trusted/" + name] = path
    for name in ("agent_problem.json", "solution.json", "README.md"):
        if (workspace / name).is_file():
            files[("candidate/" if name == "solution.json" else "trusted/") + name] = workspace / name
    suite = json.loads(suite_path.read_text())
    for index, name in enumerate(suite.get("evaluator_files", [])):
        path = (private / name).resolve()
        files[f"trusted/evaluator/{index}-{path.name}"] = path
    return files


def evidence_digest(files):
    digest = hashlib.sha256(b"precision-gate-v3\0")
    for name, path in sorted(files.items()):
        digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


INVALID_RESULT = "independent precision validation returned an invalid result"


def verdict_problem(verdict, subject):
    """Read a plugin verdict in the blocking direction, or return "" to continue.

    Both fields are authoritative: the output schema permits ``accepted: false`` with an
    empty list, and ``accepted: true`` alongside violations. Either shape must block, and
    must do so with a reason the operator can act on.
    """
    if verdict.get("accepted") is True and not verdict.get("violations"):
        return ""
    return "; ".join(verdict.get("violations") or ()) or f"{subject} was rejected without a stated reason"


def blocking_violations(campaign, workspace):
    """The call-site entry point: :func:`precision_violations` with its contract enforced.

    Every caller treats a falsy result as "promote", so a gate that returned ``None``, a
    dict, or a list containing anything but non-empty reasons would fail *open* at each of
    them. Enforcing the shape here rather than at each call site means a future replacement
    gate inherits the guarantee instead of having to re-derive it.
    """
    errors = precision_violations(campaign, workspace)
    if not isinstance(errors, list) or not all(
        isinstance(error, str) and error.strip() for error in errors
    ):
        return [INVALID_RESULT]
    return errors


def precision_violations(campaign, workspace):
    """Return blocking precision violations for the candidate at ``workspace``.

    An empty list is the only accepting result. Every failure path returns a violation
    rather than raising, so a caller can never read a blocked gate as a pass. Prefer
    :func:`blocking_violations` from a call site.
    """
    # Imported here to preserve the campaign/session module initialization order.
    from .session_io import _sandbox_command, run_session
    private = Path(campaign.private_reference_dir or workspace)
    record = {"schema_version": 1, "accepted": False}
    directory = workspace / VERIFY_DIR / ("numerical-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    try:
        registry, plugin = _plugin(campaign, workspace)
        record["plugin_version"] = plugin.manifest["version"]
        record["plugin_fingerprint"] = plugin.fingerprint
        driver_source = plugin.root / "driver.py"
        review_prompt = plugin.root / "prompts" / "review.md"
        shapes_path = private / "shapes.json" if (private / "shapes.json").is_file() else workspace / "shapes.json"
        shapes = json.loads(shapes_path.read_text())
        suite_path = resolve_suite(campaign, workspace, private, registry, plugin, shapes)
        suite = json.loads(suite_path.read_text())
        files = evidence_files(workspace, private, suite_path, driver_source, review_prompt)
        mode = gate_mode(campaign)
        source_digest = evidence_digest(files)
        digest = hashlib.sha256((source_digest + ":" + mode).encode()).hexdigest()
        record["evidence_digest"] = digest
        # Resume intentionally recertifies HEAD in each supervisor process. Saved
        # results are audit evidence, not portable GPU/runtime certificates.
        # Only a success in this process can skip the probes and reviewer.
        cache = getattr(campaign, "_numerical_review_cache", set())
        if digest in cache:
            record.update(accepted=True, cached=True)
            return []
        planned = plan(registry, workspace, suite, shapes, mode, digest, campaign.framework)
        schedule = planned["schedule"]
        coverage = dict(planned["coverage"], trust_mode=planned["trust_mode"])
        record["coverage"] = coverage
        driver = directory / "test_kernel.py"
        shutil.copy2(driver_source, driver)
        if "atrex-bench/run_eval" in (workspace / "test_kernel.py").read_text():
            snapshot = directory / "snapshots" / "evaluator.py"
            snapshot.parent.mkdir()
            shutil.copy2(HARNESS, snapshot)
        # Keep each allocation within the gateway's 600-second command limit.
        # Correctness probes never enter the ABBA timing aggregate.
        evaluation = {"schema_version": 1, "runs": [], "all_pass": True}
        record["evaluation"] = evaluation
        specs = []
        for index, row in enumerate(schedule):
            request = directory / f"request-{index:04d}.json"
            request.write_text(json.dumps({"suite": suite, "case_ids": [row["case_id"]],
                                           "rotation": digest, "mode": mode, "per_case_timeout": 540,
                                           "trust_mode": planned["trust_mode"]}))
            specs.append((index, row["case_id"], request))
        is_remote = remote_agate(campaign)
        # Concurrent submissions can wait for admission without consuming their
        # worker execution budget. Match the sandbox's existing queue allowance.
        from tools.sandbox import DEFAULT_QUEUE_WAIT_GRACE
        queue_grace = int(os.environ.get("ATREX_SANDBOX_QUEUE_WAIT_GRACE", str(DEFAULT_QUEUE_WAIT_GRACE)))
        wall_timeout = 600 + 240 + queue_grace if is_remote else None
        cancel = Event()

        def evaluate_case(spec):
            index, case_id, request = spec
            if cancel.is_set():
                raise subprocess.SubprocessError("precision batch cancelled")
            print(f"[precision-gate] mode={mode} trust={planned['trust_mode']} "
                  f"case={case_id} ({index + 1}/{len(specs)})", flush=True)
            try:
                process = _sandbox_command(
                    workspace, campaign.sandbox_hardware, campaign.sandbox_profile, campaign.sandbox_url,
                    600, ["python3", str(driver.relative_to(workspace)), str(request.relative_to(workspace))],
                    ssh=campaign.sandbox_ssh, ssh_init=campaign.sandbox_ssh_init,
                    health_command=campaign.sandbox_health_command, gateway_kind="dev",
                    private_reference_dir=campaign.private_reference_dir, cancel_event=cancel,
                    wall_timeout=wall_timeout)
            except subprocess.TimeoutExpired as exc:
                raise InfrastructureUnavailable("GPU transport wait deadline exceeded") from exc
            batch = None
            for line in process.stdout.splitlines():
                if line.startswith(RESULT_PREFIX):
                    batch = json.loads(line[len(RESULT_PREFIX):])
            if process.returncode or not isinstance(batch, dict):
                if not isinstance(batch, dict):
                    check_transport(process)
                raise ValueError(f"precision evaluator produced no result (exit={process.returncode}): {process.stderr[-1000:]}")
            return index, batch

        def run_case(spec):
            return retry_infrastructure(
                workspace, f"numerical:{digest}:{spec[1]}",
                lambda: evaluate_case(spec), cancel=cancel)

        # Each agate case owns a separate allocation. Submit all cases immediately;
        # gateway quotas/queueing, rather than a local worker cap, govern admission.
        workers = len(specs) if is_remote else 1
        coverage["concurrent_jobs"] = workers
        batches = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_case, spec) for spec in specs]
            try:
                for future in as_completed(futures):
                    index, batch = future.result()
                    batches[index] = batch
                    evaluation["runs"] = [row for i in sorted(batches) for row in batches[i].get("runs", [])]
                    if batch.get("all_pass") is not True or batch.get("error"):
                        evaluation["all_pass"] = False
                        if batch.get("error"):
                            evaluation["error"] = batch["error"]
                        raise ValueError(f"numerical distribution {specs[index][1]} failed; see numerical_result.json")
            except BaseException:
                cancel.set()
                for future in futures:
                    future.cancel()
                raise
        verdict = registry.call(
            f"{PLUGIN_ID}.check-evaluation",
            {"schema_version": 1, "evaluation": evaluation, "schedule": schedule,
             "world_size": suite["world_size"]},
            workspace,
        )
        problem = verdict_problem(verdict, "precision evidence")
        if problem:
            raise ValueError(problem)

        def review_once():
            with tempfile.TemporaryDirectory(prefix="atrex-numerical-review-") as temporary:
                review_root = Path(temporary)
                review_files = {name: path for name, path in files.items()
                                if name not in {"trusted/shapes.json", "trusted/metadata.json"}}
                visible_digest = evidence_digest(review_files)
                for name, path in review_files.items():
                    target = review_root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                (review_root / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
                (review_root / "review_request.json").write_text(json.dumps({"evidence_digest": digest, "coverage": coverage}))
                result = run_session(review_root, review_prompt.read_text(),
                                     timeout=getattr(campaign, "numerical_review_timeout", 600),
                                     agent_cli=campaign.agent_cli, reasoning_effort="high", agent_plugins=False)
                campaign._account(result, "independent precision review")
                check_review_service(result)
                if evidence_digest({name: review_root / name for name in review_files}) != visible_digest:
                    raise ValueError("precision reviewer modified supplied evidence")
                if json.loads((review_root / "evaluation.json").read_text()) != evaluation:
                    raise ValueError("precision reviewer modified evaluation evidence")
                review = json.loads((review_root / "numerical_review.json").read_text())
                record["review"] = review
                judged = registry.call(
                    f"{PLUGIN_ID}.check-review",
                    {"schema_version": 1, "review": review, "evidence_digest": digest,
                     "supplied_files": sorted(set(review_files) | {"evaluation.json", SUITE_FILENAME})},
                    workspace,
                )
                problem = verdict_problem(judged, "precision review")
                # A rejecting review's violations *are* the gate's findings, so they are
                # returned rather than raised; only an unusable verdict raises.
                return list(judged["violations"]) if judged["violations"] else (
                    [problem] if problem else []
                )

        errors = retry_review(
            workspace,
            f"numerical-review:{digest}:{campaign.agent_cli}:{getattr(campaign, 'numerical_review_timeout', 600)}",
            review_once,
        )
        if evidence_digest(files) != source_digest:
            raise ValueError("candidate or precision contract changed during validation")
        record["errors"] = errors
        record["accepted"] = not errors
        if not errors:
            cache.add(digest)
            campaign._numerical_review_cache = cache
        return errors
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError,
            PrecisionPluginMissing) as exc:
        record["errors"] = [f"production precision validation blocked: {exc}"]
        return record["errors"]
    finally:
        (directory / "numerical_result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
