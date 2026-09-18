"""Provide ``ctx.driver`` from today's campaign state machine.

Stage one of the plugin refactor: the seam and the composition exist, the host boots a plugin
tree, and this row still constructs ``orchestrator.campaign.Campaign`` and drives it in exactly
the order ``orchestrator/optimize.py`` used to. Nothing about a campaign changes; what changes
is that the entry point now composes a tree instead of wiring one object.

Its ``Config`` is the whole legacy campaign configuration, so the CLI's flag values are schema
validated before the campaign is built, and ``--dump-config`` can show them.
"""

from __future__ import annotations

from typing import Any, Mapping

from orchestrator.constants import (
    DEFAULT_CONVERT_AFTER,
    DEFAULT_FAST_EPISODES,
    DEFAULT_FAST_TRIALS,
    DEFAULT_HANDOFF_RESUMES,
    DEFAULT_SANDBOX_TIMEOUT,
    DEFAULT_VERIFY_REPEATS,
    DEFAULT_VERIFY_RUN_TIMEOUT,
    DEPENDENCY_REVIEW_TIMEOUT_S,
    FRAMEWORK_BASELINE_MODES,
    FRAMEWORK_BASELINE_TIMEOUT_S,
    MAX_SANDBOX_TIMEOUT,
)

from aka.core.context import Context
from aka.seams.driver import DRIVER

from .invariant import PACKAGE_NAME, install
from .service import UNASSIGNED_GPU, LegacyCampaignDriver, build_campaign

name = "legacy-campaign"
inject: tuple[str, ...] = ()
optional_inject: tuple[str, ...] = ()
provide: tuple[str, ...] = ("driver",)
interpolate: tuple[str, ...] = ("work_dir",)


def _text(*, min_length: int = 0, description: str = "") -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if min_length:
        schema["minLength"] = min_length
    if description:
        schema["description"] = description
    return schema


def _count(*, minimum: int = 0, maximum: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "integer", "minimum": minimum}
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


Config: dict[str, Any] = {
    "type": "object",
    "description": "One operator campaign, as the supported CLI configures it.",
    "additionalProperties": False,
    "required": ["name", "kernel_demo", "platform", "framework"],
    "properties": {
        "name": _text(min_length=1, description="Operator name derived from --op-dir."),
        "kernel_demo": _text(min_length=1, description="Reference implementation path."),
        "platform": _text(min_length=1, description="Target hardware token."),
        "framework": _text(min_length=1, description="Target DSL."),
        "notes": _text(description="Extra constraints or known bottlenecks."),
        "arch": _text(description="Runtime GPU architecture, empty when undetected."),
        "work_dir": _text(description="Directory containing the campaign workspace."),
        "workspace_suffix": _text(description="Internal auto-dispatch workspace suffix."),
        "atrex_bench_root": _text(description="Native evaluator checkout owning run_eval.py."),
        "max_iters": _count(minimum=1),
        "token_budget": _count(),
        "target_util": {"type": "number", "minimum": 0.0, "maximum": 100.0},
        "setup_timeout": _count(minimum=1),
        "max_stall": _count(),
        "fast_episodes": _count(),
        "fast_trials": _count(minimum=1),
        "convert_after": _count(),
        "handoff_resumes": _count(),
        "sandbox_hardware": _text(),
        "sandbox_profile": {"type": "string", "enum": ["", "pre", "prod"]},
        "sandbox_url": _text(),
        "sandbox_timeout": _count(minimum=1, maximum=MAX_SANDBOX_TIMEOUT),
        "sandbox_ssh": _text(),
        "sandbox_ssh_init": _text(),
        "sandbox_ssh_gpu": _count(minimum=UNASSIGNED_GPU, maximum=31),
        "sandbox_health_command": _text(),
        "agent_cli": _text(min_length=1),
        "long_reviewer_session": {
            "type": "string",
            "enum": ["", "codex", "qoder", "claude"],
        },
        "optimization_mode": {
            "type": "string",
            "enum": ["leaderboard", "production"],
        },
        "framework_baseline": {
            "type": "string",
            "enum": list(FRAMEWORK_BASELINE_MODES),
        },
        "framework_baseline_timeout": _count(minimum=1),
        "numerical_gate": {
            "type": "string",
            "enum": ["auto", "light", "thorough"],
        },
        "repair_numerical_head": {"type": "boolean"},
        "numerical_review_timeout": _count(minimum=1),
        "production_review_timeout": _count(minimum=1),
        "verify_repeats": _count(minimum=1),
        "verify_run_timeout": _count(minimum=1),
        "min_improvement_pct": {"type": "number", "minimum": 0.0, "maximum": 99.999},
        "v1_ask_codex": {"type": "boolean"},
        "v1_ask_qoder": {"type": "boolean"},
        "fast_episode_ask_codex": {"type": "boolean"},
        "fast_episode_ask_qoder": {"type": "boolean"},
        "full_episode_ask_codex": {"type": "boolean"},
        "full_episode_ask_qoder": {"type": "boolean"},
    },
}

#: Mirrors ``Campaign``'s own field defaults, sourced from ``orchestrator.constants`` so the
#: two cannot drift while both exist.
Defaults: dict[str, Any] = {
    "notes": "none",
    "arch": "",
    "work_dir": "",
    "workspace_suffix": "",
    "atrex_bench_root": "",
    "max_iters": 20,
    "token_budget": 0,
    "target_util": 90.0,
    "setup_timeout": 7200,
    "max_stall": 0,
    "fast_episodes": DEFAULT_FAST_EPISODES,
    "fast_trials": DEFAULT_FAST_TRIALS,
    "convert_after": DEFAULT_CONVERT_AFTER,
    "handoff_resumes": DEFAULT_HANDOFF_RESUMES,
    "sandbox_hardware": "",
    "sandbox_profile": "",
    "sandbox_url": "",
    "sandbox_timeout": DEFAULT_SANDBOX_TIMEOUT,
    "sandbox_ssh": "",
    "sandbox_ssh_init": "",
    "sandbox_ssh_gpu": UNASSIGNED_GPU,
    "sandbox_health_command": "",
    "agent_cli": "claude",
    "long_reviewer_session": "",
    "optimization_mode": "leaderboard",
    "framework_baseline": "auto",
    "framework_baseline_timeout": FRAMEWORK_BASELINE_TIMEOUT_S,
    "numerical_gate": "auto",
    "repair_numerical_head": False,
    "numerical_review_timeout": 600,
    "production_review_timeout": DEPENDENCY_REVIEW_TIMEOUT_S,
    "verify_repeats": DEFAULT_VERIFY_REPEATS,
    "verify_run_timeout": DEFAULT_VERIFY_RUN_TIMEOUT,
    "min_improvement_pct": 0.0,
    "v1_ask_codex": False,
    "v1_ask_qoder": False,
    "fast_episode_ask_codex": False,
    "fast_episode_ask_qoder": False,
    "full_episode_ask_codex": True,
    "full_episode_ask_qoder": True,
}


def apply(ctx: Context, config: Mapping[str, Any]) -> None:
    ctx.invariants.register(PACKAGE_NAME, install)
    driver = LegacyCampaignDriver(build_campaign(config))
    ctx.provide(DRIVER, driver)


__all__ = [
    "Config",
    "Defaults",
    "apply",
    "inject",
    "interpolate",
    "name",
    "optional_inject",
    "provide",
]
