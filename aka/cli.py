"""The CLI-to-composition boundary.

The supported entry point keeps its argparse surface and every cross-validation it already
performs; this module turns the validated values into the top composition layer. Flag values
therefore reach a plugin as schema-validated config, and ``--dump-config`` can show exactly
which layer set each field.

There is deliberately no seam for contributing flags. CLI compatibility is a hard constraint,
and a plugin-contributed flag would make ``--help`` ordering and cross-flag validation depend
on which composition happens to be selected.

The binding table below is data so the generated catalog can publish it and a verify script can
assert that every field a row accepts is reachable from the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from aka.boot import DEFAULT_PROFILE, PROFILES_DIR, compose
from aka.core.composition import ResolvedComposition, dump
from aka.plugins.legacy_campaign.service import UNASSIGNED_GPU

CAMPAIGN_ENTRY = "legacy-campaign"


@dataclass(frozen=True)
class FlagBinding:
    """One CLI value's destination in the plugin tree."""

    entry_id: str
    field: str
    flag: str = ""
    dest: str = ""
    note: str = ""
    transform: Callable[[Any], Any] | None = None


def _optional_gpu(value: Any) -> int:
    return UNASSIGNED_GPU if value is None else int(value)


#: Every campaign configuration value, and where it lands. ``flag`` is empty for values the
#: entry point derives rather than parses.
FLAG_BINDINGS: tuple[FlagBinding, ...] = (
    FlagBinding(CAMPAIGN_ENTRY, "name", note="derived from --op-dir"),
    FlagBinding(CAMPAIGN_ENTRY, "kernel_demo", note="derived from --op-dir"),
    FlagBinding(CAMPAIGN_ENTRY, "atrex_bench_root", note="derived from --op-dir"),
    FlagBinding(CAMPAIGN_ENTRY, "platform", flag="--platform", dest="platform"),
    FlagBinding(CAMPAIGN_ENTRY, "framework", flag="--framework", dest="framework"),
    FlagBinding(CAMPAIGN_ENTRY, "arch", flag="--arch", dest="arch", note="or autodetected"),
    FlagBinding(CAMPAIGN_ENTRY, "notes", flag="--notes", dest="notes"),
    FlagBinding(CAMPAIGN_ENTRY, "work_dir", flag="--workspace", dest="workspace"),
    FlagBinding(
        CAMPAIGN_ENTRY,
        "workspace_suffix",
        flag="--workspace-suffix",
        dest="workspace_suffix",
        note="or derived from framework/platform/mode",
    ),
    FlagBinding(CAMPAIGN_ENTRY, "max_iters", flag="--max-iters", dest="max_iters"),
    FlagBinding(CAMPAIGN_ENTRY, "token_budget", flag="--token-budget", dest="token_budget"),
    FlagBinding(CAMPAIGN_ENTRY, "target_util", flag="--target-util", dest="target_util"),
    FlagBinding(CAMPAIGN_ENTRY, "setup_timeout", flag="--setup-timeout", dest="setup_timeout"),
    FlagBinding(CAMPAIGN_ENTRY, "max_stall", flag="--max-stall", dest="max_stall"),
    FlagBinding(CAMPAIGN_ENTRY, "fast_episodes", flag="--fast-episodes", dest="fast_episodes"),
    FlagBinding(CAMPAIGN_ENTRY, "fast_trials", flag="--fast-trials", dest="fast_trials"),
    FlagBinding(CAMPAIGN_ENTRY, "convert_after", flag="--convert-after", dest="convert_after"),
    FlagBinding(
        CAMPAIGN_ENTRY, "handoff_resumes", flag="--handoff-resumes", dest="handoff_resumes"
    ),
    FlagBinding(
        CAMPAIGN_ENTRY,
        "sandbox_hardware",
        flag="--sandbox-hardware",
        dest="sandbox_hardware",
        note="defaults to --platform",
    ),
    FlagBinding(
        CAMPAIGN_ENTRY, "sandbox_profile", flag="--sandbox-profile", dest="sandbox_profile"
    ),
    FlagBinding(CAMPAIGN_ENTRY, "sandbox_url", flag="--sandbox-url", dest="sandbox_url"),
    FlagBinding(
        CAMPAIGN_ENTRY, "sandbox_timeout", flag="--sandbox-timeout", dest="sandbox_timeout"
    ),
    FlagBinding(CAMPAIGN_ENTRY, "sandbox_ssh", flag="--sandbox-ssh", dest="sandbox_ssh"),
    FlagBinding(
        CAMPAIGN_ENTRY, "sandbox_ssh_init", flag="--sandbox-ssh-init", dest="sandbox_ssh_init"
    ),
    FlagBinding(
        CAMPAIGN_ENTRY,
        "sandbox_ssh_gpu",
        flag="--sandbox-ssh-gpu",
        dest="sandbox_ssh_gpu",
        note="-1 means unassigned",
        transform=_optional_gpu,
    ),
    FlagBinding(
        CAMPAIGN_ENTRY,
        "sandbox_health_command",
        flag="--sandbox-health-command",
        dest="sandbox_health_command",
    ),
    FlagBinding(CAMPAIGN_ENTRY, "agent_cli", flag="--agent-cli", dest="agent_cli"),
    FlagBinding(
        CAMPAIGN_ENTRY,
        "long_reviewer_session",
        flag="--long-reviewer-session",
        dest="long_reviewer_session",
    ),
    FlagBinding(
        CAMPAIGN_ENTRY,
        "optimization_mode",
        flag="--optimization-mode",
        dest="optimization_mode",
    ),
    FlagBinding(
        CAMPAIGN_ENTRY,
        "framework_baseline",
        flag="--framework-baseline",
        dest="framework_baseline",
    ),
    FlagBinding(
        CAMPAIGN_ENTRY,
        "framework_baseline_timeout",
        flag="--framework-baseline-timeout",
        dest="framework_baseline_timeout",
    ),
    FlagBinding(
        CAMPAIGN_ENTRY, "numerical_gate", flag="--numerical-gate", dest="numerical_gate"
    ),
    FlagBinding(
        CAMPAIGN_ENTRY,
        "repair_numerical_head",
        flag="--repair-numerical-head",
        dest="repair_numerical_head",
    ),
    FlagBinding(
        CAMPAIGN_ENTRY,
        "numerical_review_timeout",
        flag="--numerical-review-timeout",
        dest="numerical_review_timeout",
    ),
    FlagBinding(
        CAMPAIGN_ENTRY,
        "production_review_timeout",
        flag="--production-review-timeout",
        dest="production_review_timeout",
    ),
    FlagBinding(
        CAMPAIGN_ENTRY, "verify_repeats", flag="--verify-repeats", dest="verify_repeats"
    ),
    FlagBinding(
        CAMPAIGN_ENTRY,
        "verify_run_timeout",
        flag="--verify-run-timeout",
        dest="verify_run_timeout",
    ),
    FlagBinding(
        CAMPAIGN_ENTRY,
        "min_improvement_pct",
        flag="--min-improvement-pct",
        dest="min_improvement_pct",
    ),
    FlagBinding(CAMPAIGN_ENTRY, "v1_ask_codex", flag="--v1-ask-codex", dest="v1_ask_codex"),
    FlagBinding(CAMPAIGN_ENTRY, "v1_ask_qoder", flag="--v1-ask-qoder", dest="v1_ask_qoder"),
    FlagBinding(
        CAMPAIGN_ENTRY,
        "fast_episode_ask_codex",
        flag="--fast-episode-ask-codex",
        dest="fast_episode_ask_codex",
    ),
    FlagBinding(
        CAMPAIGN_ENTRY,
        "fast_episode_ask_qoder",
        flag="--fast-episode-ask-qoder",
        dest="fast_episode_ask_qoder",
    ),
    FlagBinding(
        CAMPAIGN_ENTRY,
        "full_episode_ask_codex",
        flag="--full-episode-ask-codex",
        dest="full_episode_ask_codex",
    ),
    FlagBinding(
        CAMPAIGN_ENTRY,
        "full_episode_ask_qoder",
        flag="--full-episode-ask-qoder",
        dest="full_episode_ask_qoder",
    ),
)

#: Flags resolved before a tree exists: they decide which workspace and which composition to
#: boot, so they cannot be a plugin's config.
PRE_BOOT_FLAGS: tuple[str, ...] = (
    "--op-dir",
    "--sandbox-ssh-runtime-bind",
    "--environment-poll-interval",
)

_BINDINGS_BY_FIELD: Mapping[tuple[str, str], FlagBinding] = {
    (binding.entry_id, binding.field): binding for binding in FLAG_BINDINGS
}


def bound_fields(entry_id: str) -> frozenset[str]:
    return frozenset(
        binding.field for binding in FLAG_BINDINGS if binding.entry_id == entry_id
    )


def config_patch(entry_id: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """Build one whole-config patch for ``entry_id`` from field values.

    Every key must be a bound field, so a renamed campaign field cannot silently stop reaching
    the tree.
    """
    unbound = sorted(set(values) - bound_fields(entry_id))
    if unbound:
        raise KeyError(
            f'entry "{entry_id}" has no CLI binding for: {", ".join(unbound)}; '
            f"add it to FLAG_BINDINGS"
        )
    config: dict[str, Any] = {}
    for field, value in values.items():
        binding = _BINDINGS_BY_FIELD[(entry_id, field)]
        config[field] = binding.transform(value) if binding.transform else value
    return {"id": entry_id, "config": config}


def campaign_composition(
    values: Mapping[str, Any],
    *,
    profile: str = DEFAULT_PROFILE,
    profiles_dir: Path | None = None,
    patch_files: Sequence[Path] = (),
    variables: Mapping[str, str] | None = None,
) -> ResolvedComposition:
    """Compose the campaign tree from validated CLI values."""
    return compose(
        profile,
        profiles_dir=profiles_dir or PROFILES_DIR,
        patch_files=patch_files,
        patches=(config_patch(CAMPAIGN_ENTRY, values),),
        variables=variables,
    )


def dump_composition(composition: ResolvedComposition, *, fmt: str = "text") -> str:
    return dump(composition, fmt=fmt)


__all__ = [
    "CAMPAIGN_ENTRY",
    "FLAG_BINDINGS",
    "PRE_BOOT_FLAGS",
    "FlagBinding",
    "bound_fields",
    "campaign_composition",
    "config_patch",
    "dump_composition",
]
