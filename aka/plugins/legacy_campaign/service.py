"""The campaign driver provider that wraps today's ``orchestrator.campaign.Campaign``.

This is the first step of the staged refactor: the seam exists and the host talks to it, while
the implementation is still the existing state machine, called in the existing order. Behavior
is identical by construction -- the sequence below is the one ``orchestrator/optimize.py`` ran
inline -- so the plugin core can be verified before any capability moves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from aka.seams.driver import CampaignDriver, CampaignRun

#: ``Campaign`` takes ``sandbox_ssh_gpu: int | None``. The config schema dialect has no null
#: type, so an unassigned GPU travels as -1.
UNASSIGNED_GPU = -1

#: Campaign constructor fields carried verbatim by this row's config.
CAMPAIGN_FIELDS: tuple[str, ...] = (
    "name",
    "kernel_demo",
    "platform",
    "framework",
    "notes",
    "arch",
    "work_dir",
    "workspace_suffix",
    "max_iters",
    "token_budget",
    "target_util",
    "setup_timeout",
    "max_stall",
    "fast_episodes",
    "fast_trials",
    "convert_after",
    "sandbox_hardware",
    "sandbox_profile",
    "sandbox_url",
    "sandbox_timeout",
    "atrex_bench_root",
    "agent_cli",
    "optimization_mode",
    "framework_baseline",
    "framework_baseline_timeout",
    "handoff_resumes",
    "numerical_gate",
    "repair_numerical_head",
    "numerical_review_timeout",
    "production_review_timeout",
    "verify_repeats",
    "verify_run_timeout",
    "min_improvement_pct",
    "long_reviewer_session",
    "v1_ask_codex",
    "v1_ask_qoder",
    "fast_episode_ask_codex",
    "fast_episode_ask_qoder",
    "full_episode_ask_codex",
    "full_episode_ask_qoder",
    "sandbox_ssh",
    "sandbox_ssh_init",
    "sandbox_ssh_gpu",
    "sandbox_health_command",
)


def campaign_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Translate this row's config into ``Campaign`` constructor arguments."""
    kwargs = {field: config[field] for field in CAMPAIGN_FIELDS if field in config}
    gpu = kwargs.get("sandbox_ssh_gpu", UNASSIGNED_GPU)
    kwargs["sandbox_ssh_gpu"] = None if gpu == UNASSIGNED_GPU else int(gpu)
    return kwargs


def build_campaign(config: Mapping[str, Any]) -> Any:
    """Construct the legacy campaign. Imported here so declaration checks stay cheap."""
    from orchestrator.campaign import Campaign

    return Campaign(**campaign_kwargs(config))


class LegacyCampaignDriver(CampaignDriver):
    """Provider: drives ``orchestrator.campaign.Campaign`` through one campaign."""

    def __init__(self, campaign: Any):
        self._campaign = campaign

    @property
    def campaign(self) -> Any:
        """The wrapped campaign, for consumers still reaching for legacy members."""
        return self._campaign

    @property
    def workspace(self) -> Path:
        return self._campaign.workspace

    @property
    def campaign_name(self) -> str:
        return self._campaign.campaign_name

    def run(self, *, on_prepared: Callable[[], None] | None = None) -> CampaignRun:
        from orchestrator.workspace_state import latest_version, read_memory

        campaign = self._campaign
        if latest_version(campaign.workspace) < 0:
            campaign.setup_baseline()
        else:
            print(
                f"[orchestrator] resuming workspace at v{latest_version(campaign.workspace)}",
                flush=True,
            )
            campaign._link_runtime()
        baseline_coverage_problem = campaign._generalized_memory_coverage_problem(
            read_memory(campaign.workspace, 0)
        )
        if baseline_coverage_problem:
            raise RuntimeError(
                "generalized campaign baseline is incompatible with authoritative per-shape "
                f"memory: {baseline_coverage_problem}; start a fresh workspace"
            )
        if on_prepared is not None:
            on_prepared()
        campaign.ensure_framework_baseline()
        reason = campaign.run()
        return CampaignRun(status="completed", reason=reason, exit_code=0)


__all__ = [
    "CAMPAIGN_FIELDS",
    "UNASSIGNED_GPU",
    "LegacyCampaignDriver",
    "build_campaign",
    "campaign_kwargs",
]
