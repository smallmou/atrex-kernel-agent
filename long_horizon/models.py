from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from orchestrator.agent_runtime.model import (
    AgentRuntimeCapabilities,
    NormalizedAgentEvent,
    TokenUsage,
)


TERMINAL_STATUSES = frozenset({"candidate_ready", "pivot", "blocked"})
STATE_RECENT_ATTEMPTS_LIMIT = 16
STATE_ROUTE_HISTORY_LIMIT = 8
STATE_ROUTE_COMPLETED_WORK_LIMIT = 16
STATE_ATTEMPT_FIELDS = (
    "episode",
    "version",
    "status",
    "accepted",
    "checkpoint_accepted",
    "effective_attempt",
    "violation",
    "episode_branch",
    "campaign_mode",
    "route_id",
    "route_ledger",
    "promotion_commit",
    "outcome_commit",
    "recovered_after_supervisor_interruption",
)


@dataclass(frozen=True)
class EpisodeHandoff:
    status: str
    candidate_commit: str = ""
    last_trial_commit: str = ""

    def as_dict(self) -> dict[str, str]:
        return {key: value for key, value in asdict(self).items() if value}


@dataclass(frozen=True)
class InvocationObservation:
    terminal_usage: TokenUsage
    events: tuple[NormalizedAgentEvent, ...]
    capabilities: AgentRuntimeCapabilities
    observation_errors: tuple[str, ...] = ()
    resume_usage_qualified: bool = False


@dataclass
class SessionResult:
    exit_status: int
    timed_out: bool
    tokens: int
    session_id: str
    resume_count: int
    handoff: EpisodeHandoff | None
    stdout_tail: str = ""
    stderr_tail: str = ""
    completion_diagnosis: str = ""
    invocations: tuple[InvocationObservation, ...] = ()


@dataclass(frozen=True)
class VerificationRun:
    revision: str
    repeat: int
    exit_code: int
    result: dict[str, Any] | None
    stdout_tail: str = ""
    stderr_tail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    gate: str
    candidate_latency_us: float | None
    incumbent_latency_us: float | None
    improvement_pct: float | None
    runs: list[VerificationRun] = field(default_factory=list)
    error: str = ""
    artifact: str = ""
    candidate_performance_score: float | None = None
    incumbent_performance_score: float | None = None
    policy: str = "strict_promotion"

    @property
    def passed(self) -> bool:
        return self.gate == "PASS" and not self.error

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "candidate_latency_us": self.candidate_latency_us,
            "incumbent_latency_us": self.incumbent_latency_us,
            "improvement_pct": self.improvement_pct,
            "candidate_performance_score": self.candidate_performance_score,
            "incumbent_performance_score": self.incumbent_performance_score,
            "runs": [run.as_dict() for run in self.runs],
            "error": self.error or None,
            "artifact": self.artifact or None,
            "policy": self.policy,
        }


@dataclass
class CandidateAssessment:
    violation: str = ""
    changed_paths: list[str] = field(default_factory=list)
    verification: VerificationResult | None = None
    promotion_verification: VerificationResult | None = None
    checkpoint_accepted: bool = False
    accepted: bool = False


@dataclass
class SupervisorState:
    episodes: int = 0
    accepted: int = 0
    rejected: int = 0
    pivoted: int = 0
    blocked: int = 0
    protocol_failures: int = 0
    interrupted: int = 0
    tokens: int = 0
    consecutive_without_promotion: int = 0
    effective_episodes: int = 0
    consecutive_effective_stall: int = 0
    mode: str = "normal"
    route_id: str = ""
    route_base_commit: str = ""
    route_head_commit: str = ""
    route_best_commit: str = ""
    route_started_episode: int = 0
    route_used_episodes: int = 0
    route_remaining_episodes: int = 0
    route_plan: dict[str, Any] = field(default_factory=dict)
    route_progress: dict[str, Any] = field(default_factory=dict)
    route_global_best_score: float | None = None
    route_best_score: float | None = None
    route_exit_reason: str = ""
    last_refactor_best_kernel: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: object) -> "SupervisorState":
        if not isinstance(value, dict):
            return cls()
        fields = cls.__dataclass_fields__
        state = cls(**{key: value[key] for key in fields if key in value})
        if state.mode not in {"normal", "refactor"}:
            state.mode = "normal"
        if not isinstance(state.route_plan, dict):
            state.route_plan = {}
        if not isinstance(state.route_progress, dict):
            state.route_progress = {}
        state.route_progress = _compact_route_progress(state.route_progress)
        if not isinstance(state.attempts, list):
            state.attempts = []
        if "effective_episodes" not in value:
            state.effective_episodes = sum(
                1 for attempt in state.attempts if _effective_attempt(attempt)
            )
        if "consecutive_effective_stall" not in value:
            stall = 0
            for attempt in state.attempts:
                if not _effective_attempt(attempt):
                    continue
                stall = 0 if attempt.get("accepted") else stall + 1
            state.consecutive_effective_stall = stall
        state.attempts = [
            _compact_attempt(attempt)
            for attempt in state.attempts[-STATE_RECENT_ATTEMPTS_LIMIT:]
            if isinstance(attempt, dict)
        ]
        return state

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["route_progress"] = _compact_route_progress(self.route_progress)
        value["attempts"] = [
            _compact_attempt(attempt)
            for attempt in self.attempts[-STATE_RECENT_ATTEMPTS_LIMIT:]
        ]
        return value

    def remember_attempt(self, attempt: dict[str, Any]) -> None:
        compact = _compact_attempt(attempt)
        episode = compact.get("episode")
        branch = compact.get("episode_branch")
        self.attempts = [
            existing
            for existing in self.attempts
            if not (
                existing.get("episode") == episode
                and existing.get("episode_branch") == branch
            )
        ]
        self.attempts.append(compact)
        self.attempts = self.attempts[-STATE_RECENT_ATTEMPTS_LIMIT:]


def _effective_attempt(attempt: object) -> bool:
    if not isinstance(attempt, dict):
        return False
    if attempt.get("effective_attempt") is True:
        return True
    if attempt.get("violation"):
        return False
    status = attempt.get("status")
    if status == "pivot":
        return True
    if status != "candidate_ready":
        return False
    verification = attempt.get("verification")
    return isinstance(verification, dict) and verification.get("gate") in {
        "PASS",
        "FAIL",
    }


def _compact_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    compact = {key: attempt[key] for key in STATE_ATTEMPT_FIELDS if key in attempt}
    if "effective_attempt" not in compact:
        verification = attempt.get("verification")
        if isinstance(verification, dict) and verification.get("gate") in {
            "PASS",
            "FAIL",
        }:
            compact["verification"] = {"gate": verification["gate"]}
    return compact


def _compact_route_progress(progress: dict[str, Any]) -> dict[str, Any]:
    completed_work = progress.get("completed_work")
    if not isinstance(completed_work, list):
        completed_work = []
    history = progress.get("history")
    if not isinstance(history, list):
        history = []
    history_count = progress.get("history_count", len(history))
    if (
        not isinstance(history_count, int)
        or isinstance(history_count, bool)
        or history_count < 0
    ):
        history_count = len(history)
    current_milestone = progress.get("current_milestone", "")
    if not isinstance(current_milestone, str):
        current_milestone = ""
    next_task = progress.get("next_task", "")
    if not isinstance(next_task, str):
        next_task = ""
    return {
        "completed_work": completed_work[-STATE_ROUTE_COMPLETED_WORK_LIMIT:],
        "current_milestone": current_milestone,
        "next_task": next_task,
        "history": history[-STATE_ROUTE_HISTORY_LIMIT:],
        "history_count": max(history_count, len(history)),
    }
