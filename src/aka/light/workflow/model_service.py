"""Durable, bounded provider recovery; never manufactures an Episode outcome."""
from concurrent.futures import CancelledError
import random
import threading
import time

from aka.light.execution.recovery import raise_if_environment_blocked

MAX_RETRIES = 6
MAX_WAIT_SECONDS = 30 * 60


class ModelServicePaused(RuntimeError):
    """Operator intervention required; restarting must not replenish the budget."""


class ModelServiceRecovery:
    def __init__(self, active, save, *, clock=time.time, wait=None, jitter=random.uniform):
        self.active = active
        self.save = save
        self.clock = clock
        self.wait = wait or threading.Event().wait
        self.jitter = jitter
        self.state = active.setdefault("model_service", {})

    @property
    def pending(self):
        return self.state.get("status") in {"waiting", "retrying", "paused"}

    def check_paused(self):
        if self.state.get("status") == "paused":
            raise ModelServicePaused(
                "paused_model_service: " + self.state["reason"]
                + "; Episode preserved. Resolve provider availability before explicitly resetting recovery."
            )

    def pause(self, reason):
        self.state.update(status="paused", reason=reason)
        self.active["phase"] = "paused_model_service"
        self.save(self.active)
        self.check_paused()

    def checkpoint(self, *, session_id, active_seconds):
        self.state.update(session_id=session_id, active_seconds=active_seconds)
        self.save(self.active)

    def failed(self, failure):
        self.check_paused()
        now = self.clock()
        self.state.setdefault("first_failure_at", now)
        self.state.update(failure_kind=failure.kind, provider_error_code=failure.code)
        if failure.kind == "model_quota_exhausted":
            self.pause("provider quota exhausted: " + failure.code)
        retries = self.state.get("retry_count", 0)
        if retries >= MAX_RETRIES:
            self.pause(f"{MAX_RETRIES} model-service retries exhausted")
        delay = min(300.0, min(300.0, 30.0 * 2 ** retries) * self.jitter(0.8, 1.2))
        delay = max(delay, failure.retry_after)
        if now + delay >= self.state["first_failure_at"] + MAX_WAIT_SECONDS:
            self.pause("30-minute model-service recovery window exhausted")
        self.state.update(status="waiting", next_retry_at=now + delay)
        self.active["phase"] = "waiting_model_service"
        self.save(self.active)

    def before_retry(self, cancel=None):
        self.check_paused()
        if self.state.get("status") == "retrying":
            # The previous process reserved a retry, but its outcome is unknown.
            # Charge another attempt rather than replaying that reservation forever.
            from aka.legacy.agent.model_service import ModelServiceFailure
            self.failed(ModelServiceFailure(self.state["failure_kind"],
                                           self.state["provider_error_code"]))
        while True:
            if cancel is not None and cancel.is_set():
                raise CancelledError("Search coordinator stopped model-service recovery")
            raise_if_environment_blocked()
            now = self.clock()
            if now >= self.state["first_failure_at"] + MAX_WAIT_SECONDS:
                self.pause("30-minute model-service recovery window exhausted")
            remaining = self.state["next_retry_at"] - now
            if remaining <= 0:
                break
            (cancel.wait if cancel is not None else self.wait)(min(1.0, remaining))
        retries = self.state.get("retry_count", 0)
        if retries >= MAX_RETRIES:
            self.pause(f"{MAX_RETRIES} model-service retries exhausted")
        self.state.update(status="retrying", retry_count=retries + 1)
        self.active["phase"] = "exploring"
        self.save(self.active)  # Reserve before invoking the provider.

    def observed_non_service_result(self):
        if not self.pending:
            return
        # Do not replenish the per-Episode retry cap, even after useful execution.
        self.state["status"] = "idle"
        self.state.pop("next_retry_at", None)
        self.state.pop("first_failure_at", None)
        self.active["phase"] = "exploring"
        self.save(self.active)
