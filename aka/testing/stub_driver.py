"""A campaign driver that records calls instead of running a campaign."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from aka.core.context import Context
from aka.seams.driver import CampaignDriver, CampaignRun, DRIVER

name = "stub-driver"
provide: tuple[str, ...] = ("driver",)

Config: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "workspace": {"type": "string"},
        "campaign_name": {"type": "string"},
        "status": {"type": "string", "enum": ["completed", "interrupted", "failed"]},
        "reason": {"type": "string"},
    },
}
Defaults: dict[str, Any] = {
    "workspace": "/tmp/aka-stub-workspace",
    "campaign_name": "stub",
    "status": "completed",
    "reason": "stub",
}
interpolate: tuple[str, ...] = ("workspace",)


class StubDriver(CampaignDriver):
    def __init__(self, config: Mapping[str, Any]):
        self._config = config
        self.calls: list[str] = []
        self.prepared = 0

    @property
    def workspace(self) -> Path:
        return Path(self._config["workspace"])

    @property
    def campaign_name(self) -> str:
        return str(self._config["campaign_name"])

    def run(self, *, on_prepared: Callable[[], None] | None = None) -> CampaignRun:
        self.calls.append("run")
        if on_prepared is not None:
            on_prepared()
            self.prepared += 1
        return CampaignRun(status=self._config["status"], reason=self._config["reason"])


def apply(ctx: Context, config: Mapping[str, Any]) -> None:
    ctx.provide(DRIVER, StubDriver(config))


__all__ = ["Config", "Defaults", "StubDriver", "apply", "interpolate", "name", "provide"]
