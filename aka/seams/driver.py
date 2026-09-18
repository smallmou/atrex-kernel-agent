"""The campaign driver seam.

One provider owns the campaign state machine: establish or resume the workspace, establish the
framework baseline, run optimization episodes, finalize. The host calls it; it never calls the
host back except through :class:`CampaignDriver.run`'s ``on_prepared`` hook, which exists
because the SSH recovery monitor may only declare success after the workspace is resolved and
before any GPU work starts.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import seam

#: Terminal campaign statuses recorded in the trace-retention manifest.
STATUSES: tuple[str, ...] = ("completed", "interrupted", "failed")


@dataclass(frozen=True)
class CampaignRun:
    """The outcome of one campaign."""

    status: str
    reason: str = ""
    exit_code: int = 0

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unsupported campaign status: {self.status!r}")


class CampaignDriver(abc.ABC):
    """Definition: the single-implementation campaign state machine."""

    @property
    @abc.abstractmethod
    def workspace(self) -> Path:
        """The campaign workspace directory, valid before ``run`` is called."""

    @property
    @abc.abstractmethod
    def campaign_name(self) -> str:
        """The workspace-qualified campaign name."""

    @abc.abstractmethod
    def run(self, *, on_prepared: Callable[[], None] | None = None) -> CampaignRun:
        """Establish or resume the workspace, then drive the campaign to a terminal state.

        ``on_prepared`` runs once the workspace is resolved and validated and before the
        framework baseline starts.
        """


DRIVER = seam(
    "driver",
    "CampaignDriver",
    "single",
    module="aka.seams.driver",
    doc="The campaign state machine: baseline, framework baseline, episodes, finalize.",
)


__all__ = ["CampaignDriver", "CampaignRun", "DRIVER", "STATUSES"]
