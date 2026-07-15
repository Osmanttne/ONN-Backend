"""Shared plumbing for all bench devices."""
from __future__ import annotations

import enum
import logging
import pathlib
import yaml

log = logging.getLogger("onn")


class DeviceState(enum.Enum):
    DISCONNECTED = "disconnected"
    READY = "ready"
    RUNNING = "running"
    FAULT = "fault"


class Device:
    """Minimal interface every driver (real or simulated) implements."""

    name: str = "device"

    def __init__(self):
        self.state = DeviceState.DISCONNECTED

    # -- lifecycle -------------------------------------------------
    def connect(self):
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError

    def status(self) -> dict:
        return {"state": self.state.value}

    # -- helpers ---------------------------------------------------
    def _require_ready(self):
        if self.state in (DeviceState.DISCONNECTED, DeviceState.FAULT):
            raise RuntimeError(f"{self.name}: not connected (state={self.state.value})")


class SafetyLockError(RuntimeError):
    """Raised when a write is blocked by a safety interlock (e.g. SLM coverglass)."""


def load_profile(path: str | pathlib.Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
