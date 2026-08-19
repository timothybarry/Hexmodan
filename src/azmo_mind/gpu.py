"""GPU power governor.

Background: the dev PC hard-rebooted during the voice pipeline. The event log
showed Kernel-Power 41 with no thermal, TDR, or WHEA events at the crash times -
the signature of abrupt power loss, i.e. GPU current transients tripping an
aging PSU rather than a software fault. A new PSU is the real fix; until then
(and afterwards, as cheap insurance) AZMO can avoid provoking the spikes:

1. **Cap the board power** with ``nvidia-smi -pl``. This clips the transient
   peaks rather than the average framerate-style load AZMO produces, so the cost
   in inference speed is small. The cap is *temporary*: Windows resets it on
   reboot, and :func:`restore` puts it back explicitly.
2. **Stagger the workloads.** The LLM finishing and XTTS starting back-to-back
   is the sharpest ramp in the whole pipeline. :func:`stagger` inserts a short
   idle gap so the rail settles between them.
3. **Release VRAM** between turns so the 9B model and XTTS are not both holding
   peak allocations on a 12 GB card.

Everything here degrades to a no-op when nvidia-smi is missing, the process is
not elevated, or there is no NVIDIA GPU - AZMO must still run.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

_NVIDIA_SMI_TIMEOUT_S = 15


@dataclass(frozen=True)
class PowerState:
    """A snapshot of the GPU's power limits, in watts."""

    current: float | None = None
    default: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.current is not None


def nvidia_smi() -> str | None:
    """Path to nvidia-smi, or None if this machine has no NVIDIA tooling."""
    return shutil.which("nvidia-smi")


def _run(args: list[str]) -> tuple[int, str, str]:
    binary = nvidia_smi()
    if binary is None:
        return 127, "", "nvidia-smi not found on PATH"
    try:
        proc = subprocess.run(
            [binary, *args], capture_output=True, text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return 124, "", "nvidia-smi timed out"
    except OSError as exc:
        return 126, "", str(exc)
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def parse_power_csv(text: str) -> PowerState:
    """Parse ``power.limit,power.default_limit,power.min_limit,power.max_limit``.

    nvidia-smi renders these as e.g. ``250.00 W, 350.00 W, 100.00 W, 400.00 W``
    and prints ``[N/A]`` or ``[Not Supported]`` for fields it cannot read.
    """
    line = ""
    for candidate in text.splitlines():
        if candidate.strip():
            line = candidate
            break
    if not line:
        return PowerState(error="nvidia-smi returned no data")

    values: list[float | None] = []
    for field in line.split(","):
        cleaned = field.strip().rstrip("W").strip()
        try:
            values.append(float(cleaned))
        except ValueError:
            values.append(None)
    while len(values) < 4:
        values.append(None)
    return PowerState(current=values[0], default=values[1],
                      minimum=values[2], maximum=values[3])


def read_power() -> PowerState:
    """Current / default / min / max board power limits."""
    code, out, err = _run([
        "--query-gpu=power.limit,power.default_limit,power.min_limit,power.max_limit",
        "--format=csv,noheader,nounits",
    ])
    if code != 0:
        return PowerState(error=err or f"nvidia-smi exited {code}")
    return parse_power_csv(out)


@dataclass(frozen=True)
class MemoryState:
    """GPU memory in megabytes."""

    total: float | None = None
    used: float | None = None
    free: float | None = None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.total is not None


def parse_memory_csv(text: str) -> MemoryState:
    """Parse ``memory.total,memory.used,memory.free`` (nounits = MiB)."""
    line = next((c for c in text.splitlines() if c.strip()), "")
    if not line:
        return MemoryState(error="nvidia-smi returned no data")
    values: list[float | None] = []
    for field in line.split(","):
        try:
            values.append(float(field.strip().rstrip("MiB").strip()))
        except ValueError:
            values.append(None)
    while len(values) < 3:
        values.append(None)
    return MemoryState(total=values[0], used=values[1], free=values[2])


def read_memory() -> MemoryState:
    code, out, err = _run([
        "--query-gpu=memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ])
    if code != 0:
        return MemoryState(error=err or f"nvidia-smi exited {code}")
    return parse_memory_csv(out)


# Rough working-set sizes, in MiB, for the budget warning. These do not need to
# be exact - the point is to catch "these two will not fit" before the machine
# discovers it the hard way, by paging GPU memory over PCIe and taking the
# whole desktop down with it.
XTTS_WORKING_SET_MB = 3000       # ~2 GB of weights plus synthesis activations
DESKTOP_RESERVE_MB = 1200        # Windows compositor, browser, etc.
SAFE_HEADROOM_MB = 800


def vram_budget(total_mb: float, llm_used_mb: float,
                voice_mb: float = XTTS_WORKING_SET_MB) -> tuple[bool, str]:
    """Will the LLM and the voice model coexist without paging?

    When they do not fit, Windows' display driver starts evicting GPU memory to
    system RAM across PCIe. That is not a clean failure: the desktop becomes
    unresponsive, the GPU pins at maximum load moving memory around, and on a
    marginal power supply that sustained thrash is what precedes the crash.
    """
    required = llm_used_mb + voice_mb + DESKTOP_RESERVE_MB + SAFE_HEADROOM_MB
    spare = total_mb - required
    if spare >= 0:
        return True, (
            f"{total_mb / 1024:.1f} GB total: LLM {llm_used_mb / 1024:.1f} + "
            f"voice ~{voice_mb / 1024:.1f} + desktop ~{DESKTOP_RESERVE_MB / 1024:.1f} "
            f"= {spare / 1024:.1f} GB spare."
        )
    return False, (
        f"Over budget by ~{-spare / 1024:.1f} GB. {total_mb / 1024:.1f} GB total vs "
        f"LLM {llm_used_mb / 1024:.1f} + voice ~{voice_mb / 1024:.1f} + desktop "
        f"~{DESKTOP_RESERVE_MB / 1024:.1f} GB. Windows will page GPU memory over "
        "PCIe, which locks up the desktop and pins the GPU. Lower "
        "provider.context_tokens, use a smaller/more-quantized model, or set "
        "speech.clone_device: cpu."
    )


def is_elevated() -> bool:
    """True if this process can change the GPU power limit.

    On Windows that means running as administrator; on Linux, as root.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:  # noqa: BLE001 - assume not elevated
            return False
    try:
        import os

        return os.geteuid() == 0
    except AttributeError:  # pragma: no cover - exotic platform
        return False


def set_power_limit(watts: int) -> tuple[bool, str]:
    """Apply a temporary board power cap. Returns (applied, human message)."""
    state = read_power()
    if not state.available:
        return False, state.error or "No NVIDIA GPU detected; power cap skipped."

    if state.minimum is not None and watts < state.minimum:
        watts = int(state.minimum)
    if state.maximum is not None and watts > state.maximum:
        watts = int(state.maximum)
    if state.current is not None and abs(state.current - watts) < 1.0:
        return True, f"GPU power limit already {watts} W."

    code, _out, err = _run(["-pl", str(watts)])
    if code != 0:
        if not is_elevated():
            return False, (
                f"Could not cap the GPU to {watts} W: this needs administrator "
                "rights. Right-click GPU_POWER_SAFE.bat -> Run as administrator, "
                "or start AZMO from an elevated prompt."
            )
        return False, f"Could not cap the GPU to {watts} W: {err or code}"
    default = f"{state.default:.0f}" if state.default else "default"
    return True, (
        f"GPU power limit set to {watts} W (was {state.current:.0f} W, "
        f"stock {default} W). Temporary: a reboot restores full power."
    )


def restore(watts: float | None = None) -> tuple[bool, str]:
    """Put the power limit back to the stock default (or an explicit value)."""
    state = read_power()
    if not state.available:
        return False, state.error or "No NVIDIA GPU detected."
    target = watts if watts is not None else state.default
    if target is None:
        return False, "nvidia-smi did not report a default power limit."
    if state.current is not None and abs(state.current - target) < 1.0:
        return True, f"GPU already at full power ({target:.0f} W)."
    code, _out, err = _run(["-pl", str(int(target))])
    if code != 0:
        return False, f"Could not restore the GPU power limit: {err or code}"
    return True, f"GPU power limit restored to {target:.0f} W."


class PowerGovernor:
    """Applies the configured cap for the life of a session, then restores it.

    Never raises: a machine without an NVIDIA GPU, or a non-elevated process,
    simply gets a message and no cap.
    """

    def __init__(self, config, notify=None):
        self.config = config
        self.notify = notify or (lambda message, style=None: None)
        self.applied = False
        self._previous: float | None = None

    def start(self) -> None:
        watts = self.config.power_limit_watts
        if not self.config.apply_on_launch or watts is None:
            return
        state = read_power()
        if not state.available:
            return
        if state.current is not None and state.current <= watts + 1:
            self.notify(
                f"GPU already limited to {state.current:.0f} W; leaving it alone.",
                "dim",
            )
            return
        self._previous = state.current
        ok, message = set_power_limit(watts)
        self.applied = ok
        self.notify(message, "green" if ok else "yellow")
        if ok:
            self.notify(
                "REMINDER: your GPU is power-capped. Reboot (or run "
                "RESTORE_GPU_POWER.bat as admin) before gaming.",
                "bold yellow",
            )

    def stop(self) -> None:
        if not (self.applied and self.config.restore_on_exit):
            if self.applied:
                self.notify(
                    "GPU is still capped. Reboot or run RESTORE_GPU_POWER.bat "
                    "as admin before gaming.",
                    "bold yellow",
                )
            return
        ok, message = restore(self._previous)
        self.notify(message, "green" if ok else "yellow")
        self.applied = not ok

    def __enter__(self) -> "PowerGovernor":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Workload shaping
# ---------------------------------------------------------------------------

def stagger(milliseconds: int) -> None:
    """Idle gap between two GPU workloads, so their current ramps don't stack."""
    if milliseconds > 0:
        time.sleep(milliseconds / 1000)


def release_vram() -> bool:
    """Free torch's cached VRAM. Returns True if anything was released."""
    try:
        import torch
    except ImportError:
        return False
    if not (getattr(torch, "cuda", None) and torch.cuda.is_available()):
        return False
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - never fatal
        return False
    return True
