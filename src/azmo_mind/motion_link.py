"""Motion-link layer: the seed of the future azmo-motion-link service.

This module previews the Jetson-to-Teensy contract from the project brief
(section 10) so the rest of the software can develop against the real command
shape long before firmware exists:

    LLM -> performance intent -> behavior executive -> MOTION LINK -> Teensy

Boundary rules enforced here (and again in firmware later):

- Only allowlisted gesture names cross this boundary. Never joint angles,
  PWM values, torque, or gait timing.
- Every command carries an id, priority, and timeout. Duplicate ids must not
  repeat physical actions (idempotency), so the link tracks issued ids.
- Lifecycle: RECEIVED -> ACCEPTED | REJECTED -> EXECUTING -> COMPLETED | ABORTED.
- Speech-synchronized gestures run at priority 30 (brief section 11).

``SimulatedMotionLink`` is the only implementation today; it validates and
"executes" against the gesture phase timelines. ``SerialMotionLink`` is a
deliberate stub: the transport (USB CDC, heartbeats, COBS/CRC framing) arrives
with roadmap 0.6/0.7.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, get_args

from azmo_mind.gestures import TimelineEvent, simulate
from azmo_mind.schemas import GestureCommand, GestureName


PROTOCOL_VERSION = 1
GESTURE_PRIORITY = 30  # speech-synchronized gesture (brief section 11)

_ALLOWED_GESTURES: frozenset[str] = frozenset(get_args(GestureName))


class CommandState(str, Enum):
    RECEIVED = "RECEIVED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"


@dataclass(frozen=True)
class MotionCommand:
    """The exact envelope previewed in the brief, section 10."""

    version: int
    id: int
    type: str
    priority: int
    timeout_ms: int
    parameters: dict[str, Any]

    def to_wire(self) -> str:
        """Newline-delimited JSON: the early development transport format."""
        return json.dumps(
            {
                "version": self.version,
                "id": self.id,
                "type": self.type,
                "priority": self.priority,
                "timeout_ms": self.timeout_ms,
                "parameters": self.parameters,
            },
            separators=(", ", ": "),
        )


@dataclass(frozen=True)
class MotionResult:
    command: MotionCommand
    state: CommandState
    lifecycle: list[CommandState]
    timeline: list[TimelineEvent] = field(default_factory=list)
    reason: str = ""


class MotionLink:
    """Owns the path to the motor controller. The LLM never touches this."""

    def send_gesture(self, gesture: GestureCommand) -> MotionResult:
        raise NotImplementedError

    def health(self) -> dict[str, Any]:
        raise NotImplementedError


class SimulatedMotionLink(MotionLink):
    """Validates and simulates commands with the production lifecycle.

    Defense in depth: the safety arbiter has already clamped the gesture, but
    this layer re-checks the allowlist exactly the way Teensy firmware will.
    A bug upstream should be caught here, not at the servos.
    """

    def __init__(self, hardware_enabled: bool = False):
        # Hardware can never be enabled on the simulator; the flag is recorded
        # so callers can see what the config asked for.
        self.hardware_requested = hardware_enabled
        self._ids = itertools.count(1)
        self._seen_ids: set[int] = set()
        self.history: list[MotionResult] = []

    def _build(self, gesture: GestureCommand) -> MotionCommand:
        return MotionCommand(
            version=PROTOCOL_VERSION,
            id=next(self._ids),
            type="gesture",
            priority=GESTURE_PRIORITY,
            timeout_ms=gesture.duration_ms + 600,
            parameters={
                "name": gesture.name,
                "intensity": round(gesture.intensity, 3),
                "duration_ms": gesture.duration_ms,
                "target": gesture.target,
            },
        )

    def send_gesture(self, gesture: GestureCommand) -> MotionResult:
        command = self._build(gesture)
        lifecycle = [CommandState.RECEIVED]

        if command.id in self._seen_ids:
            # Idempotency: a duplicate id acknowledges but never re-executes.
            lifecycle.append(CommandState.REJECTED)
            result = MotionResult(
                command=command,
                state=CommandState.REJECTED,
                lifecycle=lifecycle,
                reason="duplicate command id",
            )
            self.history.append(result)
            return result
        self._seen_ids.add(command.id)

        name = str(command.parameters.get("name", ""))
        if name not in _ALLOWED_GESTURES:
            lifecycle.append(CommandState.REJECTED)
            result = MotionResult(
                command=command,
                state=CommandState.REJECTED,
                lifecycle=lifecycle,
                reason=f"gesture {name!r} is not in the allowlist",
            )
            self.history.append(result)
            return result

        lifecycle.append(CommandState.ACCEPTED)

        if name == "none":
            lifecycle.append(CommandState.COMPLETED)
            result = MotionResult(
                command=command,
                state=CommandState.COMPLETED,
                lifecycle=lifecycle,
                reason="no motion requested",
            )
            self.history.append(result)
            return result

        lifecycle.append(CommandState.EXECUTING)
        timeline = simulate(gesture)
        lifecycle.append(CommandState.COMPLETED)
        result = MotionResult(
            command=command,
            state=CommandState.COMPLETED,
            lifecycle=lifecycle,
            timeline=timeline,
        )
        self.history.append(result)
        return result

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "link": "simulated",
            "protocol_version": PROTOCOL_VERSION,
            "hardware_requested": self.hardware_requested,
            "hardware_enabled": False,
            "commands_sent": len(self._seen_ids),
        }


class SerialMotionLink(MotionLink):
    """Placeholder for the real USB CDC link to the Teensy 4.1 (roadmap 0.6+).

    Will own: the serial port, 50 ms bidirectional heartbeats, acks, retries
    with idempotent ids, link-health metrics, and reconnect handshakes. It must
    live in its own service/process — never inside the LLM process.
    """

    def __init__(self, port: str = "/dev/ttyACM0"):
        self.port = port

    def send_gesture(self, gesture: GestureCommand) -> MotionResult:  # pragma: no cover
        raise NotImplementedError(
            "SerialMotionLink arrives with roadmap 0.6/0.7 (Teensy firmware + protocol)."
        )

    def health(self) -> dict[str, Any]:  # pragma: no cover
        return {"ok": False, "link": "serial", "port": self.port, "error": "not implemented"}
