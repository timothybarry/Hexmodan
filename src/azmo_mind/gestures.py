from __future__ import annotations

from dataclasses import dataclass

from azmo_mind.schemas import GestureCommand


@dataclass(frozen=True)
class TimelineEvent:
    time_ms: int
    description: str


GESTURE_PHASES: dict[str, list[tuple[float, str]]] = {
    "none": [(0.0, "No chassis motion requested.")],
    "neutral": [(0.0, "Begin return to neutral stance."), (1.0, "Neutral stance reached.")],
    "listen": [(0.0, "Freeze locomotion."), (0.35, "Orient toward speaker."), (1.0, "Hold.")],
    "survey": [
        (0.0, "Raise chassis slightly."),
        (0.25, "Begin slow battlefield scan."),
        (0.75, "Complete scan."),
        (1.0, "Face speaker or neutral bearing."),
    ],
    "loom": [
        (0.0, "Widen stance."),
        (0.18, "Lower body."),
        (0.35, "Pitch chassis forward."),
        (0.58, "Advance one slow, bounded step."),
        (1.0, "Hold looming pose."),
    ],
    "recoil": [
        (0.0, "Shift center of mass backward."),
        (0.45, "Take bounded retreat step."),
        (1.0, "Stabilize."),
    ],
    "stomp": [
        (0.0, "Prepare controlled front-leg lift."),
        (0.45, "Lift front leg."),
        (0.72, "Controlled plant."),
        (1.0, "Redistribute weight."),
    ],
    "boast": [(0.0, "Raise body."), (0.35, "Widen stance."), (1.0, "Hold elevated posture.")],
    "enthrone": [
        (0.0, "Set a broad, symmetrical stance."),
        (0.30, "Raise chassis deliberately."),
        (0.65, "Angle front legs outward."),
        (1.0, "Hold imperial posture."),
    ],
    "contempt": [(0.0, "Rotate slightly away."), (0.45, "Pause without approaching."), (1.0, "Hold.")],
    "rage": [
        (0.0, "Lower stance."),
        (0.25, "Begin bounded weight shifts."),
        (0.75, "Stop weight shifts."),
        (1.0, "Stabilize."),
    ],
    "circle": [(0.0, "Begin slow lateral orbit."), (1.0, "End orbit and stabilize.")],
    "dismiss": [(0.0, "Rotate away."), (0.55, "Begin slow departure."), (1.0, "Stop.")],
    "victory": [
        (0.0, "Raise to configured safe maximum."),
        (0.45, "Widen stance."),
        (1.0, "Hold triumphant posture."),
    ],
    "approach": [(0.0, "Begin safe approach."), (1.0, "Stop at configured distance.")],
    "retreat": [(0.0, "Begin safe retreat."), (1.0, "Stop and stabilize.")],
}


def simulate(command: GestureCommand) -> list[TimelineEvent]:
    phases = GESTURE_PHASES.get(command.name, GESTURE_PHASES["none"])
    return [
        TimelineEvent(
            time_ms=round(command.duration_ms * fraction),
            description=description,
        )
        for fraction, description in phases
    ]
