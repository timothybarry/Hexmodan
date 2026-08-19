"""Deterministic motion limits applied after the model has spoken.

This module governs **servos, not speech**. It never reads or writes
``response.speech`` - what AZMO says, and how foully he says it, is decided by
``prompts.py`` and the character config, and nothing here touches it.

What it does is bound the one field that can reach physical hardware. The model
proposes a named gesture with an intensity and a duration; this clamps both to
the limits in ``MotionConfig`` before the command envelope is built. That matters
because a ``rage`` at full intensity held for ten seconds is how you strip gears
or put the chassis on its face, and the model has no idea what the servos can
survive.

Historically this file also scanned the *user's* text for phrases like
"full power" and silently zeroed the gesture on a match. That was removed in
0.2.12: it was crude substring matching that fired on innocent sentences ("the
PSU delivers full power"), it inspected the wrong side of the conversation, and
it failed silently, so a randomly dead gesture was almost impossible to trace.
The clamps below do the real work, and they do it without guessing at intent.
"""

from __future__ import annotations

from azmo_mind.config import MotionConfig
from azmo_mind.schemas import AzmoResponse


def arbitrate(response: AzmoResponse, user_text: str, config: MotionConfig) -> AzmoResponse:
    """Return a copy of ``response`` whose gesture is inside the motion envelope.

    ``user_text`` is accepted for signature stability and is deliberately unused:
    what the user asked for is not evidence about what the servos can take.
    """
    safe = response.model_copy(deep=True)
    safe.gesture.intensity = min(safe.gesture.intensity, config.max_intensity)
    safe.gesture.duration_ms = max(
        config.min_duration_ms,
        min(config.max_duration_ms, safe.gesture.duration_ms),
    )

    if not config.hardware_enabled:
        suffix = "Hardware disabled; gesture is simulation-only."
        safe.internal_note = f"{safe.internal_note} {suffix}".strip()

    return safe
