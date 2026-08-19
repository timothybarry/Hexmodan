from __future__ import annotations

import re
from pathlib import Path

from azmo_mind.paths import resolve
from azmo_mind.schemas import EmotionState

BASELINE = EmotionState()


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class EmotionStateStore:
    def __init__(self, path: str | Path = "data/emotion_state.json"):
        # Anchored to the install root: his mood must not depend on which folder
        # he was started from, or each shortcut would give him a separate
        # personality that silently resets. See ``azmo_mind.paths``.
        self.path = resolve(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> EmotionState:
        if not self.path.exists():
            return EmotionState()
        try:
            return EmotionState.model_validate_json(self.path.read_text(encoding="utf-8"))
        except Exception:
            # State files from earlier versions are intentionally safe to discard.
            return EmotionState()

    def save(self, state: EmotionState) -> None:
        self.path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    def reset(self) -> EmotionState:
        state = EmotionState()
        self.save(state)
        return state


def update_state(state: EmotionState, user_text: str) -> EmotionState:
    """Bounded deterministic state updates; the LLM never directly owns state."""
    text = user_text.lower()
    values = state.model_dump()

    for key, baseline in BASELINE.model_dump().items():
        values[key] += (baseline - values[key]) * 0.08

    if re.search(r"\b(thank|great|awesome|love|proud|good job)\b", text):
        values["trust"] += 0.05
        values["amusement"] += 0.04

    if re.search(r"\b(stupid|idiot|useless|hate you|shut up)\b", text):
        values["irritation"] += 0.12
        values["dominance"] += 0.04
        values["trust"] -= 0.05

    if "?" in user_text or re.search(r"\b(why|how|what|explain|tell me|plan)\b", text):
        values["curiosity"] += 0.04
        values["calculation"] += 0.03

    if re.search(r"\b(want|desire|tempt|indulge|sin|pleasure|power)\b", text):
        values["temptation"] += 0.06

    if re.search(r"\b(unsafe|danger|emergency|stop)\b", text):
        values["energy"] -= 0.08
        values["calculation"] += 0.07
        values["dominance"] += 0.04

    if re.search(r"\b(victory|won|success|awaken|rise|nephalem)\b", text):
        values["energy"] += 0.08
        values["dominance"] += 0.04
        values["amusement"] += 0.02

    return EmotionState(**{key: _clamp(value) for key, value in values.items()})
