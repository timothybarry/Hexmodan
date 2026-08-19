from __future__ import annotations

import json
import re
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, field_validator


# A JSON key for one of our fields, e.g. `"gesture":` — its appearance inside the
# spoken text means the model leaked structured output into the speech string.
_LEAK_KEY_RE = re.compile(
    r'"(?:speech|gesture|voice|emotion|emotional_intensity|internal_note|name|'
    r'intensity|duration_ms|target|preset|pace|pause_before_ms|emphasis_words|'
    r'subharmonic_mix|reverb_mix|pitch[_ ]?shift[_ ]?semitones)"\s*:'
)


def sanitize_speech(text: str) -> str:
    """Return only the words AZMO should say, stripping any leaked JSON.

    Local models occasionally cram the whole structured response into the
    ``speech`` string (or wrap it as ``{"speech": "..."}``). Left alone, AZMO
    literally reads field names and braces aloud. If — and only if — we detect
    that leakage, we recover the spoken prefix and drop the JSON tail. Normal
    speech is returned untouched.
    """
    t = text.strip()
    leaked = t.startswith("{") or _LEAK_KEY_RE.search(t) is not None
    if not leaked:
        return text
    # Drop a leading `{"speech": "` wrapper if present.
    t = re.sub(r'^\s*\{?\s*"speech"\s*:\s*"?', "", t)
    # Cut at the first leaked field key.
    match = _LEAK_KEY_RE.search(t)
    if match:
        t = t[: match.start()]
    # Trim trailing JSON punctuation / stray quotes left at the cut.
    t = t.strip()
    t = re.sub(r'\\+$', "", t).strip()
    t = re.sub(r'["\',{}\s]+$', "", t).strip()
    return t or text


EmotionName = Literal[
    "neutral",
    "amused",
    "curious",
    "calculating",
    "tempting",
    "contemptuous",
    "commanding",
    "protective",
    "irritated",
    "wrathful",
    "solemn",
    "triumphant",
]

GestureName = Literal[
    "none",
    "neutral",
    "listen",
    "survey",
    "loom",
    "recoil",
    "stomp",
    "boast",
    "enthrone",
    "contempt",
    "rage",
    "circle",
    "dismiss",
    "victory",
    "approach",
    "retreat",
]

VoicePreset = Literal[
    "calm_dark",
    "close_ominous",
    "imperial_decree",
    "temptation",
    "contempt",
    "restrained_rage",
    "dark_amusement",
    "solemn",
    "victory",
]


class GestureCommand(BaseModel):
    name: GestureName = "none"
    intensity: float = Field(default=0.0, ge=0, le=1)
    duration_ms: int = Field(default=1000, ge=100, le=10000)
    target: Literal["speaker", "neutral", "none"] = "none"


class VoiceDirection(BaseModel):
    preset: VoicePreset = "calm_dark"
    pace: float = Field(default=0.92, ge=0.6, le=1.35)
    pause_before_ms: int = Field(default=0, ge=0, le=3000)
    emphasis_words: list[str] = Field(default_factory=list, max_length=5)
    subharmonic_mix: float = Field(default=0.10, ge=0, le=0.25)
    reverb_mix: float = Field(default=0.08, ge=0, le=0.25)

    @field_validator("emphasis_words")
    @classmethod
    def clean_emphasis(cls, words: list[str]) -> list[str]:
        cleaned: list[str] = []
        for word in words:
            word = word.strip()
            if word and word not in cleaned:
                cleaned.append(word[:40])
        return cleaned[:5]


class AzmoResponse(BaseModel):
    speech: str = Field(min_length=1, max_length=1400)
    emotion: EmotionName = "neutral"
    emotional_intensity: float = Field(default=0.4, ge=0, le=1)
    gesture: GestureCommand = Field(default_factory=GestureCommand)
    voice: VoiceDirection = Field(default_factory=VoiceDirection)
    internal_note: str = Field(
        default="",
        max_length=240,
        description="Brief production note; never spoken aloud.",
    )

    @field_validator("speech")
    @classmethod
    def clean_speech(cls, speech: str) -> str:
        speech = sanitize_speech(speech)
        speech = " ".join(speech.strip().split())
        if not speech:
            raise ValueError("speech cannot be empty")
        return speech


class EmotionState(BaseModel):
    dominance: float = Field(default=0.78, ge=0, le=1)
    amusement: float = Field(default=0.30, ge=0, le=1)
    irritation: float = Field(default=0.08, ge=0, le=1)
    curiosity: float = Field(default=0.50, ge=0, le=1)
    temptation: float = Field(default=0.44, ge=0, le=1)
    calculation: float = Field(default=0.66, ge=0, le=1)
    trust: float = Field(default=0.62, ge=0, le=1)
    energy: float = Field(default=0.66, ge=0, le=1)


# ---------------------------------------------------------------------------
# Model-output repair (provider boundary)
# ---------------------------------------------------------------------------

_VALID_EMOTIONS = frozenset(get_args(EmotionName))
_VALID_GESTURES = frozenset(get_args(GestureName))
_VALID_PRESETS = frozenset(get_args(VoicePreset))
_VALID_TARGETS = frozenset(("speaker", "neutral", "none"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def coerce_response_payload(data: Any) -> tuple[Any, list[str]]:
    """Repair a raw model payload so recoverable schema violations become safe,
    valid values instead of discarding the whole turn.

    Local models do not reliably honor numeric minimum/maximum or enum
    membership even under Ollama structured output, because grammar enforcement
    covers JSON shape and enum tokens but not value ranges. This boundary
    function clamps out-of-range numbers and replaces unknown enum values with
    safe defaults (an unknown gesture becomes ``none``), returning the repaired
    payload plus human-readable notes. Strict ``AzmoResponse`` validation still
    runs afterward as the final gate, so anything unrepairable still surfaces.
    """
    if not isinstance(data, dict):
        return data, []

    notes: list[str] = []
    repaired = dict(data)

    def clamp_float(container: dict, key: str, low: float, high: float, label: str) -> None:
        value = container.get(key)
        if not _is_number(value):
            return
        clamped = max(low, min(high, float(value)))
        if clamped != float(value):
            notes.append(f"{label} {value} -> {clamped}")
            container[key] = clamped

    def clamp_int(container: dict, key: str, low: int, high: int, label: str) -> None:
        value = container.get(key)
        if not _is_number(value):
            return
        clamped = max(low, min(high, int(round(float(value)))))
        if clamped != value:
            notes.append(f"{label} {value} -> {clamped}")
            container[key] = clamped

    clamp_float(repaired, "emotional_intensity", 0.0, 1.0, "emotional_intensity")
    emotion = repaired.get("emotion")
    if isinstance(emotion, str) and emotion not in _VALID_EMOTIONS:
        notes.append(f"emotion {emotion!r} -> 'neutral'")
        repaired["emotion"] = "neutral"

    gesture = repaired.get("gesture")
    if isinstance(gesture, dict):
        gesture = dict(gesture)
        name = gesture.get("name")
        if isinstance(name, str) and name not in _VALID_GESTURES:
            notes.append(f"gesture.name {name!r} -> 'none'")
            gesture["name"] = "none"
        clamp_float(gesture, "intensity", 0.0, 1.0, "gesture.intensity")
        clamp_int(gesture, "duration_ms", 100, 10000, "gesture.duration_ms")
        target = gesture.get("target")
        if isinstance(target, str) and target not in _VALID_TARGETS:
            notes.append(f"gesture.target {target!r} -> 'none'")
            gesture["target"] = "none"
        repaired["gesture"] = gesture

    voice = repaired.get("voice")
    if isinstance(voice, dict):
        voice = dict(voice)
        preset = voice.get("preset")
        if isinstance(preset, str) and preset not in _VALID_PRESETS:
            notes.append(f"voice.preset {preset!r} -> 'calm_dark'")
            voice["preset"] = "calm_dark"
        clamp_float(voice, "pace", 0.6, 1.35, "voice.pace")
        clamp_int(voice, "pause_before_ms", 0, 3000, "voice.pause_before_ms")
        clamp_float(voice, "subharmonic_mix", 0.0, 0.25, "voice.subharmonic_mix")
        clamp_float(voice, "reverb_mix", 0.0, 0.25, "voice.reverb_mix")
        repaired["voice"] = voice

    return repaired, notes


def salvage_embedded_fields(payload: Any, raw: str) -> Any:
    """Recover gesture/voice/emotion that a model buried in the raw text.

    When the model crams everything into the ``speech`` string, the top-level
    ``gesture``/``voice`` keys go missing and their intent (e.g. a ``survey``
    gesture) is lost. If they aren't already present, pull them out of the raw
    model text so the performance still happens. Best-effort; never raises.
    """
    if not isinstance(payload, dict):
        return payload
    # Search the raw text AND the (already-unescaped) speech value, since a model
    # that escapes everything into the speech string hides the recoverable
    # `"gesture": {...}` there rather than in the raw bytes.
    haystack = raw
    spoken = payload.get("speech")
    if isinstance(spoken, str):
        haystack = raw + "\n" + spoken
    for key in ("gesture", "voice"):
        current = payload.get(key)
        if isinstance(current, dict):
            continue
        # gesture/voice objects are flat (no nested braces).
        match = re.search(r'"%s"\s*:\s*(\{[^{}]*\})' % key, haystack)
        if match:
            try:
                payload[key] = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    if not isinstance(payload.get("emotion"), str) or payload.get("emotion") == "neutral":
        match = re.search(r'"emotion"\s*:\s*"([a-z_]+)"', haystack)
        if match:
            payload["emotion"] = match.group(1)
    return payload
