import json

from azmo_mind.schemas import (
    AzmoResponse,
    coerce_response_payload,
    salvage_embedded_fields,
    sanitize_speech,
)


def test_normal_speech_is_untouched():
    text = "Your restraint is not virtue. It is merely appetite awaiting permission."
    assert sanitize_speech(text) == text


def test_speech_with_leaked_json_is_stripped():
    leaked = ('A waste of potential energy.", "gesture": {"name": "survey", '
              '"intensity": 0.65}, "voice": {"pace": 1.1}')
    assert sanitize_speech(leaked) == "A waste of potential energy."


def test_speech_wrapped_as_object_is_unwrapped():
    leaked = '{"speech": "The throne endures.", "emotion": "commanding"}'
    assert sanitize_speech(leaked) == "The throne endures."


def test_azmo_never_speaks_json_end_to_end():
    # The exact failure: the model crams everything into the speech string.
    raw = ('{"speech": "A hound. A waste of potential energy.\\", \\"gesture\\": '
           '{\\"name\\": \\"survey\\", \\"intensity\\": 0.65, \\"duration_ms\\": 2500}, '
           '\\"voice\\": {\\"pace\\": 1.1}"}')
    payload = json.loads(raw)
    payload = salvage_embedded_fields(payload, raw)
    payload, _ = coerce_response_payload(payload)
    response = AzmoResponse.model_validate(payload)
    assert '"gesture"' not in response.speech and "{" not in response.speech
    assert response.speech == "A hound. A waste of potential energy."
    # and the leaked gesture is recovered, not lost to the default
    assert response.gesture.name == "survey"
    assert response.gesture.duration_ms == 2500
    assert response.voice.pace == 1.1


def test_out_of_range_subharmonic_is_clamped_not_rejected():
    """Reproduces the real qwen3.5:9b failure: subharmonic_mix = 1.5."""
    payload = {
        "speech": "Nephalem. Speak quickly; my processors do not suffer hesitation.",
        "gesture": {"name": "enthrone", "duration_ms": 200, "target": "neutral"},
        "voice": {"emphasis_words": ["Nephalem"], "subharmonic_mix": 1.5},
    }
    repaired, notes = coerce_response_payload(payload)
    # It now validates instead of raising.
    response = AzmoResponse.model_validate(repaired)
    assert response.voice.subharmonic_mix == 0.25
    assert any("subharmonic_mix" in note for note in notes)


def test_valid_payload_reports_no_repairs():
    payload = {
        "speech": "The campaign proceeds.",
        "emotion": "commanding",
        "emotional_intensity": 0.5,
        "gesture": {"name": "loom", "intensity": 0.4, "duration_ms": 1500, "target": "speaker"},
        "voice": {"preset": "close_ominous", "pace": 0.9, "subharmonic_mix": 0.12},
    }
    repaired, notes = coerce_response_payload(payload)
    assert notes == []
    AzmoResponse.model_validate(repaired)


def test_unknown_enums_fall_back_to_safe_defaults():
    payload = {
        "speech": "Unknown terrain.",
        "emotion": "furious",            # not in the emotion enum
        "gesture": {"name": "obliterate", "target": "everyone"},  # not allowlisted
        "voice": {"preset": "screaming"},  # not a valid preset
    }
    repaired, notes = coerce_response_payload(payload)
    response = AzmoResponse.model_validate(repaired)
    assert response.emotion == "neutral"
    assert response.gesture.name == "none"           # unknown gesture -> safe no-motion
    assert response.gesture.target == "none"
    assert response.voice.preset == "calm_dark"
    assert len(notes) == 4


def test_numeric_clamps_cover_all_bounded_fields():
    payload = {
        "speech": "Bounds test.",
        "emotional_intensity": 9.0,
        "gesture": {"name": "rage", "intensity": 5.0, "duration_ms": 50},
        "voice": {"pace": 3.0, "pause_before_ms": 99999, "reverb_mix": -1.0},
    }
    repaired, _ = coerce_response_payload(payload)
    response = AzmoResponse.model_validate(repaired)
    assert response.emotional_intensity == 1.0
    assert response.gesture.intensity == 1.0
    assert response.gesture.duration_ms == 100
    assert response.voice.pace == 1.35
    assert response.voice.pause_before_ms == 3000
    assert response.voice.reverb_mix == 0.0


def test_non_dict_payload_is_returned_untouched():
    data, notes = coerce_response_payload("not a dict")
    assert data == "not a dict"
    assert notes == []
