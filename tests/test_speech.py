from pathlib import Path

from azmo_mind.config import SpeechConfig
from azmo_mind.speech import (
    NullSpeech,
    _temp_wav,
    effective_pace,
    pace_to_espeak_wpm,
    pace_to_piper_length_scale,
    pace_to_sapi_rate,
    select_speech_adapter,
)
from azmo_mind.schemas import VoiceDirection


def test_speed_multiplier_clamps():
    assert effective_pace(1.3, 2.0) == 1.35          # clamped to the upper bound
    assert effective_pace(0.5, 0.5) == 0.6           # clamped to the lower bound


def test_pace_stays_in_a_natural_band():
    # Whatever the model picks across its full range, delivery stays near normal
    # (news-anchor), never dragging or racing.
    for model_pace in (0.6, 0.75, 0.92, 1.0, 1.15, 1.35):
        assert 0.9 <= effective_pace(model_pace, 1.05) <= 1.2


def test_temp_wav_does_not_exist_yet_and_is_unique():
    # Regression: a pre-created temp file got briefly locked on Windows, so SAPI
    # could not write it. The path must not exist until the synthesizer makes it.
    a = _temp_wav()
    b = _temp_wav()
    assert a != b
    assert not a.exists()
    a.write_bytes(b"RIFF")          # a fresh writer can create it
    try:
        assert a.exists()
    finally:
        a.unlink()


def test_neutral_pace_maps_to_neutral_rates():
    assert pace_to_sapi_rate(1.0) == 0
    assert pace_to_espeak_wpm(1.0) == 150
    assert pace_to_piper_length_scale(1.0) == 1.0


def test_slow_pace_slows_every_engine():
    assert pace_to_sapi_rate(0.7) < 0
    assert pace_to_espeak_wpm(0.7) < 150
    assert pace_to_piper_length_scale(0.7) > 1.0


def test_fast_pace_speeds_every_engine():
    assert pace_to_sapi_rate(1.3) > 0
    assert pace_to_espeak_wpm(1.3) > 150
    assert pace_to_piper_length_scale(1.3) < 1.0


def test_pace_is_clamped_to_schema_bounds():
    assert -10 <= pace_to_sapi_rate(0.0) <= 10
    assert -10 <= pace_to_sapi_rate(9.9) <= 10
    assert 80 <= pace_to_espeak_wpm(0.0) <= 300


def test_disabled_speech_selects_null_adapter():
    adapter = select_speech_adapter(SpeechConfig(enabled=False))
    assert isinstance(adapter, NullSpeech)
    adapter = select_speech_adapter(SpeechConfig(engine="none"))
    assert isinstance(adapter, NullSpeech)


def test_null_adapter_never_fails():
    metrics = NullSpeech().speak("Silence is also a performance.", VoiceDirection())
    assert metrics["spoken"] is False


def test_missing_piper_model_falls_back_silently():
    adapter = select_speech_adapter(
        SpeechConfig(engine="piper", piper_model_path="does/not/exist.onnx")
    )
    assert isinstance(adapter, NullSpeech)
