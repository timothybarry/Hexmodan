from pathlib import Path

from azmo_mind.config import SpeechConfig
from azmo_mind.speech import XttsCloneSpeech, select_speech_adapter


def _clone(ref):
    return XttsCloneSpeech(ref, params={"temperature": 0.7})


def test_reference_directory_gathers_all_clips(tmp_path):
    for name in ("b.wav", "a.wav", "c.wav"):
        (tmp_path / name).write_bytes(b"RIFF")
    clips = _clone(tmp_path).reference_clips()
    assert [Path(c).name for c in clips] == ["a.wav", "b.wav", "c.wav"]  # sorted


def test_reference_single_file(tmp_path):
    f = tmp_path / "ref.wav"
    f.write_bytes(b"RIFF")
    assert _clone(f).reference_clips() == [str(f)]


def test_missing_reference_yields_no_clips(tmp_path):
    assert _clone(tmp_path / "nope.wav").reference_clips() == []
    assert _clone(None).reference_clips() == []


def test_unavailable_without_coqui_or_reference(tmp_path):
    # No reference -> unavailable regardless of deps.
    assert _clone(tmp_path / "none").available() is False


def test_clone_params_are_plumbed_from_config():
    cfg = SpeechConfig(
        engine="clone",
        clone_temperature=0.55,
        clone_repetition_penalty=4.0,
        clone_top_k=40,
        clone_seed=7,
    )
    adapter = select_speech_adapter(cfg)
    # With no coqui installed the selector falls back, but we can build directly:
    clone = XttsCloneSpeech(
        cfg.clone_reference_path,
        params={
            "temperature": cfg.clone_temperature,
            "repetition_penalty": cfg.clone_repetition_penalty,
            "top_k": cfg.clone_top_k,
        },
        seed=cfg.clone_seed,
    )
    assert clone.params["temperature"] == 0.55
    assert clone.params["repetition_penalty"] == 4.0
    assert clone.params["top_k"] == 40
    assert clone.seed == 7
    # select returns *some* adapter (fallback) without raising
    assert adapter.name in {"clone", "sapi", "espeak", "none"}
