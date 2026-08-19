import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pedalboard")
pytest.importorskip("soundfile")

from azmo_mind.config import VoiceDspConfig
from azmo_mind.schemas import VoiceDirection
from azmo_mind.voice_dsp import apply_azmo_voice, dsp_available, heaviness, process_wav

# Keep tests fast/deterministic: pitch-only path, not the WORLD vocoder.
CFG = VoiceDspConfig(use_world=False)


def _tone(seconds: float = 1.0, sr: int = 24000) -> "np.ndarray":
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 130 * t) + 0.2 * np.sin(2 * np.pi * 260 * t)
    return sig.astype(np.float32)


def test_dsp_reports_available_in_this_env():
    assert dsp_available() is True


def test_declamatory_presets_are_heavier_than_calm():
    calm = heaviness(VoiceDirection(preset="calm_dark"), CFG)
    decree = heaviness(VoiceDirection(preset="imperial_decree"), CFG)
    assert decree > calm


def test_subharmonic_mix_raises_heaviness():
    low = heaviness(VoiceDirection(preset="calm_dark", subharmonic_mix=0.0), CFG)
    high = heaviness(VoiceDirection(preset="calm_dark", subharmonic_mix=0.25), CFG)
    assert high > low


def test_heaviness_is_bounded():
    hot = heaviness(VoiceDirection(preset="victory", subharmonic_mix=0.25),
                    VoiceDspConfig(intensity_bias=1.0, use_world=False))
    assert 0.0 <= hot <= 1.0


def test_output_is_finite_and_not_clipping():
    out = apply_azmo_voice(_tone(), 24000, VoiceDirection(), CFG)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert float(np.max(np.abs(out))) <= 0.98


def test_calm_and_heavy_render_differently():
    calm = apply_azmo_voice(_tone(), 24000, VoiceDirection(preset="calm_dark"), CFG)
    heavy = apply_azmo_voice(_tone(), 24000,
                             VoiceDirection(preset="victory", subharmonic_mix=0.25), CFG)
    n = min(len(calm), len(heavy))
    assert not np.allclose(calm[:n], heavy[:n], atol=1e-3)


def test_formant_warp_lowers_the_spectral_peak():
    """ratio<1 must move the vocal-tract resonance down in frequency."""
    from azmo_mind.voice_dsp import _warp_formants
    envelope = np.zeros((3, 200), dtype=np.float64)
    envelope[:, 100] = 1.0  # a formant peak at bin 100
    warped = _warp_formants(envelope, 0.7)
    assert int(np.argmax(warped[0])) < 100


def test_world_path_runs_and_stays_finite():
    pytest.importorskip("pyworld")
    from azmo_mind.voice_dsp import _deepen
    sr = 24000
    t = np.linspace(0, 1, sr, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 160 * t) + 0.25 * np.sin(2 * np.pi * 320 * t)).astype(np.float32)
    out = _deepen(tone, sr, pitch_ratio=0.6, formant_ratio=0.7, use_world=True)
    assert len(out) > 0 and np.all(np.isfinite(out))


def test_empty_input_is_handled():
    out = apply_azmo_voice(np.zeros(0, dtype=np.float32), 24000, VoiceDirection(), CFG)
    assert len(out) == 0


def test_process_wav_roundtrip(tmp_path):
    import soundfile as sf
    src = tmp_path / "in.wav"
    dst = tmp_path / "out.wav"
    sf.write(str(src), _tone(), 24000)
    ran = process_wav(str(src), str(dst), VoiceDirection(), CFG)
    assert ran is True
    assert dst.exists()
