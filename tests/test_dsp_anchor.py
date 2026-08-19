"""One gain frame across a reply rendered in pieces.

``apply_azmo_voice`` peak-normalises three times per call. Over a whole reply
that is correct: one frame, consistent loudness, dynamics preserved between
sentences. Run once per chunk - which is what streaming does - it silently
becomes a defect, because each chunk is normalised to the same ceiling on its
own and a murmured clause is lifted to the level of a shouted one.

The result is loudness pumping at every chunk boundary. It is not a stutter and
it will not raise; it just makes him sound wrong, which costs a listening
session to chase. ``GainAnchor`` puts the chunks back in one frame.

The pair of tests below is the point of the module: the anchored render must
preserve the level difference between chunks, and the unanchored one must
destroy it. They are written as a comparison rather than against a fixed
number, because the chain's own saturation and compression already squash most
of the range - a 20 dB difference at the input leaves roughly 3 dB at the
output. That is still plainly audible as pumping when it flips chunk to chunk,
but it means an absolute threshold here would be a magic number describing the
compressor rather than the anchor.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pedalboard")
pytest.importorskip("soundfile")

from azmo_mind.config import VoiceDspConfig
from azmo_mind.schemas import VoiceDirection
from azmo_mind.voice_dsp import GainAnchor, apply_azmo_voice

CFG = VoiceDspConfig(use_world=False)
DIRECTION = VoiceDirection()
SR = 24000


def _tone(amplitude: float, seconds: float = 0.6) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    sig = amplitude * (np.sin(2 * np.pi * 130 * t) + 0.4 * np.sin(2 * np.pi * 260 * t))
    return sig.astype(np.float32)


def _rms(signal) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


# A loud opening clause and a much quieter closing one - the dynamic range the
# anchor exists to preserve.
LOUD = _tone(0.90)
QUIET = _tone(0.09)


def _level_ratio(anchor: GainAnchor | None) -> float:
    """RMS of the quiet chunk over the loud one, rendered as two passes."""
    loud = apply_azmo_voice(LOUD, SR, DIRECTION, CFG, anchor=anchor)
    quiet = apply_azmo_voice(QUIET, SR, DIRECTION, CFG, anchor=anchor)
    return _rms(quiet) / _rms(loud)


def test_without_an_anchor_the_quiet_chunk_is_lifted_to_match_the_loud_one():
    """This is the defect. It must stay demonstrable."""
    assert _level_ratio(None) == pytest.approx(1.0, abs=0.05)


def test_with_an_anchor_the_quiet_chunk_stays_quiet():
    assert _level_ratio(GainAnchor()) < 0.85


def test_the_anchor_preserves_a_difference_the_unanchored_path_destroys():
    """The comparison is the real assertion: one frame versus two."""
    assert _level_ratio(GainAnchor()) < _level_ratio(None) - 0.15


def test_the_first_anchored_chunk_is_identical_to_an_unanchored_render():
    """The whole-reply path must be untouched by 0.2.10.

    An anchor only remembers; it never changes the first render. So a reply that
    is not streamed sounds exactly as it did before this feature existed.
    """
    plain = apply_azmo_voice(LOUD, SR, DIRECTION, CFG)
    anchored = apply_azmo_voice(LOUD, SR, DIRECTION, CFG, anchor=GainAnchor())
    assert np.array_equal(plain, anchored)


def test_an_anchor_records_the_first_scale_and_then_reuses_it():
    anchor = GainAnchor()
    assert anchor.scale("input_peak", 0.8) == 0.8
    assert anchor.scale("input_peak", 0.1) == 0.8
    assert anchor.input_peak == 0.8


def test_an_anchored_chunk_still_cannot_clip():
    """A chunk louder than the first is held by the limiter, not by rescaling."""
    anchor = GainAnchor()
    apply_azmo_voice(_tone(0.1), SR, DIRECTION, CFG, anchor=anchor)
    hot = apply_azmo_voice(_tone(1.0), SR, DIRECTION, CFG, anchor=anchor)
    assert np.all(np.isfinite(hot))
    assert float(np.max(np.abs(hot))) <= 1.0


def test_an_anchor_survives_an_empty_chunk():
    anchor = GainAnchor()
    apply_azmo_voice(LOUD, SR, DIRECTION, CFG, anchor=anchor)
    out = apply_azmo_voice(np.zeros(0, dtype=np.float32), SR, DIRECTION, CFG, anchor=anchor)
    assert len(out) == 0
    assert anchor.input_peak is not None


def test_process_wav_threads_the_anchor_through(tmp_path):
    import soundfile as sf

    from azmo_mind.voice_dsp import process_wav

    def render(anchor: GainAnchor | None) -> float:
        levels = []
        for index, source in enumerate((LOUD, QUIET)):
            src = tmp_path / f"in{index}.wav"
            dst = tmp_path / f"out{index}-{anchor is not None}.wav"
            sf.write(str(src), source, SR)
            assert process_wav(str(src), str(dst), DIRECTION, CFG, anchor=anchor) is True
            audio, _ = sf.read(str(dst))
            levels.append(_rms(audio))
        return levels[1] / levels[0]

    assert render(GainAnchor()) < render(None) - 0.15
