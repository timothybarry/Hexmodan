"""Making the presence pool sound like a pool rather than one noise.

Diagnosed by ear on 2026-08-05: twelve clips, twelve genuinely distinct files
(all md5s differed), that nonetheless sounded like the same two sounds repeating.

The picker was not at fault. Every clip had been rendered from a single sampling
roll - ``speech.clone_seed`` fixed, ``clone_temperature`` at 0.26 - because those
settings exist so a good *spoken* take stays good. Presence wants the opposite
trade: the pool's entire purpose is that no single breath becomes a recognisable
tic, and one seed across every clip works directly against it.

The second complaint was a rising chirp at the end of each clip. XTTS voices a
trailing vowel, a voiced vowel carries pitch, and the 7 kHz presence shelf then
lifts it - a boost tuned for consonants in speech, applied to a sound that is
mostly breath noise.
"""

from __future__ import annotations

import pytest

from azmo_mind.presence import render_seed, shape_clip

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

RATE = 24000


def _clip(path, seconds: float = 1.0, level: float = 0.8):
    """A steady tone - any fade shows up cleanly against a flat envelope."""
    t = np.linspace(0, seconds, int(RATE * seconds), endpoint=False)
    sf.write(str(path), (level * np.sin(2 * np.pi * 180 * t)).astype("float32"), RATE)
    return path


# ---------------------------------------------------------------------------
# Seed variety
# ---------------------------------------------------------------------------

def test_each_clip_gets_its_own_seed():
    """The fix for the actual complaint: one roll became twelve personalities."""
    seeds = [render_seed(20260726, i, 997) for i in range(1, 13)]
    assert len(set(seeds)) == 12


def test_a_rebuild_reproduces_the_same_pool():
    """Varied is not the same as random. A good pool must survive a rebuild."""
    first = [render_seed(20260726, i, 997) for i in range(1, 7)]
    second = [render_seed(20260726, i, 997) for i in range(1, 7)]
    assert first == second


def test_stride_zero_keeps_every_clip_on_the_speech_seed():
    """The escape hatch back to the old behaviour."""
    seeds = [render_seed(20260726, i, 0) for i in range(1, 7)]
    assert set(seeds) == {20260726}


def test_a_zero_base_seed_stays_zero():
    """clone_seed 0 means 'fresh each time' - deriving from it would be nonsense."""
    assert render_seed(0, 5, 997) == 0


# ---------------------------------------------------------------------------
# Envelope shaping
# ---------------------------------------------------------------------------

def test_the_tail_is_faded_to_silence(tmp_path):
    """Kills the rising chirp on a voiced trailing vowel."""
    path = _clip(tmp_path / "c.wav")
    assert shape_clip(path, fade_in_ms=20, fade_out_ms=200) is True
    audio, _ = sf.read(str(path), dtype="float32")
    assert abs(float(audio[-1])) < 0.01


def test_the_head_is_faded_in(tmp_path):
    """Softens the low thump the octave/sub layers make of a sharp onset."""
    path = _clip(tmp_path / "c.wav")
    shape_clip(path, fade_in_ms=20, fade_out_ms=200)
    audio, _ = sf.read(str(path), dtype="float32")
    assert abs(float(audio[0])) < 0.01


def test_the_middle_of_the_clip_is_untouched(tmp_path):
    """Shaping the envelope must not change the sound that is actually his."""
    path = _clip(tmp_path / "c.wav")
    before, _ = sf.read(str(path), dtype="float32")
    mid = len(before) // 2
    reference = float(np.max(np.abs(before[mid - 400:mid + 400])))
    shape_clip(path, fade_in_ms=20, fade_out_ms=200)
    after, _ = sf.read(str(path), dtype="float32")
    assert float(np.max(np.abs(after[mid - 400:mid + 400]))) == pytest.approx(
        reference, rel=0.02
    )


def test_the_fade_is_asymmetric(tmp_path):
    """A breath begins fairly abruptly and dies away slowly."""
    path = _clip(tmp_path / "c.wav")
    shape_clip(path, fade_in_ms=20, fade_out_ms=220)
    audio, _ = sf.read(str(path), dtype="float32")
    head = int(RATE * 0.020)
    tail = int(RATE * 0.220)
    assert tail > head
    # The tail region is measurably quieter than an equally long head region.
    assert float(np.mean(np.abs(audio[-tail:]))) < float(np.mean(np.abs(audio[:tail])))


def test_zero_fades_leave_the_clip_alone(tmp_path):
    path = _clip(tmp_path / "c.wav")
    before, _ = sf.read(str(path), dtype="float32")
    shape_clip(path, fade_in_ms=0, fade_out_ms=0)
    after, _ = sf.read(str(path), dtype="float32")
    assert np.allclose(before, after, atol=1e-6)


def test_a_fade_longer_than_the_clip_cannot_erase_it(tmp_path):
    """Clamped to half the clip each side, so something always survives."""
    path = _clip(tmp_path / "c.wav", seconds=0.05)
    assert shape_clip(path, fade_in_ms=5000, fade_out_ms=5000) is True
    audio, _ = sf.read(str(path), dtype="float32")
    assert float(np.max(np.abs(audio))) > 0.05


def test_a_missing_file_degrades_instead_of_raising(tmp_path):
    """Presence must never break a build, let alone a turn."""
    assert shape_clip(tmp_path / "nope.wav") is False
