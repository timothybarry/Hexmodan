"""azmo-voice: the demonic voice DSP chain (brief section 8).

Runs *after* text-to-speech, sculpting an existing voice (the cloned Azmodan
voice, a piper voice, even Windows SAPI) into AZMO's. It never generates speech.

The voice is **dynamic** — "blend by moment". Each utterance gets a *heaviness*
in 0..1 from ``intensity_bias`` plus the model's own ``VoiceDirection`` (its
``preset`` and ``subharmonic_mix``). Calm presets stay near the light end (deep
but intelligible "Commander"); declamatory presets (``imperial_decree``,
``restrained_rage``, ``victory``, ``contempt``) push toward the heavy end, where
layered detuned "Legion" voices and a sub-bass "growl" fade in. Every parameter
interpolates between its light and heavy ends by that heaviness.

The core move is independent **pitch + formant lowering** via the WORLD vocoder:
dropping the spectral envelope makes the vocal tract sound enormous ("impossible
mass") instead of merely sped-down. pyworld and pedalboard are optional: without
pyworld it degrades to pitch-only shifting; without pedalboard the chain is a
transparent pass-through so speech still plays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from azmo_mind.config import VoiceDspConfig
from azmo_mind.schemas import VoiceDirection

# How declamatory each voice preset is -> how much it pushes heaviness up.
_PRESET_WEIGHT = {
    "calm_dark": 0.00,
    "temptation": 0.05,
    "close_ominous": 0.10,
    "dark_amusement": 0.10,
    "solemn": 0.15,
    "contempt": 0.28,
    "imperial_decree": 0.45,
    "restrained_rage": 0.50,
    "victory": 0.50,
}

# Fixed layer/EQ constants (the interpolated knobs live in VoiceDspConfig).
_OCTAVE_GAIN_LIGHT_DB = -16.0
_OCTAVE_GAIN_HEAVY_DB = -10.0
_LEGION_MAX_GAIN_DB = -15.0
_GROWL_MAX_GAIN_DB = -9.0
_HIGHPASS_HZ = 70.0
_LOW_SHELF_HZ = 150.0
_LOW_SHELF_GAIN_DB = 4.0
_REVERB_ROOM_LIGHT = 0.14
_REVERB_ROOM_HEAVY = 0.30
_LIMITER_THRESHOLD_DB = -1.0


@dataclass
class GainAnchor:
    """Shared normalisation scale for a reply rendered in several passes.

    ``apply_azmo_voice`` peak-normalises three times: the incoming voice, the
    deepened primary layer, and the finished mix. Over a whole reply that is
    exactly right - one gain frame, consistent loudness, dynamics preserved
    between sentences.

    Run per chunk it becomes a defect. Each chunk would be normalised to the
    same 0.97 peak independently, so a murmured closing clause is lifted to the
    level of the sentence that was shouted. The result is loudness pumping at
    every chunk boundary: not a stutter, but audible, and squarely in the
    "uncrisp / wrong" category that costs a listening session to chase.

    An anchor is captured on the first chunk and replayed on every chunk after
    it, which puts the whole reply back into one gain frame. It is not identical
    to whole-reply DSP - the divisor is the first chunk's peak rather than the
    reply's - but that difference is a single constant applied uniformly, which
    is inaudible, while the pumping it removes is not. The limiter at -1 dBFS
    still backstops any chunk louder than the first.

    Streaming is the only reason this exists. A non-streamed reply passes
    ``None`` and behaves exactly as it did before 0.2.10.
    """

    input_peak: float | None = None
    primary_peak: float | None = None
    output_peak: float | None = None

    def scale(self, name: str, measured: float) -> float:
        """Return the divisor to use, remembering the first one seen."""
        existing = getattr(self, name)
        if existing is None:
            setattr(self, name, measured)
            return measured
        return existing


def dsp_available() -> bool:
    try:
        import numpy  # noqa: F401
        import pedalboard  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError:
        return False
    return True


def world_available() -> bool:
    try:
        import pyworld  # noqa: F401
    except ImportError:
        return False
    return True


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _fade(value: float, threshold: float) -> float:
    """0 below threshold, ramping to 1 at heaviness 1.0."""
    if value <= threshold or threshold >= 1.0:
        return 0.0
    return (value - threshold) / (1.0 - threshold)


def heaviness(direction: VoiceDirection, config: VoiceDspConfig) -> float:
    """Per-utterance 0..1 intensity: a constant ``intensity_bias`` register plus a
    scaled swing from the model's delivery. ``heaviness_variation`` controls how
    much presets/mixes are allowed to move it — low keeps a consistent voice."""
    preset_weight = _PRESET_WEIGHT.get(direction.preset, 0.1)
    mix_push = (direction.subharmonic_mix / 0.25) * 0.30
    swing = (preset_weight + mix_push) * getattr(config, "heaviness_variation", 1.0)
    return max(0.0, min(1.0, config.intensity_bias + swing))


def _warp_formants(spectrogram, ratio: float):
    """Shift the spectral envelope; ratio<1 lowers formants (bigger tract)."""
    import numpy as np

    bins = spectrogram.shape[1]
    source = np.clip(np.arange(bins) / ratio, 0, bins - 1)
    index = np.arange(bins)
    warped = np.empty_like(spectrogram)
    for frame in range(spectrogram.shape[0]):
        warped[frame] = np.interp(source, index, spectrogram[frame])
    return warped


def _deepen(mono, sample_rate: int, pitch_ratio: float, formant_ratio: float, use_world: bool):
    """Lower pitch and formants. WORLD when available; else pitch-only shift."""
    import numpy as np

    if use_world and world_available():
        import pyworld as pw

        y = np.ascontiguousarray(mono, dtype=np.float64)
        f0, t = pw.harvest(y, sample_rate)
        sp = pw.cheaptrick(y, f0, t, sample_rate)
        ap = pw.d4c(y, f0, t, sample_rate)
        out = pw.synthesize(f0 * pitch_ratio, _warp_formants(sp, formant_ratio), ap, sample_rate)
        return out.astype(np.float32)

    from pedalboard import PitchShift

    semitones = 12.0 * math.log2(max(0.3, pitch_ratio))
    return PitchShift(semitones=semitones)(mono.astype(np.float32), sample_rate)


def apply_azmo_voice(
    audio,
    sample_rate: int,
    direction: VoiceDirection,
    config: VoiceDspConfig,
    anchor: GainAnchor | None = None,
):
    """Return ``audio`` (mono float32 ndarray) transformed into AZMO's voice.

    Falls back to the input unchanged if pedalboard/numpy are unavailable.

    ``anchor`` shares one normalisation frame across several calls, for a reply
    rendered chunk by chunk while it streams. Pass ``None`` (the default) for a
    whole reply in one pass, which is the behaviour that predates 0.2.10.
    """
    try:
        import numpy as np
        from pedalboard import (
            Compressor,
            Distortion,
            Gain,
            HighpassFilter,
            HighShelfFilter,
            Limiter,
            LowpassFilter,
            LowShelfFilter,
            Pedalboard,
            PeakFilter,
            PitchShift,
            Reverb,
        )
    except ImportError:
        return audio

    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    if mono.size == 0:
        return mono
    input_peak = float(np.max(np.abs(mono))) + 1e-9
    if anchor is not None:
        input_peak = anchor.scale("input_peak", input_peak)
    mono = mono / input_peak

    h = heaviness(direction, config)

    # 1) Deep, huge primary (pitch + formants down).
    pitch_ratio = _lerp(config.pitch_ratio_light, config.pitch_ratio_heavy, h)
    formant_ratio = _lerp(config.formant_ratio_light, config.formant_ratio_heavy, h)
    primary = _deepen(mono, sample_rate, pitch_ratio, formant_ratio, config.use_world)
    primary_peak = float(np.max(np.abs(primary))) + 1e-9
    if anchor is not None:
        primary_peak = anchor.scale("primary_peak", primary_peak)
    primary = primary / primary_peak

    layers = [primary]

    # 2) Octave-down mass layer, tightly low-passed so it adds weight, not mud.
    octave_gain = _lerp(_OCTAVE_GAIN_LIGHT_DB, _OCTAVE_GAIN_HEAVY_DB, h)
    layers.append(
        Pedalboard([PitchShift(semitones=-12), LowpassFilter(150), Gain(gain_db=octave_gain)])(
            primary, sample_rate
        )
    )

    # 3) Legion: a couple of quiet detuned voices, only on the biggest
    #    declarations (it is inherently washy, so kept rare and subtle).
    legion = _fade(h, config.legion_threshold)
    if legion > 0:
        gain = _LEGION_MAX_GAIN_DB - (1.0 - legion) * 8.0
        for semis in (0.2, -5.0):
            layers.append(
                Pedalboard([PitchShift(semitones=semis), Gain(gain_db=gain)])(primary, sample_rate)
            )

    # 5) Sub growl, fading in at the heaviest moments (now grittier).
    growl = _fade(h, config.growl_threshold)
    if growl > 0:
        gain = _GROWL_MAX_GAIN_DB - (1.0 - growl) * 12.0
        layers.append(
            Pedalboard([
                PitchShift(semitones=-19),
                LowpassFilter(130),
                Distortion(drive_db=16),
                Gain(gain_db=gain),
            ])(primary, sample_rate)
        )

    # 6) Guttural grit: a saturated low-mid throat-rasp band. This is the
    #    "guttural" character — gravel in the 140-1100 Hz band, present even on
    #    calm speech and stronger as heaviness rises.
    grit = _fade(h, config.grit_threshold)
    if grit > 0 and config.grit_gain_db > -55:
        grit_gain = config.grit_gain_db - (1.0 - grit) * 6.0
        layers.append(
            Pedalboard([
                HighpassFilter(140),
                LowpassFilter(1100),
                Distortion(drive_db=config.grit_drive_db),
                Compressor(threshold_db=-22, ratio=4.0, attack_ms=5, release_ms=120),
                Gain(gain_db=grit_gain),
            ])(primary, sample_rate)
        )

    # 7) Air/exciter: high-passed, lightly saturated harmonics for crisp highs
    #    and consonant clarity — the "crispier highs".
    if config.air_gain_db > -55:
        layers.append(
            Pedalboard([
                HighpassFilter(config.air_hz),
                Distortion(drive_db=5),
                Gain(gain_db=config.air_gain_db),
            ])(primary, sample_rate)
        )

    length = max(layer.reshape(-1).shape[-1] for layer in layers)

    def pad(layer):
        flat = np.asarray(layer, dtype=np.float32).reshape(-1)
        return np.pad(flat, (0, length - flat.shape[-1]))

    mixed = sum(pad(layer) for layer in layers)

    # 6) Tone shaping, saturation, chamber, limiter.
    drive = _lerp(config.drive_light_db, config.drive_heavy_db, h)
    # Reverb kept near-dry for intelligibility; the model's reverb_mix nudges it.
    wet = max(0.0, min(0.15, _lerp(config.reverb_wet_light, config.reverb_wet_heavy, h)
                       + direction.reverb_mix * 0.2))
    room = _lerp(_REVERB_ROOM_LIGHT, _REVERB_ROOM_HEAVY, h)
    out = Pedalboard([
        HighpassFilter(_HIGHPASS_HZ),
        LowShelfFilter(_LOW_SHELF_HZ, gain_db=_LOW_SHELF_GAIN_DB),
        PeakFilter(cutoff_frequency_hz=config.mud_cut_hz, gain_db=config.mud_cut_gain_db,
                   q=config.mud_cut_q),
        PeakFilter(cutoff_frequency_hz=config.clarity_hz, gain_db=config.clarity_gain_db,
                   q=config.clarity_q),
        HighShelfFilter(config.presence_hz, gain_db=config.presence_gain_db),
        Distortion(drive_db=drive),
        Compressor(threshold_db=-19, ratio=3.0, attack_ms=8, release_ms=140),
        Reverb(room_size=room, damping=0.5, wet_level=wet, dry_level=1.0 - wet, width=0.6),
        Limiter(threshold_db=_LIMITER_THRESHOLD_DB, release_ms=80),
    ])(mixed, sample_rate)

    peak = float(np.max(np.abs(out))) or 1.0
    if anchor is not None:
        peak = anchor.scale("output_peak", peak)
    return (out / peak * 0.97).astype(np.float32)


def process_wav(
    in_path: str,
    out_path: str,
    direction: VoiceDirection,
    config: VoiceDspConfig,
    anchor: GainAnchor | None = None,
) -> bool:
    """Read a WAV, apply the AZMO voice, write it back. Returns True if the DSP
    ran, False if it degraded to a copy (dependencies missing).

    ``anchor`` keeps a streamed reply in one gain frame - see ``GainAnchor``."""
    try:
        import soundfile as sf
    except ImportError:
        return False
    audio, sample_rate = sf.read(in_path)
    processed = apply_azmo_voice(audio, sample_rate, direction, config, anchor=anchor)
    sf.write(out_path, processed, sample_rate)
    return dsp_available()
