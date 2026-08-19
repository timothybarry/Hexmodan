"""Build a clean voice-clone reference from source audio.

Pipeline (brief section 8, cloning path):
  1. Demucs isolates the vocal stem from each source file, stripping music/SFX.
  2. Each vocal stem is split into spoken segments.
  3. Every segment is scored for "dryness" (silent gaps well below the voice),
     harmonic content (speech is harmonic; music/percussion is not), and low
     spectral flatness. The cleanest segments are light-denoised and stitched
     into a single reference WAV that XTTS clones from.

Run on the dev machine (GPU recommended for Demucs) after installing the prep
extra:  pip install -e ".[prep]"

    python scripts/prepare_reference.py "Diablo 3- All Azmodan Voice Lines.mp3" \
        --out data/voices/azmo_reference.wav --seconds 25

Dependencies (demucs, librosa, noisereduce, soundfile) are only needed here, not
to run AZMO. Isolation is far better than raw game rips, but always give the
result a listen — and mind the likeness/copyright notes in the project brief if
AZMO will ever be shared.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

SR = 24000  # XTTS reference sample rate


def _demucs_vocals(source: Path, workdir: Path) -> Path:
    """Isolate the vocal stem with Demucs two-stem separation."""
    subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", str(workdir), str(source)],
        check=True,
    )
    matches = list(workdir.glob(f"*/{source.stem}/vocals.wav"))
    if not matches:
        raise FileNotFoundError(f"Demucs produced no vocal stem for {source}")
    return matches[0]


def _score_segments(vocals_path: Path):
    import librosa
    import numpy as np

    y, sr = librosa.load(str(vocals_path), sr=SR, mono=True)
    intervals = librosa.effects.split(y, top_db=32, frame_length=2048, hop_length=512)
    scored = []
    for start, end in intervals:
        seg = y[start:end]
        duration = (end - start) / sr
        if duration < 2.0 or duration > 12.0:
            continue
        rms = librosa.feature.rms(y=seg, frame_length=1024, hop_length=256)[0]
        peak = float(np.max(rms)) or 1e-9
        floor_db = 20 * np.log10((np.percentile(rms, 10) / peak) + 1e-9)
        harmonic, _ = librosa.effects.hpss(seg)
        harm_ratio = float(np.sum(harmonic**2) / (np.sum(seg**2) + 1e-9))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=seg)))
        score = (-floor_db) + 30 * harm_ratio - 40 * flatness
        scored.append((score, start, end))
    scored.sort(reverse=True)
    return y, sr, scored


def build_reference(sources: list[Path], out_path: Path, seconds: float) -> None:
    import noisereduce as nr
    import numpy as np
    import soundfile as sf

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pieces = []
    total = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for source in sources:
            print(f"[isolate] {source.name}")
            vocals = _demucs_vocals(source, workdir)
            audio, sr, scored = _score_segments(vocals)
            print(f"[score]   {len(scored)} candidate segments")
            for _, start, end in scored:
                seg = audio[start:end]
                seg = nr.reduce_noise(y=seg, sr=sr, stationary=False, prop_decrease=0.7)
                seg = seg / (np.max(np.abs(seg)) + 1e-9) * 0.95
                pieces.append(seg.astype("float32"))
                pieces.append(np.zeros(int(0.25 * sr), dtype="float32"))
                total += (end - start) / sr + 0.25
                if total >= seconds:
                    break
            if total >= seconds:
                break

    if not pieces:
        raise SystemExit("No usable segments found. Try a cleaner source.")
    reference = np.concatenate(pieces)
    sf.write(str(out_path), reference, SR)
    print(f"[done]    wrote {out_path} ({len(reference) / SR:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean XTTS clone reference.")
    parser.add_argument("sources", nargs="+", help="Source audio files (mp3/wav/flac).")
    parser.add_argument("--out", default="data/voices/azmo_reference.wav")
    parser.add_argument("--seconds", type=float, default=25.0)
    args = parser.parse_args()
    build_reference([Path(s) for s in args.sources], Path(args.out), args.seconds)


if __name__ == "__main__":
    main()
