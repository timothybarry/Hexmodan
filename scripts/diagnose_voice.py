"""Pinpoint where the voice stack dies.

XTTS has been aborting the whole process (Windows exit code -1073740791 /
0xC0000409). That is a *native* abort: no Python traceback, no exception to
catch, nothing in the log. The only way to locate it is to print a marker
before every step and see which marker is last.

Run it directly, and paste the whole output:

    .\\.venv312\\Scripts\\python.exe scripts\\diagnose_voice.py

The last "[step N]" line printed is the operation that killed the process.
Add --cpu to force CPU and confirm whether the GPU path is the problem.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

STEP = 0


def step(message: str) -> None:
    """Print a marker and flush immediately.

    Flushing matters more than it looks: a native abort does not unwind, so
    anything still sitting in a buffer is lost and the crash appears to happen
    one step earlier than it did.
    """
    global STEP
    STEP += 1
    print(f"\n[step {STEP}] {message}", flush=True)


def detail(message: str) -> None:
    print(f"          {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Force XTTS onto the CPU.")
    parser.add_argument("--config", default="config/azmo.yaml")
    parser.add_argument(
        "--text",
        default="The Sin War was merely a rehearsal.",
        help="Line to synthesize.",
    )
    parser.add_argument(
        "--with-whisper",
        action="store_true",
        help="Load faster-whisper FIRST, reproducing the order azmo listen uses.",
    )
    args = parser.parse_args()

    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    print("=" * 68, flush=True)
    print(" AZMO voice diagnostic - the last [step] printed is the crash site", flush=True)
    print("=" * 68, flush=True)

    step("Python and platform")
    import platform

    detail(f"{platform.python_version()} | {platform.platform()}")
    detail(f"executable: {sys.executable}")

    step("Import torch")
    import torch

    detail(f"torch {torch.__version__}")
    detail(f"built with CUDA: {torch.version.cuda}")
    # Deliberately NOT touching torch.backends.cudnn yet - that forces cuDNN to
    # initialise, and we want to know whether the earlier steps survive first.

    step("torch.cuda.is_available()")
    available = torch.cuda.is_available()
    detail(f"CUDA available: {available}")
    if available:
        detail(f"device: {torch.cuda.get_device_name(0)}")

    step("Query cuDNN version (this alone has been enough to abort)")
    try:
        detail(f"cudnn version: {torch.backends.cudnn.version()}")
        detail(f"cudnn enabled: {torch.backends.cudnn.enabled}")
    except Exception as exc:  # noqa: BLE001
        detail(f"cudnn query raised: {type(exc).__name__}: {exc}")

    step("Installed package versions")
    for name in ("TTS", "transformers", "ctranslate2", "faster_whisper", "numpy"):
        try:
            module = __import__(name)
            detail(f"{name}: {getattr(module, '__version__', 'unknown')}")
        except Exception as exc:  # noqa: BLE001
            detail(f"{name}: NOT IMPORTABLE ({type(exc).__name__})")

    step("Look for duplicate cuDNN DLLs (the usual culprit)")
    seen: list[str] = []
    for entry in sys.path:
        candidate = Path(entry)
        if not candidate.is_dir():
            continue
        for match in list(candidate.glob("nvidia/cudnn/bin/cudnn*.dll"))[:6]:
            seen.append(str(match))
    for found in seen or ["(none found on sys.path)"]:
        detail(found)

    if args.with_whisper:
        step("Load faster-whisper FIRST (same order as azmo listen)")
        from faster_whisper import WhisperModel

        model = WhisperModel("small.en", device="cpu", compute_type="int8")
        import numpy as np

        segments, _info = model.transcribe(np.zeros(8000, dtype=np.float32))
        detail(f"whisper ok, {len(list(segments))} segment(s)")

    step("Load AZMO config")
    from azmo_mind.config import load_config

    cfg = load_config(args.config)
    device = "cpu" if args.cpu else cfg.speech.clone_device
    detail(f"clone_device: {device}")
    detail(f"reference: {cfg.speech.clone_reference_path}")

    step("Build the clone adapter")
    from azmo_mind.speech import XttsCloneSpeech

    clone = XttsCloneSpeech(
        cfg.speech.clone_reference_path,
        model=cfg.speech.clone_model,
        language=cfg.speech.clone_language,
        dsp=cfg.speech.dsp,
        speed=cfg.speech.speed,
        params={
            "temperature": cfg.speech.clone_temperature,
            "repetition_penalty": cfg.speech.clone_repetition_penalty,
            "top_k": cfg.speech.clone_top_k,
            "top_p": cfg.speech.clone_top_p,
            "length_penalty": cfg.speech.clone_length_penalty,
            "split_text": cfg.speech.clone_split_text,
        },
        seed=cfg.speech.clone_seed,
        latent_cache=cfg.speech.clone_latent_cache,
        device=device,
    )
    detail(f"reference clips: {len(clone.reference_clips())}")
    if not clone.reference_clips():
        detail("NO REFERENCE CLIPS - run SETUP_VOICE.bat first.")
        return 1

    step("Load the XTTS model (~2 GB)")
    clone._engine()
    detail("model loaded")

    step("Compute or load speaker latents")
    latents = clone._conditioning()
    detail("latents ready" if latents is not None else "latents unavailable (will use high-level API)")

    step("Synthesize one line")
    from azmo_mind.schemas import VoiceDirection

    out = REPO_ROOT / "voice_diagnostic.wav"
    clone._synthesize(args.text, out, 1.0)
    detail(f"wrote {out} ({out.stat().st_size} bytes)")

    step("Apply the DSP chain")
    from azmo_mind import voice_dsp

    ran = voice_dsp.process_wav(str(out), str(out), VoiceDirection(), cfg.speech.dsp)
    detail(f"dsp ran: {ran}")

    step("Play it back")
    from azmo_mind.speech import _play_wav

    _play_wav(out)
    detail("playback finished")

    print("\n" + "=" * 68, flush=True)
    print(" ALL STEPS PASSED - the voice stack is healthy.", flush=True)
    print(f" Listen to {out} to check it sounds right.", flush=True)
    print("=" * 68, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
