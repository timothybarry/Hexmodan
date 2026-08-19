"""Local text-to-speech output adapters (the seed of the future azmo-speech service).

Design rules, inherited from the project brief:

- Fully local. No cloud voices during normal operation.
- Half-duplex for now: ``speak`` blocks until playback ends, so the caller can
  safely resume listening afterwards (brief section 7).
- The model never chooses an engine or a device. It emits a ``VoiceDirection``
  (preset, pace, pauses, mix levels); this module maps that intent onto
  whatever engine is actually available:

  1. ``piper``      — neural TTS, best quality, requires a downloaded .onnx voice
  2. ``sapi``       — Windows built-in System.Speech (no install, dev machine)
  3. ``espeak-ng``  — formant synthesis, robotic but dependable (Linux/Jetson)
  4. ``none``       — silent fallback; AZMO remains text-only

The demonic DSP chain (subharmonics, entity double, chamber reverb) belongs to
the future azmo-voice service and is deliberately NOT faked here. The
``subharmonic_mix`` / ``reverb_mix`` fields are carried through untouched so the
DSP stage can consume them later.
"""

from __future__ import annotations

import importlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from azmo_mind import voice_dsp
from azmo_mind.config import SpeechConfig, VoiceDspConfig
from azmo_mind.schemas import VoiceDirection


class SpeechError(RuntimeError):
    """A recoverable speech-output failure. Never fatal to a conversation."""


# The classes coqui pickles into the XTTS checkpoint. Imported by name because
# the set has shifted between coqui releases and a missing one must not stop
# the others from being allowlisted.
_XTTS_PICKLED_CLASSES = (
    ("TTS.tts.configs.xtts_config", "XttsConfig"),
    ("TTS.tts.models.xtts", "XttsAudioConfig"),
    ("TTS.tts.models.xtts", "XttsArgs"),
    ("TTS.config.shared_configs", "BaseDatasetConfig"),
)

_xtts_globals_allowed = False


def allow_xtts_globals() -> bool:
    """Let torch 2.6+ load coqui's XTTS checkpoint. Returns True if anything was allowed.

    torch 2.6 flipped the default of ``torch.load``'s ``weights_only`` from
    False to True. That is the right default — unpickling arbitrary objects
    from a downloaded file is remote code execution — but coqui's XTTS
    checkpoint is not a bare state dict. It contains pickled *config objects*,
    so the stricter loader refuses it:

        WeightsUnpickler error: Unsupported global: GLOBAL
        TTS.tts.configs.xtts_config.XttsConfig was not an allowed global

    Nothing is wrong with the file; torch simply will not instantiate classes
    it has not been told about. So we tell it about exactly these four, which
    is what the error message itself recommends.

    Deliberately narrow: the alternative fix circulating for this is to force
    ``weights_only=False`` globally, which turns the protection off for *every*
    ``torch.load`` in the process. Allowlisting keeps the safer default
    everywhere else and only vouches for classes that ship with the TTS package
    we already trust enough to import.

    Safe to call repeatedly, and a no-op on torch < 2.6 (no such API, and the
    old permissive default already applies).
    """
    global _xtts_globals_allowed
    if _xtts_globals_allowed:
        return True
    try:
        import torch
    except ImportError:
        return False
    register = getattr(getattr(torch, "serialization", None), "add_safe_globals", None)
    if register is None:
        _xtts_globals_allowed = True   # torch < 2.6 needs nothing
        return False

    allowed = []
    for module_path, name in _XTTS_PICKLED_CLASSES:
        try:
            module = importlib.import_module(module_path)
        except Exception:  # noqa: BLE001, S112 - a moved class must not block the rest
            continue
        obj = getattr(module, name, None)
        if obj is not None:
            allowed.append(obj)
    if not allowed:
        return False
    try:
        register(allowed)
    except Exception:  # noqa: BLE001 - already registered, or an API change
        return False
    _xtts_globals_allowed = True
    return True


def torch_load_trusted(path, **kwargs):
    """``torch.load`` for files this application wrote itself.

    The speaker-latent cache is produced by ``torch.save`` on this machine, so
    the weights-only protection is guarding against ourselves. Falls back to a
    plain call on torch versions predating the argument.
    """
    import torch

    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


# ---------------------------------------------------------------------------
# VoiceDirection mapping helpers (pure functions: easy to unit test)
# ---------------------------------------------------------------------------

def pace_to_sapi_rate(pace: float) -> int:
    """Map VoiceDirection.pace (0.6..1.35, 1.0 = neutral) to SAPI rate (-10..10).

    SAPI rate is roughly logarithmic; a linear map around the neutral point is
    close enough for performance direction.
    """
    pace = max(0.6, min(1.35, pace))
    if pace >= 1.0:
        rate = (pace - 1.0) / 0.35 * 10
    else:
        rate = (pace - 1.0) / 0.40 * 10
    return max(-10, min(10, round(rate)))


def pace_to_espeak_wpm(pace: float, base_wpm: int = 150) -> int:
    """Map pace onto espeak-ng words-per-minute. AZMO speaks deliberately."""
    pace = max(0.6, min(1.35, pace))
    return max(80, min(300, round(base_wpm * pace)))


def pace_to_piper_length_scale(pace: float) -> float:
    """Piper's length_scale stretches phoneme durations; inverse of pace."""
    pace = max(0.6, min(1.35, pace))
    return round(1.0 / pace, 3)


def effective_pace(pace: float, speed: float) -> float:
    """Anchor delivery near a natural, news-anchor clip.

    The model's ``pace`` swings from 0.6 to 1.35, which makes AZMO drag or race.
    We keep only ~30% of that swing around 1.0 so the tempo stays consistent and
    human, then apply the global ``speed`` (config) and clamp.
    """
    damped = 1.0 + (pace - 1.0) * 0.3
    return max(0.6, min(1.35, damped * speed))


# XTTS v2 generates in a fixed window; English is capped at 250 characters.
# Beyond that, with text splitting disabled, the generation loop runs past its
# positional limit instead of stopping - which produces garbage at best and a
# native process abort at worst.
XTTS_CHARACTER_LIMIT = 250


def _pack(parts: list[str], limit: int) -> list[str]:
    """Greedily join parts into chunks no longer than ``limit``."""
    chunks: list[str] = []
    current = ""
    for part in parts:
        if not current:
            current = part
        elif len(current) + 1 + len(part) <= limit:
            current = f"{current} {part}"
        else:
            chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks


def split_for_xtts(text: str, limit: int = 220) -> list[str]:
    """Break ``text`` into chunks XTTS can synthesize in one pass each.

    Preference order: whole sentences, then clauses at commas and semicolons,
    then a hard wrap at word boundaries. AZMO's replies run to ~100 words, so
    they routinely exceed the model's 250-character window.

    We do this rather than letting XTTS split internally because we re-seed
    before every chunk, so each one is the same sampling roll and the voice
    stays in character across the whole reply - which is what made the model's
    own splitting sound inconsistent ("some sentences demonic, others generic").
    """
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    for sentence in sentences:
        if len(sentence) <= limit:
            chunks.append(sentence)
            continue
        # One sentence longer than the window: fall back to clause boundaries.
        clauses = re.split(r"(?<=[,;:])\s+", sentence)
        for clause in clauses:
            if len(clause) <= limit:
                chunks.append(clause)
                continue
            # Still too long: hard wrap on words, never mid-word.
            words = clause.split()
            buffer = ""
            for word in words:
                if buffer and len(buffer) + 1 + len(word) > limit:
                    chunks.append(buffer)
                    buffer = word
                else:
                    buffer = f"{buffer} {word}".strip()
            if buffer:
                chunks.append(buffer)

    # Recombine adjacent short sentences so we make as few passes as possible;
    # every extra pass is another seam in the delivery.
    return _pack(chunks, limit)


def _temp_wav() -> Path:
    """A unique temp .wav path that does NOT exist yet.

    We must not create the file ourselves. mkstemp/NamedTemporaryFile leave a
    real 0-byte file that Windows (Defender/indexer scanning a fresh temp file)
    can briefly lock, which makes SAPI's SetOutputToWaveFile fail with "being
    used by another process". Handing the synthesizer a not-yet-existing path
    lets it create and own the file cleanly.
    """
    return Path(tempfile.gettempdir()) / f"azmo_{uuid.uuid4().hex}.wav"


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class SpeechAdapter(ABC):
    name: str = "abstract"

    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def speak(self, text: str, voice: VoiceDirection) -> dict[str, Any]:
        """Blocking synthesis + playback. Returns small metrics dict."""
        raise NotImplementedError

    def warm(self) -> None:
        """Load anything expensive now rather than mid-conversation.

        Default: nothing to do. Overridden by the clone engine, whose ~2 GB
        model load would otherwise land immediately after the first LLM turn -
        two heavy GPU loads back to back, which is the exact pattern we are
        trying to avoid.
        """
        return None


class NullSpeech(SpeechAdapter):
    """Silent adapter. Used when speech is disabled or nothing is installed."""

    name = "none"

    def available(self) -> bool:
        return True

    def speak(self, text: str, voice: VoiceDirection) -> dict[str, Any]:
        return {"engine": self.name, "spoken": False}


class SapiSpeech(SpeechAdapter):
    """Windows built-in TTS through PowerShell + System.Speech.

    Zero-install voice output on the current dev machine. Text goes through a
    temp file so no user content is ever interpolated into a shell command.
    """

    name = "sapi"

    def __init__(self, voice_hint: str = "David", volume: int = 100,
                 dsp: VoiceDspConfig | None = None, speed: float = 1.0):
        self.voice_hint = voice_hint
        self.volume = max(0, min(100, volume))
        self.dsp = dsp
        self.speed = speed

    def available(self) -> bool:
        return sys.platform == "win32" and shutil.which("powershell.exe") is not None

    def _select_voice_ps(self) -> str:
        return (
            "$hint=$env:AZMO_VOICE_HINT;"
            "if($hint){$v=$s.GetInstalledVoices()|Where-Object "
            "{$_.VoiceInfo.Name -like ('*'+$hint+'*')}|Select-Object -First 1;"
            "if($v){$s.SelectVoice($v.VoiceInfo.Name)}};"
        )

    def _run_ps(self, script: str, text_path: str, wav_path: Path | None) -> None:
        env = {**_safe_env(), "AZMO_SPEECH_FILE": text_path, "AZMO_VOICE_HINT": self.voice_hint}
        if wav_path is not None:
            env["AZMO_WAV"] = str(wav_path)
        try:
            subprocess.run(
                ["powershell.exe", "-NoLogo", "-NoProfile", "-Command", script],
                check=True, capture_output=True, timeout=120, env=env,
            )
        except subprocess.CalledProcessError as exc:
            raise SpeechError(
                f"Windows SAPI speech failed: {exc.stderr.decode(errors='replace')[:300]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SpeechError("Windows SAPI speech timed out after 120 s.") from exc

    def speak(self, text: str, voice: VoiceDirection) -> dict[str, Any]:
        rate = pace_to_sapi_rate(effective_pace(voice.pace, self.speed))
        # Only take the render-to-WAV detour when the DSP can actually run;
        # otherwise speak straight to the device (no temp file, nothing to lock).
        use_dsp = self.dsp is not None and self.dsp.enabled and voice_dsp.dsp_available()

        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", encoding="utf-8-sig", delete=False
        ) as handle:
            handle.write(text)
            text_path = handle.name
        wav_path = _temp_wav() if use_dsp else None
        started = time.perf_counter()
        try:
            head = (
                "$ErrorActionPreference='Stop';"
                "Add-Type -AssemblyName System.Speech;"
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                f"$s.Rate={rate};$s.Volume={self.volume};"
            ) + self._select_voice_ps()
            read_speak = (
                "$t=Get-Content -Raw -Encoding UTF8 $env:AZMO_SPEECH_FILE;"
                "$s.Speak($t);$s.Dispose()"
            )
            if use_dsp:
                self._run_ps(head + "$s.SetOutputToWaveFile($env:AZMO_WAV);" + read_speak,
                             text_path, wav_path)
                dsp_ran = _apply_dsp(wav_path, voice, self.dsp)
                if voice.pause_before_ms > 0:
                    time.sleep(voice.pause_before_ms / 1000)
                _play_wav(wav_path)
            else:
                if voice.pause_before_ms > 0:
                    time.sleep(voice.pause_before_ms / 1000)
                self._run_ps(head + read_speak, text_path, None)
                dsp_ran = False
        finally:
            Path(text_path).unlink(missing_ok=True)
            if wav_path is not None:
                wav_path.unlink(missing_ok=True)

        return {
            "engine": self.name,
            "spoken": True,
            "rate": rate,
            "dsp": dsp_ran,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }


class EspeakNgSpeech(SpeechAdapter):
    """espeak-ng formant synthesis. Robotic, but runs anywhere, including Jetson."""

    name = "espeak"

    def __init__(self, base_wpm: int = 150, voice: str = "en-us", pitch: int = 28,
                 dsp: VoiceDspConfig | None = None, speed: float = 1.0):
        # pitch 0..99 (50 = default). Low pitch suits AZMO until real DSP exists.
        self.base_wpm = base_wpm
        self.voice = voice
        self.pitch = max(0, min(99, pitch))
        self.dsp = dsp
        self.speed = speed

    def _binary(self) -> str | None:
        return shutil.which("espeak-ng") or shutil.which("espeak")

    def available(self) -> bool:
        return self._binary() is not None

    def speak(self, text: str, voice: VoiceDirection) -> dict[str, Any]:
        binary = self._binary()
        if binary is None:
            raise SpeechError("espeak-ng binary not found.")
        wpm = pace_to_espeak_wpm(effective_pace(voice.pace, self.speed), self.base_wpm)
        if voice.pause_before_ms > 0:
            time.sleep(voice.pause_before_ms / 1000)
        started = time.perf_counter()
        try:
            subprocess.run(
                [binary, "-v", self.voice, "-s", str(wpm), "-p", str(self.pitch), "--", text],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as exc:
            raise SpeechError(
                f"espeak-ng failed: {exc.stderr.decode(errors='replace')[:300]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SpeechError("espeak-ng timed out after 120 s.") from exc
        return {
            "engine": self.name,
            "spoken": True,
            "wpm": wpm,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }


def _wav_player() -> list[str] | None:
    """First available local WAV player command prefix (offline only)."""
    if sys.platform == "win32" and shutil.which("powershell.exe"):
        return ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]
    for candidate, args in (
        ("aplay", ["-q"]),
        ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
        ("paplay", []),
    ):
        if shutil.which(candidate):
            return [candidate, *args]
    return None


def _play_wav(path: Path) -> None:
    """Play a WAV and block until it has finished.

    Blocking matters: the listener reopens the microphone the moment this
    returns, so returning early would let AZMO hear his own tail.

    The path goes through an environment variable rather than being pasted into
    the PowerShell command - a temp path containing a quote would otherwise
    break the command (or worse, extend it).
    """
    player = _wav_player()
    if player is None:
        raise SpeechError("Synthesized audio, but no local WAV player was found.")
    if player[0] == "powershell.exe":
        play = "(New-Object Media.SoundPlayer $env:AZMO_PLAY_WAV).PlaySync()"
        subprocess.run(
            [*player, play], check=True, capture_output=True, timeout=300,
            env={**_safe_env(), "AZMO_PLAY_WAV": str(path)},
        )
    else:
        subprocess.run([*player, str(path)], check=True, capture_output=True, timeout=300)


def _apply_dsp(
    wav_path: Path,
    voice: VoiceDirection,
    dsp: VoiceDspConfig | None,
    anchor: voice_dsp.GainAnchor | None = None,
) -> bool:
    """Apply the azmo-voice chain to a rendered WAV in place. Returns True if it ran.

    ``anchor`` is only passed when a reply is being rendered in pieces while it
    streams; it keeps every piece in one gain frame. See ``voice_dsp.GainAnchor``.
    """
    if dsp is None or not dsp.enabled:
        return False
    return voice_dsp.process_wav(str(wav_path), str(wav_path), voice, dsp, anchor=anchor)


class PiperSpeech(SpeechAdapter):
    """Piper neural TTS (offline once a voice model file is present).

    Configure ``speech.piper_model_path`` to point at a downloaded ``.onnx``
    voice (its ``.onnx.json`` must sit beside it). Synthesis writes a WAV, the
    azmo-voice DSP chain is applied, and the result is played locally.
    """

    name = "piper"

    def __init__(self, model_path: str | Path | None, dsp: VoiceDspConfig | None = None,
                 speed: float = 1.0):
        self.model_path = Path(model_path) if model_path else None
        self.dsp = dsp
        self.speed = speed

    def available(self) -> bool:
        if self.model_path is None or not self.model_path.exists():
            return False
        try:
            import piper  # noqa: F401
        except ImportError:
            return shutil.which("piper") is not None
        return True

    def speak(self, text: str, voice: VoiceDirection) -> dict[str, Any]:
        if self.model_path is None:
            raise SpeechError("No piper voice model configured (speech.piper_model_path).")
        length_scale = pace_to_piper_length_scale(effective_pace(voice.pace, self.speed))
        wav_path = _temp_wav()
        started = time.perf_counter()
        try:
            command = [
                sys.executable, "-m", "piper",
                "-m", str(self.model_path),
                "-f", str(wav_path),
                "--length-scale", str(length_scale),
            ]
            if shutil.which("piper") and _module_missing("piper"):
                command = [
                    "piper", "-m", str(self.model_path), "-f", str(wav_path),
                    "--length-scale", str(length_scale),
                ]
            subprocess.run(
                command, input=text.encode("utf-8"), check=True,
                capture_output=True, timeout=300,
            )
            dsp_ran = _apply_dsp(wav_path, voice, self.dsp)
            if voice.pause_before_ms > 0:
                time.sleep(voice.pause_before_ms / 1000)
            _play_wav(wav_path)
        except subprocess.CalledProcessError as exc:
            raise SpeechError(
                f"Piper speech failed: {exc.stderr.decode(errors='replace')[:300]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SpeechError("Piper synthesis or playback timed out.") from exc
        finally:
            wav_path.unlink(missing_ok=True)
        return {
            "engine": self.name,
            "spoken": True,
            "length_scale": length_scale,
            "dsp": dsp_ran,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }


class XttsCloneSpeech(SpeechAdapter):
    """Voice-cloned neural TTS via Coqui XTTS v2, plus the azmo-voice DSP chain.

    Clones the target voice from clean reference audio (``clone_reference_path``,
    a single WAV or a directory of clips) and speaks arbitrary text in it, then
    runs the demonic DSP on top.

    Quality path (used when the low-level model is reachable): the speaker
    conditioning latents are computed **once** from the reference clips, cached
    to disk, and reused for every line — this keeps the voice consistent
    utterance-to-utterance and skips re-analysis, and lets us pass tuned XTTS
    generation params (temperature, repetition penalty, etc.). If any of that is
    unavailable it falls back to the high-level ``tts_to_file`` API, and finally
    to a minimal call — so it degrades gracefully across coqui-tts versions.

    Heavy deps (coqui-tts / torch) and the ~2 GB model load lazily. Best on a
    CUDA GPU (the dev machine's RTX 3080 Ti); CPU works but is slow.
    """

    name = "clone"

    def __init__(
        self,
        reference_path: str | Path | None,
        model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        language: str = "en",
        dsp: VoiceDspConfig | None = None,
        speed: float = 1.0,
        params: dict[str, Any] | None = None,
        seed: int = 0,
        latent_cache: str | Path | None = None,
        device: str = "auto",
        max_chars: int = 220,
        chunk_gap_ms: int = 120,
        disable_cudnn: bool = False,
    ):
        self.reference_path = Path(reference_path) if reference_path else None
        self.model = model
        self.language = language
        self.dsp = dsp
        self.speed = speed
        self.params = dict(params or {})
        self.seed = seed
        self.latent_cache = Path(latent_cache) if latent_cache else None
        # "auto" | "cuda" | "cpu". Forcing cpu is the fallback when the GPU
        # stack itself is unstable: much slower, but it always produces audio.
        self.device = device
        # Kept safely under XTTS_CHARACTER_LIMIT - at the limit exactly, a
        # trailing token can still tip the generation over.
        self.max_chars = max(40, min(int(max_chars), XTTS_CHARACTER_LIMIT - 20))
        self.chunk_gap_ms = max(0, int(chunk_gap_ms))
        self.disable_cudnn = disable_cudnn
        self._api = None      # high-level TTS.api instance
        self._xtts = None     # low-level Xtts model (for cached-latent inference)
        self._latents = None  # (gpt_cond_latent, speaker_embedding)

    def reference_clips(self) -> list[str]:
        """The reference as a list of WAV paths (all clips if it's a directory)."""
        ref = self.reference_path
        if ref is None:
            return []
        if ref.is_dir():
            return sorted(str(p) for p in ref.glob("*.wav"))
        return [str(ref)] if ref.exists() else []

    def available(self) -> bool:
        if not self.reference_clips():
            return False
        try:
            import TTS  # noqa: F401  (provided by the coqui-tts package)
        except ImportError:
            return False
        return True

    # -- lazy model + latents ------------------------------------------------
    def _engine(self):
        if self._api is None:
            try:
                import torch
                from TTS.api import TTS as CoquiTTS
            except ImportError as exc:  # pragma: no cover - needs the clone extra
                raise SpeechError(
                    'Voice cloning needs the clone extra: pip install -e ".[clone]"'
                ) from exc
            disable_cudnn = self.disable_cudnn
            # cuDNN escape hatch. `cudnnGetLibConfig` is a cuDNN 9 symbol; if
            # the cuDNN DLLs on the machine are version 8 (commonly pinned there
            # by an older ctranslate2, which faster-whisper pulls in), torch
            # asks for a symbol that does not exist. The loader prints
            # "Could not load symbol cudnnGetLibConfig. Error code 127"
            # (127 = ERROR_PROC_NOT_FOUND) and the process aborts natively soon
            # after - in practice when XTTS reaches its conv-heavy HiFiGAN
            # decoder, several seconds into synthesis.
            #
            # Turning cuDNN off makes torch use its built-in convolution
            # kernels. Slightly slower, still fully on the GPU, and it does not
            # need the broken library at all. Setting the flag does not load
            # cuDNN (unlike reading .version()), so this is safe to do here.
            if disable_cudnn:
                try:
                    torch.backends.cudnn.enabled = False
                except Exception:  # noqa: BLE001 - never fatal
                    pass
            if self.device == "cpu":
                use_gpu = False
            elif self.device == "cuda":
                use_gpu = True
            else:
                use_gpu = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
            # Must precede the load: torch 2.6+ refuses coqui's pickled config
            # objects unless they are allowlisted first. See allow_xtts_globals.
            allow_xtts_globals()
            self._api = CoquiTTS(self.model).to("cuda" if use_gpu else "cpu")
            # Reach the low-level Xtts model for cached-latent inference.
            self._xtts = getattr(getattr(self._api, "synthesizer", None), "tts_model", None)
        return self._api

    def warm(self) -> None:
        """Load the model and speaker latents now (no audio produced)."""
        if not self.reference_clips():
            return
        self._engine()
        try:
            self._conditioning()
        except Exception:  # noqa: BLE001 - warmup is best-effort
            pass

    def _conditioning(self):
        """Compute (and cache) the speaker latents once from the reference."""
        if self._latents is not None:
            return self._latents
        if self._xtts is None or not hasattr(self._xtts, "get_conditioning_latents"):
            return None
        import torch
        clips = self.reference_clips()
        if self.latent_cache and self.latent_cache.exists() and clips:
            newest = max(Path(c).stat().st_mtime for c in clips)
            if self.latent_cache.stat().st_mtime >= newest:
                try:
                    # Written by torch.save on this machine, so the weights-only
                    # guard would only be protecting us from ourselves - and
                    # under torch 2.6 it rejects the cache and silently forces a
                    # full re-analysis of the reference clips on every launch.
                    self._latents = torch_load_trusted(self.latent_cache, map_location="cpu")
                    return self._latents
                except Exception:  # noqa: BLE001 - stale/incompatible cache
                    pass
        gpt_cond_latent, speaker_embedding = self._xtts.get_conditioning_latents(audio_path=clips)
        self._latents = (gpt_cond_latent, speaker_embedding)
        if self.latent_cache:
            try:
                self.latent_cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self._latents, self.latent_cache)
            except Exception:  # noqa: BLE001
                pass
        return self._latents

    # -- synthesis -----------------------------------------------------------
    @staticmethod
    def _inference_context():
        """torch.no_grad() when torch is present, else a no-op.

        Deliberately ``no_grad`` and not ``inference_mode``. inference_mode is
        stricter: the tensors it produces are tagged, and any later attempt to
        use one where autograd metadata is expected is a hard error rather than
        a slow path. XTTS's generation loop runs through HuggingFace code that
        does exactly that, so inference_mode crashed synthesis. no_grad gives
        essentially the same VRAM and speed benefit with none of that risk.
        """
        try:
            import torch

            return torch.no_grad()
        except Exception:  # noqa: BLE001 - torch missing or too old
            from contextlib import nullcontext

            return nullcontext()

    def _synthesize(self, text: str, wav_path: Path, speed: float) -> None:
        """Render text to wav_path. Tries cached-latent inference, then the
        high-level API, then a minimal call. Raises SpeechError only if all fail.
        """
        with self._inference_context():
            self._synthesize_inner(text, wav_path, speed)

    def _synthesize_inner(self, text: str, wav_path: Path, speed: float) -> None:
        self._engine()
        if self.seed:
            try:
                import torch
                torch.manual_seed(self.seed)
            except Exception:  # noqa: BLE001
                pass

        # 1) Best: low-level inference with cached latents + tuned params.
        latents = None
        try:
            latents = self._conditioning()
        except Exception:  # noqa: BLE001 - fall through to high-level
            latents = None
        if latents is not None and hasattr(self._xtts, "inference"):
            try:
                import numpy as np
                import soundfile as sf
                gpt_cond_latent, speaker_embedding = latents

                # Every chunk is guaranteed to be inside the model's window, so
                # XTTS's own splitting is left off and never sees long input.
                chunks = split_for_xtts(text, self.max_chars)
                pieces = []
                for chunk in chunks:
                    # Re-seed per chunk: identical sampling state each time, so
                    # the character does not drift between sentences.
                    if self.seed:
                        try:
                            import torch
                            torch.manual_seed(self.seed)
                        except Exception:  # noqa: BLE001
                            pass
                    out = self._xtts.inference(
                        chunk,
                        self.language,
                        gpt_cond_latent,
                        speaker_embedding,
                        temperature=self.params.get("temperature", 0.7),
                        length_penalty=self.params.get("length_penalty", 1.0),
                        repetition_penalty=self.params.get("repetition_penalty", 3.0),
                        top_k=self.params.get("top_k", 50),
                        top_p=self.params.get("top_p", 0.85),
                        speed=speed,
                        enable_text_splitting=False,
                    )
                    pieces.append(np.asarray(out["wav"], dtype="float32").reshape(-1))

                if not pieces:
                    raise SpeechError("Nothing to speak.")
                if len(pieces) == 1:
                    audio = pieces[0]
                else:
                    # A short breath between chunks; without it the joins sound
                    # clipped together rather than spoken.
                    gap = np.zeros(int(24000 * self.chunk_gap_ms / 1000), dtype="float32")
                    joined: list = []
                    for index, piece in enumerate(pieces):
                        if index:
                            joined.append(gap)
                        joined.append(piece)
                    audio = np.concatenate(joined)
                # DSP runs later, once, over the whole reply - so peak
                # normalisation is consistent across every chunk.
                sf.write(str(wav_path), audio, 24000)
                return
            except SpeechError:
                raise
            except Exception:  # noqa: BLE001 - fall back to high-level API
                pass

        # 2) High-level API with the tuned params.
        clips = self.reference_clips()
        try:
            self._api.tts_to_file(
                text=text, speaker_wav=clips, language=self.language,
                file_path=str(wav_path), speed=speed,
                temperature=self.params.get("temperature", 0.7),
                repetition_penalty=self.params.get("repetition_penalty", 3.0),
                top_k=self.params.get("top_k", 50),
                top_p=self.params.get("top_p", 0.85),
                length_penalty=self.params.get("length_penalty", 1.0),
            )
            return
        except TypeError:
            pass  # older/newer signature — use the minimal call
        except Exception as exc:  # noqa: BLE001
            raise SpeechError(f"Voice clone synthesis failed: {exc}") from exc

        # 3) Minimal, maximally-compatible call.
        try:
            self._api.tts_to_file(
                text=text, speaker_wav=clips, language=self.language,
                file_path=str(wav_path), speed=speed,
            )
        except Exception as exc:  # noqa: BLE001
            raise SpeechError(f"Voice clone synthesis failed: {exc}") from exc

    def render_to_file(self, text: str, voice: VoiceDirection, out_path: str | Path,
                       dsp: VoiceDspConfig | None = "__default__",
                       anchor: voice_dsp.GainAnchor | None = None) -> bool:
        """Synthesize + DSP to a file without playing it.

        Used by ``voicetune``, by ``presence build``, and - with an ``anchor`` -
        by ``StreamedSpeech`` for one chunk of a reply that is still arriving.
        """
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._synthesize(text, out, effective_pace(voice.pace, self.speed))
        chain = self.dsp if dsp == "__default__" else dsp
        return _apply_dsp(out, voice, chain, anchor=anchor)

    def speak(self, text: str, voice: VoiceDirection) -> dict[str, Any]:
        if not self.reference_clips():
            raise SpeechError(
                "No clone reference found (speech.clone_reference_path). "
                "Expected a WAV or a directory of clips."
            )
        wav_path = _temp_wav()
        started = time.perf_counter()
        try:
            self._synthesize(text, wav_path, effective_pace(voice.pace, self.speed))
            dsp_ran = _apply_dsp(wav_path, voice, self.dsp)
            if voice.pause_before_ms > 0:
                time.sleep(voice.pause_before_ms / 1000)
            _play_wav(wav_path)
        except SpeechError:
            raise
        except Exception as exc:  # pragma: no cover - runtime/model failures
            raise SpeechError(f"Voice clone synthesis failed: {exc}") from exc
        finally:
            wav_path.unlink(missing_ok=True)
        return {
            "engine": self.name,
            "spoken": True,
            "dsp": dsp_ran,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }


# ---------------------------------------------------------------------------
# Streamed delivery (0.2.10)
# ---------------------------------------------------------------------------

class StreamedSpeech:
    """Render a reply chunk by chunk while it is still being written, then
    deliver it as one unbroken line.

    The design log (2026-07-30) settled the shape of this and it is worth
    restating, because the obvious version is the wrong one. Pure overlap -
    play chunk 1 the instant it exists - buys the largest latency win and buys
    it at the cost of the *one* failure this project cannot accept: if chunk 3
    is not rendered before playback drains chunk 2, he stutters mid-sentence. A
    pause before he speaks reads as deliberate. A gap inside a line reads as
    broken, and presence already exists to cover the front of the turn.

    So playback does not begin until ``prebuffer`` chunks are rendered (or the
    reply turns out to be shorter than that). That deliberately spends part of
    the latency win to make a stall unlikely rather than merely rare.

    ``stalls`` counts the times playback still had to wait on the renderer. It
    is reported per turn on purpose: it is the only direct evidence of the
    failure this class exists to prevent, and a non-zero count is the signal to
    raise ``prebuffer`` rather than to guess.

    Threading: one worker renders, the calling thread plays. Playback stays on
    the caller's thread so ``speak``'s half-duplex contract is unchanged - when
    ``play`` returns, the sound is genuinely finished and the microphone can
    safely reopen.
    """

    #: Pushed onto the queue by the renderer to mean "no more chunks".
    _END = object()

    def __init__(
        self,
        adapter: XttsCloneSpeech,
        voice: VoiceDirection,
        prebuffer: int = 2,
        play: Callable[[Path], None] | None = None,
    ) -> None:
        self.adapter = adapter
        self.voice = voice
        self.prebuffer = max(1, int(prebuffer))
        self._play = play if play is not None else _play_wav
        # One anchor for the whole reply: chunk 1 sets the gain frame and every
        # later chunk reuses it, so the reply does not pump between chunks.
        self._anchor = voice_dsp.GainAnchor()
        self._ready: list[Path] = []
        self._cond = threading.Condition()
        self._done = False
        self._error: Exception | None = None
        self._worker: threading.Thread | None = None
        self._rendered = 0
        self.stalls = 0
        self.dsp_ran = False

    # -- production ---------------------------------------------------------
    def begin(self, chunks: Iterable[str]) -> None:
        """Start rendering ``chunks`` in the background."""
        if self._worker is not None:
            raise SpeechError("This stream has already been started.")
        self._worker = threading.Thread(
            target=self._render_all, args=(chunks,), name="azmo-render", daemon=True
        )
        self._worker.start()

    def _render_all(self, chunks: Iterable[str]) -> None:
        try:
            for text in chunks:
                if not text.strip():
                    continue
                path = _temp_wav()
                ran = self.adapter.render_to_file(
                    text, self.voice, path, anchor=self._anchor
                )
                with self._cond:
                    self.dsp_ran = self.dsp_ran or bool(ran)
                    self._ready.append(path)
                    self._rendered += 1
                    self._cond.notify_all()
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller below
            with self._cond:
                self._error = exc
        finally:
            with self._cond:
                self._done = True
                self._cond.notify_all()

    # -- consumption --------------------------------------------------------
    def await_prebuffer(self, timeout: float | None = None) -> int:
        """Block until the prebuffer is full, or the reply is fully rendered.

        Returns how many chunks are ready. Called inside the presence block, so
        that his contemplation covers not just the model but the first passes of
        synthesis too - and the breath drains only once he is ready to speak
        without interruption.
        """
        deadline = None if timeout is None else time.perf_counter() + timeout
        with self._cond:
            while len(self._ready) < self.prebuffer and not self._done:
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                self._cond.wait(remaining)
            return len(self._ready)

    def play(self) -> dict[str, Any]:
        """Play every chunk in order, blocking until the last one ends."""
        started = time.perf_counter()
        index = 0
        spoken = 0
        if self.voice.pause_before_ms > 0:
            time.sleep(self.voice.pause_before_ms / 1000)
        try:
            while True:
                with self._cond:
                    if index >= len(self._ready) and not self._done:
                        # The renderer has fallen behind playback: the stutter
                        # this class exists to avoid. Counted, not hidden.
                        self.stalls += 1
                        while index >= len(self._ready) and not self._done:
                            self._cond.wait()
                    if index >= len(self._ready):
                        break
                    path = self._ready[index]
                try:
                    self._play(path)
                    spoken += 1
                finally:
                    path.unlink(missing_ok=True)
                index += 1
        finally:
            self.close()

        if self._error is not None and spoken == 0:
            raise SpeechError(f"Streamed synthesis failed: {self._error}")

        return {
            "engine": self.adapter.name,
            "spoken": spoken > 0,
            "dsp": self.dsp_ran,
            "streamed": True,
            "chunks": spoken,
            "stalls": self.stalls,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": None if self._error is None else str(self._error),
        }

    def close(self) -> None:
        """Stop waiting and remove any chunk that was rendered but not played."""
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=1.0)
        with self._cond:
            leftovers = list(self._ready)
            self._ready = []
            self._done = True
        for path in leftovers:
            path.unlink(missing_ok=True)


def streaming_supported(adapter: SpeechAdapter) -> bool:
    """True when this engine can render a chunk without playing it.

    Only the clone can: streaming needs render-to-file so chunks can be built
    ahead of the playhead. SAPI and espeak synthesise straight to the speaker,
    and piper has no per-chunk gain frame, so they keep the whole-reply path.
    That is not a limitation worth fixing - the clone is the voice.
    """
    return isinstance(adapter, XttsCloneSpeech)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _safe_env() -> dict[str, str]:
    import os

    return dict(os.environ)


def _module_missing(name: str) -> bool:
    try:
        __import__(name)
        return False
    except ImportError:
        return True


def select_speech_adapter(config: SpeechConfig) -> SpeechAdapter:
    """Pick the best available engine, honoring an explicit choice first."""
    if not config.enabled or config.engine == "none":
        return NullSpeech()

    candidates: list[SpeechAdapter]
    clone = XttsCloneSpeech(
        config.clone_reference_path,
        model=config.clone_model,
        language=config.clone_language,
        dsp=config.dsp,
        speed=config.speed,
        params={
            "temperature": config.clone_temperature,
            "repetition_penalty": config.clone_repetition_penalty,
            "top_k": config.clone_top_k,
            "top_p": config.clone_top_p,
            "length_penalty": config.clone_length_penalty,
            "split_text": config.clone_split_text,
        },
        seed=config.clone_seed,
        latent_cache=config.clone_latent_cache,
        device=config.clone_device,
        max_chars=config.clone_max_chars,
        chunk_gap_ms=config.clone_chunk_gap_ms,
        disable_cudnn=config.clone_disable_cudnn,
    )
    piper = PiperSpeech(config.piper_model_path, dsp=config.dsp, speed=config.speed)
    sapi = SapiSpeech(voice_hint=config.sapi_voice_hint, volume=config.volume,
                      dsp=config.dsp, speed=config.speed)
    espeak = EspeakNgSpeech(base_wpm=config.espeak_base_wpm, dsp=config.dsp, speed=config.speed)

    if config.engine == "auto":
        candidates = [clone, piper, sapi, espeak]
    else:
        by_name = {"clone": clone, "piper": piper, "sapi": sapi, "espeak": espeak}
        candidates = [by_name[config.engine]]

    for adapter in candidates:
        if adapter.available():
            return adapter
    return NullSpeech()
