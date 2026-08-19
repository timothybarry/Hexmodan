"""azmo-listener: microphone -> voice-activity -> wake word -> speech-to-text.

The "Hear" half of the pipeline (project brief section 7). Flow:

    mic (16 kHz) -> webrtcvad segments your speech -> faster-whisper transcribes
    -> a pluggable WakeDetector decides if you said "Azmodan" -> the rest of the
    sentence is returned as the command for AZMO Mind.

Design choices:

- **Wake word is pluggable.** The default ``WhisperWake`` reuses the same
  faster-whisper we already need (no extra service, key, or trained model), with
  tolerant matching because Whisper spells an unusual name loosely. On the Jetson
  a leaner always-on hotword engine (openWakeWord / Porcupine) can replace it via
  the same ``WakeDetector`` interface, for a smaller always-on footprint.
- **Strict half-duplex, enforced at the audio callback.** AZMO says his own name
  constantly, so any audio of his own voice reaching Whisper is a self-trigger
  and therefore an infinite loop. Three independent guards stop that:

    1. ``MicStream`` is *gated*: while AZMO is thinking or speaking the capture
       callback discards frames instead of queueing them. Nothing to leak.
    2. A cooldown after playback ends covers the speaker tail and room
       reverb, then the buffer is drained before the gate reopens.
    3. ``EchoGuard`` compares each fresh transcript against what AZMO just said
       and rejects near-matches, in case his voice reaches the mic anyway
       (loud speakers, a shared audio device, a stuck gate).

- **Nothing blocks forever.** Every queue read has a timeout and honours a stop
  event, so Ctrl+C always lands and a dead audio device cannot hang the loop.
- **Lazy heavy imports.** sounddevice / webrtcvad / faster-whisper load only when
  the listener actually runs, so importing this module is cheap and the rest of
  AZMO works without the listen extra installed.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from contextlib import contextmanager
from difflib import SequenceMatcher

from azmo_mind.config import ListenerConfig

SAMPLE_RATE = 16000          # webrtcvad + whisper both like 16 kHz mono
_FRAME_MS = 30              # webrtcvad accepts 10/20/30 ms frames
_FRAME_SAMPLES = SAMPLE_RATE * _FRAME_MS // 1000
_QUEUE_READ_TIMEOUT_S = 0.25  # how often a blocked read re-checks the stop flag
_DEAD_STREAM_TIMEOUT_S = 5.0  # silence this long with an open gate = suspect the device


class ListenerError(RuntimeError):
    """A recoverable listener/audio failure."""


def listener_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        import sounddevice  # noqa: F401
        import webrtcvad  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Wake detection (pluggable)
# ---------------------------------------------------------------------------

def _wake_variants(wake_word: str, extra: "list[str] | None" = None) -> list[str]:
    """Spellings Whisper is likely to produce for the wake word."""
    base = wake_word.strip().lower()
    variants = {base, base.replace(" ", ""), base.replace(" ", "-")}
    if base == "azmodan":
        # Only spellings that are not themselves ordinary English go in here,
        # because this list is matched *anywhere* in the sentence. Manglings
        # that are real phrases ("as madam", "as modern") are deliberately left
        # out: they are handled by the phonetic pass below, which only looks at
        # the start of an utterance and so cannot fire mid-sentence.
        variants.update({
            "azmodan", "azmodon", "azmoden", "asmodan", "asmodian",
            "as modan", "az modan", "azmo dan", "azmodann", "azmo",
            "ashmodan", "ozmodan", "osmodan", "as mo dan",
        })
    for item in extra or []:
        cleaned = item.strip().lower()
        if cleaned:
            variants.add(cleaned)
    return sorted(variants, key=len, reverse=True)


# Soundex-style consonant codes. Vowels (and h/w/y) carry almost no information
# when a speech model is guessing at an unfamiliar proper noun, so we drop them:
# "Azmodan" and "As Madam" collapse to the same key.
_SOUNDEX_CODES = {
    **dict.fromkeys("bfpv", "1"),
    **dict.fromkeys("cgjkqsxz", "2"),
    **dict.fromkeys("dt", "3"),
    "l": "4",
    **dict.fromkeys("mn", "5"),
    "r": "6",
}


def phonetic_key(word: str) -> str:
    """A Soundex-like key: first letter plus the consonant skeleton.

    Not truncated to the classic four characters - we want the whole skeleton,
    because a short key would match far too many ordinary English words.
    """
    letters = [c for c in word.lower() if c.isalpha()]
    if not letters:
        return ""
    key = letters[0].upper()
    previous = _SOUNDEX_CODES.get(letters[0], "")
    for char in letters[1:]:
        code = _SOUNDEX_CODES.get(char, "")
        if code and code != previous:
            key += code
        # h and w are transparent: they do not break a run of like consonants.
        if char not in "hw":
            previous = code
    return key


def squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _fuzzy_wake_end(words: list[str], wake_word: str, threshold: float,
                    max_span: int = 3, max_offset: int = 2) -> int | None:
    """Index just past the wake word if the opening words sound like it.

    Whisper renders an unfamiliar name differently almost every time
    ("As Madam", "As been in", "Az modern"), so an exact list of spellings can
    never be complete. Instead we compare the *sound* of the opening words
    against the wake word two ways: a character-similarity ratio, and the
    consonant skeleton from :func:`phonetic_key`.

    Only the first few words are considered. A fuzzy match anywhere in a long
    sentence would fire constantly on ordinary speech; at the start of an
    utterance, where a wake word actually belongs, it is safe.
    """
    target = squash(wake_word)
    if not target:
        return None
    target_key = phonetic_key(target)

    for offset in range(min(max_offset, len(words)) + 1):
        for span in range(1, max_span + 1):
            end = offset + span
            if end > len(words):
                break
            candidate = squash("".join(words[offset:end]))
            if not candidate:
                continue
            # A candidate wildly longer or shorter than the name is not it.
            if abs(len(candidate) - len(target)) > 3:
                continue
            if SequenceMatcher(None, candidate, target).ratio() >= threshold:
                return end
            if target_key and phonetic_key(candidate) == target_key:
                return end
    return None


def strip_wake(text: str, wake_word: str, fuzzy_threshold: float = 0.72,
               extra_variants: "list[str] | None" = None) -> str | None:
    """If ``text`` contains the wake word, return what follows it (the command);
    return an empty string if only the wake word was said, or None if absent.

    Exact spellings are matched first (fast, anywhere in the sentence); failing
    that, the opening words are matched phonetically. Set ``fuzzy_threshold`` to
    1.0 to disable the phonetic pass entirely.
    """
    normalized = " ".join(text.split())      # collapse whitespace, keep original case
    lowered = normalized.lower()              # same length, so indices line up
    for variant in _wake_variants(wake_word, extra_variants):
        pattern = r"\b" + re.escape(variant) + r"\b"
        match = re.search(pattern, lowered)
        if match:
            remainder = normalized[match.end():]
            return remainder.lstrip(" ,.;:!?-").strip()

    if fuzzy_threshold >= 1.0:
        return None
    words = normalized.split()
    if not words:
        return None
    end = _fuzzy_wake_end(words, wake_word, fuzzy_threshold)
    if end is None:
        return None
    return " ".join(words[end:]).lstrip(" ,.;:!?-").strip()


class WakeDetector:
    """Given an already-transcribed utterance, decide if AZMO was addressed.

    Returns the command text (may be empty if only the wake word was spoken) or
    None if the wake word was not detected. A future hotword engine can implement
    this over raw audio instead of transcript text.
    """

    def command_from(self, transcript: str) -> str | None:
        raise NotImplementedError


class WhisperWake(WakeDetector):
    def __init__(self, wake_word: str = "Azmodan", always_on: bool = False,
                 fuzzy_threshold: float = 0.72,
                 extra_variants: "list[str] | None" = None):
        self.wake_word = wake_word
        self.always_on = always_on  # if True, every utterance is a command
        self.fuzzy_threshold = fuzzy_threshold
        self.extra_variants = list(extra_variants or [])

    def command_from(self, transcript: str) -> str | None:
        if self.always_on:
            return transcript.strip()
        return strip_wake(
            transcript, self.wake_word,
            fuzzy_threshold=self.fuzzy_threshold,
            extra_variants=self.extra_variants,
        )


# ---------------------------------------------------------------------------
# Echo suppression (last line of defence against self-hearing)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9']+")

# Words that carry no identifying signal; a transcript that overlaps AZMO's
# reply only on these is not an echo.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "have", "i", "in", "is", "it", "its", "not", "of", "on", "or", "that",
    "the", "their", "them", "they", "this", "to", "was", "were", "will",
    "with", "you", "your", "yours",
})


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def similarity(heard: str, spoken: str) -> float:
    """Fraction of the heard utterance's content words that AZMO just said.

    1.0 = every meaningful word came from his own reply (a pure echo);
    0.0 = nothing in common. Stopwords are ignored so a short human command that
    happens to share "the"/"you" with his reply is not mistaken for an echo.
    """
    heard_tokens = [t for t in _tokens(heard) if t not in _STOPWORDS]
    if not heard_tokens:
        return 0.0
    spoken_tokens = set(_tokens(spoken))
    if not spoken_tokens:
        return 0.0
    hits = sum(1 for t in heard_tokens if t in spoken_tokens)
    return hits / len(heard_tokens)


class EchoGuard:
    """Remembers what AZMO just said so his own voice cannot become a command.

    Only active for ``window_s`` seconds after he stops speaking: beyond that the
    audio physically cannot be his, and we must not censor a user who genuinely
    repeats his words back at him.
    """

    def __init__(self, window_s: float = 8.0, threshold: float = 0.6,
                 min_words: int = 2, wake_word: str = "Azmodan"):
        self.window_s = window_s
        self.threshold = threshold
        self.min_words = min_words
        # AZMO says his own name constantly, so the wake word carries no
        # evidence either way. It is excluded from the comparison entirely -
        # otherwise a user summoning him with a bare "Azmodan" in the seconds
        # after a reply would be silently swallowed as an echo of that reply.
        self.wake_word = wake_word
        self._spoken = ""
        self._until = 0.0

    def _content(self, text: str) -> list[str]:
        wake_tokens = set(_tokens(self.wake_word))
        return [
            t for t in _tokens(text)
            if t not in _STOPWORDS and t not in wake_tokens
        ]

    def remember(self, spoken: str, now: float | None = None) -> None:
        self._spoken = spoken or ""
        self._until = (time.monotonic() if now is None else now) + self.window_s

    def clear(self) -> None:
        self._spoken = ""
        self._until = 0.0

    def is_echo(self, transcript: str, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        if not self._spoken or current > self._until:
            return False
        content = self._content(transcript)
        if not content:
            # Nothing but stopwords and his own name. That is either a bare
            # summons or a scrap of noise - the capture gate and the cooldown
            # are the guards for that case, not this one.
            return False
        if len(content) < self.min_words:
            # Too short to judge by overlap. An echo only if the one content
            # word came from his reply.
            spoken_tokens = set(_tokens(self._spoken))
            return all(t in spoken_tokens for t in content)
        spoken_tokens = set(_tokens(self._spoken))
        hits = sum(1 for t in content if t in spoken_tokens)
        return hits / len(content) >= self.threshold


# ---------------------------------------------------------------------------
# Audio capture + VAD segmentation
# ---------------------------------------------------------------------------

class MicStream:
    """Captures 16 kHz mono and yields VAD-delimited speech segments.

    The stream stays physically open for the whole session (repeatedly opening
    and closing a WASAPI device is slow and flaky), but a *gate* in the capture
    callback decides whether frames are kept. While the gate is shut, audio is
    discarded at the source, so AZMO's own voice can never reach the queue.
    """

    def __init__(self, config: ListenerConfig):
        self.config = config
        maxlen = max(50, int(config.max_utterance_ms / _FRAME_MS) * 2)
        self._queue: "queue.Queue[bytes]" = queue.Queue(maxsize=maxlen)
        self._stream = None
        self._vad = None
        self._open_gate = threading.Event()
        self._open_gate.set()
        self._stop = threading.Event()
        self.dropped_frames = 0   # overflow counter, surfaced by diagnostics

    # -- lifecycle -----------------------------------------------------------
    def open(self) -> None:
        try:
            import sounddevice as sd
            import webrtcvad
        except ImportError as exc:  # pragma: no cover - needs the listen extra
            raise ListenerError(
                'Listening needs the listen extra: pip install -e ".[listen]"'
            ) from exc

        self._vad = webrtcvad.Vad(int(self.config.vad_aggressiveness))
        self._stop.clear()

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            # Gate first: while AZMO thinks or speaks we throw the audio away
            # here, before it can ever be transcribed. This is the primary
            # defence against the self-hearing feedback loop.
            if not self._open_gate.is_set():
                return
            try:
                self._queue.put_nowait(bytes(indata))
            except queue.Full:
                # Keep the newest audio: drop the oldest frame, then retry once.
                self.dropped_frames += 1
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(bytes(indata))
                except (queue.Empty, queue.Full):
                    pass

        device = self.config.mic_device
        try:
            self._stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=_FRAME_SAMPLES,
                device=device if device is not None and device >= 0 else None,
                dtype="int16",
                channels=1,
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001 - surface as a listener error
            self._stream = None
            raise ListenerError(f"Could not open the microphone: {exc}") from exc

    def close(self) -> None:
        self._stop.set()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            except Exception:  # noqa: BLE001 - closing a dead device
                pass
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        self.drain()

    # -- gating --------------------------------------------------------------
    @property
    def gated(self) -> bool:
        return not self._open_gate.is_set()

    def pause(self) -> None:
        """Stop accepting audio (call before AZMO thinks or speaks)."""
        self._open_gate.clear()
        self.drain()

    def resume(self, cooldown_s: float = 0.0) -> None:
        """Wait out the speaker tail, discard anything buffered, then listen."""
        if cooldown_s > 0:
            time.sleep(cooldown_s)
        self.drain()
        self._open_gate.set()

    def drain(self) -> None:
        """Discard buffered audio (belt-and-braces alongside the gate)."""
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    # -- capture -------------------------------------------------------------
    def _stream_is_dead(self) -> bool:
        """True if the capture device stopped delivering audio underneath us."""
        stream = self._stream
        if stream is None:
            return True
        active = getattr(stream, "active", None)
        return active is False

    def _next_frame(self, deadline: float | None):
        """One 30 ms frame, or None if we stopped or ran out of time.

        Raises ListenerError if the audio device dies mid-session. Without this
        the loop would spin silently forever on a unplugged or reset mic, which
        looks identical to "AZMO stopped responding".
        """
        silent_reads = 0
        while not self._stop.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                return None
            try:
                frame = self._queue.get(timeout=_QUEUE_READ_TIMEOUT_S)
            except queue.Empty:
                silent_reads += 1
                # Only suspect the device once it has been quiet for a while AND
                # the gate is open (silence while gated is expected, not a fault).
                if (
                    silent_reads * _QUEUE_READ_TIMEOUT_S >= _DEAD_STREAM_TIMEOUT_S
                    and self._open_gate.is_set()
                    and self._stream_is_dead()
                ):
                    raise ListenerError(
                        "The microphone stopped delivering audio. Reconnect it "
                        "or pick a device with listener.mic_device, then restart."
                    )
                continue
            silent_reads = 0
            return frame
        return None

    def next_utterance(self, timeout_s: float | None = None):
        """Block until a full spoken utterance is captured.

        Returns float32 audio @16 kHz, or ``None`` if the listener was stopped,
        the timeout expired, or the captured audio was too short to be speech.
        """
        voiced: list[bytes] = []
        silence_frames = 0
        started = False
        max_silence = max(1, self.config.end_silence_ms // _FRAME_MS)
        # Lead-in kept from before the VAD fired. A soft opening consonant (the
        # "Az" of "Azmodan") often does not trip the VAD, so without enough
        # pre-roll Whisper receives a decapitated word and guesses badly.
        pre_roll = max(1, self.config.pre_roll_ms // _FRAME_MS)
        min_frames = max(1, self.config.min_utterance_ms // _FRAME_MS)
        ring: list[bytes] = []
        deadline = None if timeout_s is None else time.monotonic() + timeout_s

        while True:
            frame = self._next_frame(deadline)
            if frame is None:
                if not started:
                    return None
                break            # timed out mid-sentence: keep what we have
            if len(frame) < _FRAME_SAMPLES * 2:
                continue
            is_speech = self._vad.is_speech(frame, SAMPLE_RATE)
            if not started:
                ring.append(frame)
                if len(ring) > pre_roll:
                    ring.pop(0)
                if is_speech:
                    started = True
                    voiced.extend(ring)
                    voiced.append(frame)
                    ring.clear()
                continue
            voiced.append(frame)
            if is_speech:
                silence_frames = 0
            else:
                silence_frames += 1
                if silence_frames >= max_silence:
                    break
            # guard against a runaway (someone holding the floor forever)
            if len(voiced) * _FRAME_MS > self.config.max_utterance_ms:
                break

        # Reject blips: a door, a keystroke, a cough. Transcribing these wastes
        # seconds of Whisper time and invites hallucinated wake words.
        speech_frames = len(voiced) - min(len(voiced), pre_roll) - silence_frames
        if speech_frames < min_frames:
            return None

        pcm = b"".join(voiced)
        if not pcm:
            return None

        # numpy is imported only once there is real audio to convert, so the
        # early-return paths (stopped, timed out, too short) stay dependency-free.
        import numpy as np

        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

class Transcriber:
    def __init__(self, config: ListenerConfig):
        self.config = config
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - needs the listen extra
                raise ListenerError(
                    'Transcription needs the listen extra: pip install -e ".[listen]"'
                ) from exc
            self._model = WhisperModel(
                self.config.whisper_model,
                device=self.config.whisper_device,
                compute_type=self.config.whisper_compute_type,
                # Whisper stays on the CPU by default so it never competes with
                # the LLM or XTTS for the GPU (see the gpu section of the config).
                cpu_threads=max(1, self.config.whisper_cpu_threads),
            )
        return self._model

    def transcribe(self, audio) -> str:
        if audio is None or len(audio) == 0:
            return ""
        model = self._ensure()
        options = {
            "language": self.config.language,
            "beam_size": self.config.whisper_beam_size,
            "vad_filter": False,
        }
        # Tell Whisper the name exists. Left to itself it renders an unfamiliar
        # proper noun as whatever ordinary words it sounds like ("As Madam"),
        # which no amount of downstream matching can reliably undo. Biasing the
        # decoder is far more effective than correcting its output afterwards.
        prompt = self.config.whisper_initial_prompt
        if prompt:
            options["initial_prompt"] = prompt
        segments = None
        if prompt:
            try:
                # hotwords is a stronger, more targeted bias, but it only exists
                # in faster-whisper >= 1.0.2.
                segments, _info = model.transcribe(audio, hotwords=prompt, **options)
            except TypeError:
                segments = None
        if segments is None:
            segments, _info = model.transcribe(audio, **options)
        return " ".join(seg.text for seg in segments).strip()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Listener:
    """Wait-for-wake-word, capture-command loop over the microphone."""

    def __init__(self, config: ListenerConfig, wake: WakeDetector | None = None):
        self.config = config
        self.mic = MicStream(config)
        self.transcriber = Transcriber(config)
        self.wake = wake or WhisperWake(
            wake_word=config.wake_word,
            always_on=config.always_on,
            fuzzy_threshold=config.wake_fuzzy_threshold,
            extra_variants=config.extra_wake_variants,
        )
        self.echo = EchoGuard(
            window_s=config.echo_guard_window_ms / 1000,
            threshold=config.echo_similarity_threshold,
            wake_word=config.wake_word,
        )
        self._open = False

    def start(self) -> None:
        if not self._open:
            self.mic.open()
            self._open = True

    def stop(self) -> None:
        if self._open:
            self.mic.close()
            self._open = False

    def drain(self) -> None:
        self.mic.drain()

    def warmup(self) -> None:
        """Load the whisper model up front so the first turn isn't slow."""
        import numpy as np

        self.transcriber.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))

    # -- half-duplex ---------------------------------------------------------
    @contextmanager
    def deaf(self, spoken: str | None = None, cooldown_ms: int | None = None):
        """Hold the mic shut for the duration of the block.

        Wrap everything between hearing a command and the end of AZMO's reply in
        this. On exit it waits out the cooldown (speaker tail and room reverb),
        throws away anything buffered, arms the echo guard with what he said, and
        only then reopens the gate.

        ``cooldown_ms`` overrides ``post_speech_cooldown_ms`` for this block. The
        cooldown exists to cover the tail of a sound, and the tail is
        proportional to how loud and how long that sound was: 700 ms is sized for
        a full reply at volume. A short quiet non-verbal (the wake-word breath)
        needs far less, and over-waiting there is not harmless - the gate is shut
        while the user is being invited to speak, so every extra millisecond is
        an opportunity to eat the first word of their command.
        """
        self.mic.pause()
        try:
            yield
        finally:
            if spoken:
                self.echo.remember(spoken)
            cooldown = (
                self.config.post_speech_cooldown_ms if cooldown_ms is None else cooldown_ms
            )
            self.mic.resume(cooldown_s=max(0, cooldown) / 1000)

    # -- main loop -----------------------------------------------------------
    def wait_for_command(self, on_transcript=None, on_awaiting=None,
                         timeout_s: float | None = None):
        """Block until the wake word + a command are heard; return the command.

        ``on_transcript(text, is_wake)`` fires for every heard utterance.
        ``on_awaiting()`` fires when the wake word was heard alone and AZMO is now
        waiting for you to speak the command. Returns the command, or None so the
        caller keeps waiting.
        """
        self.start()
        audio = self.mic.next_utterance(timeout_s=timeout_s)
        if audio is None:
            return None
        transcript = self.transcriber.transcribe(audio)
        if not transcript:
            return None

        # Guard 3: even if the gate leaked, do not act on AZMO's own words.
        if self.echo.is_echo(transcript):
            if on_transcript:
                on_transcript(transcript, False)
            return None

        command = self.wake.command_from(transcript)
        if command is None:
            if on_transcript:
                on_transcript(transcript, False)
            return None
        if on_transcript:
            on_transcript(transcript, True)
        if command:
            return command

        # Only the wake word was spoken - capture the next utterance as the
        # command, but never wait forever for one.
        if on_awaiting:
            on_awaiting()
        time.sleep(self.config.wake_cooldown_ms / 1000)
        self.mic.drain()
        follow = self.mic.next_utterance(
            timeout_s=self.config.follow_up_timeout_ms / 1000
        )
        if follow is None:
            return None
        text = self.transcriber.transcribe(follow).strip()
        if not text or self.echo.is_echo(text):
            return None
        return text
