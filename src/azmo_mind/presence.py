"""azmo-presence: the sounds he makes while he is not speaking.

The problem this solves is not latency, it is *dead air*. A machine that sits
silent for six seconds reads as broken; one that audibly turns the question over
for eight reads as thinking. See ``docs/DESIGN_LOG.md`` (2026-07-30).

AZMO has no body yet, so audio is his only channel. This module plays short
**pre-rendered** non-verbals — a slow deliberate exhale, a low considering growl
— while the LLM works. Pre-rendered is the whole point: no model, no GPU, no
synthesis at request time. Selecting and starting a clip costs a file read, so
the first sound lands in well under a second no matter how long the reply takes.

Three properties matter, and each is a deliberate design choice:

**A pool, not a clip.** One breath on repeat becomes a tic inside a single
session. ``pick`` will not replay a clip while it sits in the recent window.

**Sustained, not one-shot.** A sound at t=0 does nothing for the person still
waiting at t=9. ``thinking()`` keeps breathing every ``sustain_gap_ms`` for as
long as the turn runs, up to a cap.

**It never overlaps his speech.** ``thinking()`` blocks on exit until the
current clip has drained (bounded by ``max_drain_ms``). AZMO talking over his own
breath would read as a glitch, which is the exact thing we are removing.

The module does not care where the WAVs came from. ``azmo presence build``
renders them through the existing clone engine and the azmo-voice chain, but
hand-recorded or sourced clips dropped into the pool directory work identically.

Half-duplex note: playing audio while the microphone is open would feed his own
breath back into the transcriber. Every call site must be inside
``Listener.deaf()``. ``thinking()`` is used inside the existing deaf window in
``cli.listen``; the bare-wake acknowledgement opens its own.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from azmo_mind.config import PresenceConfig

# Clip kinds. These are directory names under ``presence.clips_path`` and the
# keys used in ``presence.weights``.
KINDS: tuple[str, ...] = ("exhale", "growl")

WAV_GLOB = "*.wav"


def _clip_sort_key(path: Path) -> str:
    return path.name.lower()


class PresencePlayer:
    """Selects and plays pre-rendered non-verbals from a pool on disk.

    Layout::

        data/presence/
          exhale/*.wav
          growl/*.wav

    Missing directories, an empty pool, or a broken audio player are all
    non-fatal: presence is an enhancement, and a turn that produces a real reply
    with no breath in front of it is still a good turn. Every entry point
    degrades to a no-op rather than raising into the conversation loop.
    """

    def __init__(
        self,
        config: PresenceConfig,
        play: Callable[[Path], None] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        # Injected so tests never touch a real audio device, and so the caller
        # can route playback through whatever the platform supports.
        self._play = play if play is not None else _default_play
        self._rng = rng or random.Random()
        self._recent: deque[Path] = deque(maxlen=max(0, config.avoid_repeat_window))
        self._lock = threading.Lock()

    # -- pool ---------------------------------------------------------------
    def clips(self, kind: str | None = None) -> list[Path]:
        """Every WAV in the pool, or just those of one kind. Sorted, so the
        pool is stable across runs and a seeded rng is reproducible."""
        root = Path(self.config.clips_path)
        kinds: Sequence[str] = (kind,) if kind else self.enabled_kinds()
        found: list[Path] = []
        for name in kinds:
            directory = root / name
            if directory.is_dir():
                found.extend(sorted(directory.glob(WAV_GLOB), key=_clip_sort_key))
        return found

    def enabled_kinds(self) -> tuple[str, ...]:
        """Kinds with a positive weight, in a stable order."""
        return tuple(k for k in KINDS if self.config.weights.get(k, 0.0) > 0)

    def available(self) -> bool:
        """True when presence is switched on and there is at least one clip."""
        return bool(self.config.enabled) and bool(self.clips())

    def thinking_available(self) -> bool:
        """True when a contemplation track would actually make sound."""
        return bool(self.config.on_think) and self.available()

    def describe(self) -> dict[str, int]:
        """Clip count per kind — for ``azmo check`` and ``azmo presence list``."""
        return {kind: len(self.clips(kind)) for kind in KINDS}

    # -- selection ----------------------------------------------------------
    def pick(self, kind: str | None = None) -> Path | None:
        """Choose a clip, avoiding anything played recently.

        Kind is chosen first, by weight, then a clip within it. Choosing the
        kind first keeps the exhale/growl balance honest even when one folder
        holds far more clips than the other — but only kinds that still have an
        unplayed clip are eligible, so the weighting can never corner us into a
        forced choice.

        **The window is clamped to one less than the number of clips available.**
        A window as large as the pool would leave exactly zero (or exactly one)
        legal choices, and the selection stops being random: it becomes a fixed
        cycle, which is precisely the recognisable tic the window exists to
        prevent. Guarding against repetition by making the sequence predictable
        would be worse than not guarding at all.
        """
        with self._lock:
            pools = self._pools(kind)
            if not pools:
                return None

            total = sum(len(clips) for _, clips in pools)
            blocked = self._blocked(total)

            eligible = [(k, [c for c in clips if c not in blocked]) for k, clips in pools]
            eligible = [(k, clips) for k, clips in eligible if clips]
            if not eligible:
                # Only reachable with a single clip in the whole pool. A repeated
                # breath still beats silence.
                eligible = pools

            chosen_pool = self._weighted_pool(eligible)
            chosen = self._rng.choice(chosen_pool)
            if self._recent.maxlen:
                self._recent.append(chosen)
            return chosen

    def _blocked(self, total: int) -> set[Path]:
        """Recently-played clips to avoid, clamped so real choice always remains.

        The clamp is ``total - 2``, not ``total - 1``. Leaving exactly one legal
        clip is not randomness — it is a fixed rotation with a period equal to
        the pool size, which is a *more* recognisable pattern than the occasional
        repeat the window was added to prevent.

        Two clips is the one case with no good answer: strict alternation is
        audible, but so is an immediate repeat. Alternation is the lesser evil,
        so the window still applies there.
        """
        if total <= 1 or not self._recent:
            return set()
        allowance = 1 if total == 2 else max(0, total - 2)
        keep = min(len(self._recent), allowance)
        if keep <= 0:
            return set()
        return set(list(self._recent)[-keep:])

    def _pools(self, kind: str | None) -> list[tuple[str, list[Path]]]:
        """Populated (kind, clips) pairs, honouring an explicit kind request."""
        kinds = (kind,) if kind is not None else self.enabled_kinds()
        return [(k, clips) for k in kinds if (clips := self.clips(k))]

    def _weighted_pool(self, pools: list[tuple[str, list[Path]]]) -> list[Path]:
        if len(pools) == 1:
            return pools[0][1]
        weights = [max(0.0, float(self.config.weights.get(k, 0.0))) for k, _ in pools]
        if sum(weights) <= 0:
            return [clip for _, clips in pools for clip in clips]
        return self._rng.choices([clips for _, clips in pools], weights=weights, k=1)[0]

    # -- playback -----------------------------------------------------------
    def play(self, kind: str | None = None) -> Path | None:
        """Play one clip and block until it finishes. Returns what was played.

        Never raises: a missing player or an unreadable WAV degrades to None.
        """
        if not self.config.enabled:
            return None
        clip = self.pick(kind)
        if clip is None:
            return None
        try:
            self._play(clip)
        except Exception:  # noqa: BLE001 - presence must never break a turn
            return None
        return clip

    @contextmanager
    def thinking(self, kind: str | None = None) -> Iterator[PresenceTrack]:
        """Breathe for as long as the block runs, then drain before returning.

        Plays immediately on entry, then every ``sustain_gap_ms`` while the
        block is still running, up to ``max_sustain_clips`` in total. On exit it
        stops scheduling and waits (at most ``max_drain_ms``) for the clip in
        flight, so his breath is finished before his words begin.

        A no-op when presence is unavailable or ``on_think`` is off, so call
        sites need no branch: the same ``with`` runs either way and
        ``track.played`` is simply empty.
        """
        track = PresenceTrack(self, kind)
        if not self.thinking_available():
            yield track
            return
        track.start()
        try:
            yield track
        finally:
            track.stop()


class PresenceTrack:
    """A background contemplation track: one clip, then another, until stopped.

    Runs on a daemon thread so a wedged turn can never hold the process open.
    ``played`` is the record of what was actually heard, which is what the
    listen loop reports and what the tests assert on.
    """

    def __init__(self, player: PresencePlayer, kind: str | None = None) -> None:
        self._player = player
        self._kind = kind
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._played: list[Path] = []
        self._lock = threading.Lock()

    @property
    def played(self) -> list[Path]:
        with self._lock:
            return list(self._played)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="azmo-presence", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        config = self._player.config
        gap = max(0, int(config.sustain_gap_ms)) / 1000
        limit = max(1, int(config.max_sustain_clips))
        while not self._stop.is_set() and len(self._played) < limit:
            clip = self._player.play(self._kind)
            if clip is None:
                # Nothing playable: stop rather than spin on a broken pool.
                return
            with self._lock:
                self._played.append(clip)
            if len(self._played) >= limit:
                return
            # Event.wait doubles as the sleep and the cancellation check, so a
            # turn that finishes mid-gap stops immediately instead of waiting
            # out the full interval.
            if self._stop.wait(gap):
                return

    def stop(self) -> None:
        """Stop scheduling and wait for the clip in flight to drain."""
        self._stop.set()
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=max(0, int(self._player.config.max_drain_ms)) / 1000)
        self._thread = None


def _default_play(path: Path) -> None:
    """Play a WAV via the speech module's platform player.

    Imported lazily: ``speech`` pulls in optional audio dependencies, and
    ``presence`` must stay importable (and testable) without them.
    """
    from azmo_mind.speech import _play_wav

    _play_wav(path)


def build_player(config: PresenceConfig, **kwargs) -> PresencePlayer:
    """Convenience constructor mirroring ``select_speech_adapter``."""
    return PresencePlayer(config, **kwargs)


# ---------------------------------------------------------------------------
# Rendering the pool
# ---------------------------------------------------------------------------

def clip_path(config: PresenceConfig, kind: str, index: int) -> Path:
    """Destination for a rendered clip. Stable naming so a rebuild replaces
    rather than accumulates."""
    return Path(config.clips_path) / kind / f"{kind}_{index:02d}.wav"


def render_texts(config: PresenceConfig, kind: str) -> list[str]:
    """The source utterances for a kind.

    XTTS speaks text, so a non-verbal is coaxed out of it with vowel-and-breath
    spellings rather than words. These are deliberately configurable: which
    spellings actually render as a convincing breath is an empirical question
    about the voice, not something to hard-code. Render, listen, keep the good
    ones, delete the rest.
    """
    if kind == "exhale":
        return list(config.exhale_texts)
    if kind == "growl":
        return list(config.growl_texts)
    return []


def shape_clip(path: str | Path, fade_in_ms: int = 20, fade_out_ms: int = 200) -> bool:
    """Fade the head and tail of a rendered clip. Returns True if it ran.

    Two artefacts come from asking a speech model for a non-word, and both live
    in the envelope rather than in the middle of the sound.

    **The tail rises.** XTTS voices a trailing vowel, and a voiced vowel carries
    pitch - often rising, because trailing punctuation reads as continuation. On
    a breath that lands as a chirp at the end. The presence shelf at 7 kHz then
    lifts it further, because breath is mostly high-frequency content and that
    boost was tuned for consonants in speech.

    **The head thumps.** The DSP's octave-down and sub-growl layers pitch-shift
    a sharp onset into a low warble - the "bouncy ball" before the breath.

    A short fade in and a longer fade out remove both without touching the part
    of the sound that is actually his. Deliberately asymmetric: a breath starts
    fairly abruptly and dies away slowly, so a long head fade would sound wrong
    while a long tail fade sounds natural.

    Degrades to a no-op if numpy/soundfile are missing, like everything else in
    this module - a clip that was not shaped still beats no clip.
    """
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return False

    file = Path(path)
    try:
        audio, rate = sf.read(str(file), dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001 - an unreadable clip must not break a build
        return False

    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    total = int(getattr(audio, "size", 0))
    if total == 0:
        return False

    envelope = np.ones(total, dtype="float32")
    head = min(int(rate * max(0, fade_in_ms) / 1000), total // 2)
    tail = min(int(rate * max(0, fade_out_ms) / 1000), total // 2)
    if head > 0:
        envelope[:head] = np.linspace(0.0, 1.0, head, dtype="float32")
    if tail > 0:
        # Squared curve: fades faster at first, then trails off. A linear fade on
        # a decaying breath sounds like someone turning a knob.
        envelope[-tail:] = np.linspace(1.0, 0.0, tail, dtype="float32") ** 2

    try:
        sf.write(str(file), (audio * envelope).astype("float32"), rate)
    except Exception:  # noqa: BLE001
        return False
    return True


def render_seed(base_seed: int, index: int, stride: int) -> int:
    """The seed for one presence clip.

    ``speech.clone_seed`` is fixed and non-zero on purpose: it is what makes a
    good spoken take stay good, and re-seeding before every chunk is why our
    splitting preserves his character where the model's own did not.

    Presence wants the opposite trade. The pool exists so that no single breath
    becomes a recognisable tic, and rendering every clip from one sampling roll
    works directly against that - you get twelve files with one personality.
    So each clip gets its own seed, derived from the base so a rebuild still
    reproduces the same pool exactly.

    ``stride`` of 0 keeps the old behaviour: every clip on the speech seed.
    """
    if stride <= 0 or base_seed <= 0:
        return base_seed
    return base_seed + index * stride


def time_ms() -> float:
    """Wall clock in ms. Indirected so tests can freeze it."""
    return time.perf_counter() * 1000
