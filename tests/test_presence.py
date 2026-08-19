"""Tests for azmo-presence.

The interesting properties are behavioural, not structural: does he keep
breathing through a long think, does he stop when the reply lands, and does he
ever talk over his own breath. Playback is injected so nothing here touches an
audio device.
"""

from __future__ import annotations

import random
import threading
import time
from pathlib import Path

import pytest

from azmo_mind.config import PresenceConfig
from azmo_mind.presence import KINDS, PresencePlayer, PresenceTrack, clip_path, render_texts


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_pool(root: Path, exhale: int = 3, growl: int = 3) -> None:
    """Create a pool of empty WAV files. Contents never matter - playback is
    always injected in these tests."""
    for kind, count in (("exhale", exhale), ("growl", growl)):
        directory = root / kind
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(1, count + 1):
            (directory / f"{kind}_{index:02d}.wav").write_bytes(b"RIFF")


class Recorder:
    """Injected player that records what was played and how long it took."""

    def __init__(self, duration: float = 0.0) -> None:
        self.duration = duration
        self.played: list[Path] = []
        self.active = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def __call__(self, path: Path) -> None:
        with self._lock:
            self.active += 1
            self.max_concurrent = max(self.max_concurrent, self.active)
            self.played.append(path)
        if self.duration:
            time.sleep(self.duration)
        with self._lock:
            self.active -= 1


def config_for(root: Path, **overrides) -> PresenceConfig:
    base = {"clips_path": root, "sustain_gap_ms": 300, "max_drain_ms": 3000}
    base.update(overrides)
    return PresenceConfig(**base)


# ---------------------------------------------------------------------------
# pool discovery
# ---------------------------------------------------------------------------

def test_empty_pool_is_unavailable_not_an_error(tmp_path):
    player = PresencePlayer(config_for(tmp_path / "missing"))
    assert player.available() is False
    assert player.clips() == []
    # The whole point: no pool means silence, never an exception into the loop.
    assert player.play() is None


def test_available_once_clips_exist(tmp_path):
    make_pool(tmp_path)
    player = PresencePlayer(config_for(tmp_path))
    assert player.available() is True
    assert len(player.clips()) == 6
    assert player.describe() == {"exhale": 3, "growl": 3}


def test_disabled_config_plays_nothing(tmp_path):
    make_pool(tmp_path)
    recorder = Recorder()
    player = PresencePlayer(config_for(tmp_path, enabled=False), play=recorder)
    assert player.available() is False
    assert player.play() is None
    assert recorder.played == []


def test_zero_weight_excludes_a_kind(tmp_path):
    make_pool(tmp_path)
    player = PresencePlayer(
        config_for(tmp_path, weights={"exhale": 1.0, "growl": 0.0}),
        rng=random.Random(7),
    )
    assert player.enabled_kinds() == ("exhale",)
    picks = {player.pick().parent.name for _ in range(30)}
    assert picks == {"exhale"}


# ---------------------------------------------------------------------------
# selection - the "don't become a tic" property
# ---------------------------------------------------------------------------

def test_pick_avoids_immediate_repeats(tmp_path):
    make_pool(tmp_path, exhale=4, growl=4)
    player = PresencePlayer(config_for(tmp_path, avoid_repeat_window=3), rng=random.Random(1))
    picks = [player.pick() for _ in range(25)]
    # Nothing repeats inside the window - that is what stops one breath from
    # becoming a recognisable tic across a session.
    for index in range(1, len(picks)):
        window = picks[max(0, index - 3):index]
        assert picks[index] not in window


def test_pick_still_returns_something_when_pool_is_smaller_than_window(tmp_path):
    # One clip, window of 5. Repeating a breath beats going silent.
    make_pool(tmp_path, exhale=1, growl=0)
    player = PresencePlayer(
        config_for(tmp_path, avoid_repeat_window=5, weights={"exhale": 1.0, "growl": 0.0})
    )
    assert all(player.pick() is not None for _ in range(10))


def test_window_never_forces_a_deterministic_cycle(tmp_path):
    """A window as large as the pool leaves one legal choice, and the sequence
    stops being random - it becomes a fixed rotation, which is exactly the
    recognisable tic the window exists to prevent. The window is clamped so at
    least two clips always remain in play.
    """
    make_pool(tmp_path, exhale=3, growl=0)
    player = PresencePlayer(
        config_for(tmp_path, avoid_repeat_window=10, weights={"exhale": 1.0, "growl": 0.0}),
        rng=random.Random(3),
    )
    sequence = [player.pick().name for _ in range(60)]
    # A forced rotation would repeat with period 3. Real randomness will not.
    rotations = sum(1 for i in range(3, len(sequence)) if sequence[i] == sequence[i - 3])
    assert rotations < len(sequence) - 3, "selection degenerated into a fixed cycle"


def test_default_config_pool_size_does_not_degenerate(tmp_path):
    """The shipped defaults (4 source texts per kind, window 3) must still leave
    room to choose. This is the configuration people will actually run."""
    default = PresenceConfig()
    make_pool(tmp_path, exhale=len(default.exhale_texts), growl=len(default.growl_texts))
    player = PresencePlayer(
        config_for(tmp_path, avoid_repeat_window=default.avoid_repeat_window),
        rng=random.Random(11),
    )
    sequence = [player.pick().name for _ in range(80)]
    assert len(set(sequence)) > 2
    period = len(default.exhale_texts)
    rotations = sum(
        1 for i in range(period, len(sequence)) if sequence[i] == sequence[i - period]
    )
    assert rotations < len(sequence) - period


def test_pick_by_kind_is_honoured(tmp_path):
    make_pool(tmp_path)
    player = PresencePlayer(config_for(tmp_path))
    assert player.pick("growl").parent.name == "growl"
    assert player.pick("exhale").parent.name == "exhale"


def test_pool_order_is_stable_so_a_seeded_rng_is_reproducible(tmp_path):
    make_pool(tmp_path)
    first = PresencePlayer(config_for(tmp_path), rng=random.Random(99))
    second = PresencePlayer(config_for(tmp_path), rng=random.Random(99))
    assert [first.pick() for _ in range(8)] == [second.pick() for _ in range(8)]


# ---------------------------------------------------------------------------
# the contemplation track - the property the whole module exists for
# ---------------------------------------------------------------------------

def test_thinking_keeps_breathing_through_a_long_turn(tmp_path):
    """A sound at t=0 does nothing for someone still waiting at t=9."""
    make_pool(tmp_path)
    recorder = Recorder()
    player = PresencePlayer(
        config_for(tmp_path, sustain_gap_ms=300, max_sustain_clips=10), play=recorder
    )
    with player.thinking():
        time.sleep(1.1)
    # ~t=0, 0.3, 0.6, 0.9 - allow slack for scheduling jitter.
    assert len(recorder.played) >= 3


def test_thinking_plays_immediately_not_after_the_first_gap(tmp_path):
    make_pool(tmp_path)
    recorder = Recorder()
    player = PresencePlayer(config_for(tmp_path, sustain_gap_ms=5000), play=recorder)
    started = time.perf_counter()
    with player.thinking():
        time.sleep(0.25)
    elapsed = time.perf_counter() - started
    assert recorder.played, "first breath must not wait for the sustain gap"
    assert elapsed < 2.0, "exiting must not wait out the full gap"


def test_thinking_stops_promptly_when_the_reply_lands(tmp_path):
    """A turn that finishes mid-gap must stop now, not at the end of the gap."""
    make_pool(tmp_path)
    player = PresencePlayer(config_for(tmp_path, sustain_gap_ms=8000), play=Recorder())
    started = time.perf_counter()
    with player.thinking():
        time.sleep(0.05)
    assert time.perf_counter() - started < 1.0


def test_thinking_respects_the_sustain_cap(tmp_path):
    make_pool(tmp_path)
    recorder = Recorder()
    player = PresencePlayer(
        config_for(tmp_path, sustain_gap_ms=300, max_sustain_clips=3), play=recorder
    )
    with player.thinking():
        time.sleep(1.6)
    # A wedged turn becomes silence, not an endless growl.
    assert len(recorder.played) == 3


def test_he_never_speaks_over_his_own_breath(tmp_path):
    """Exiting the block must wait for the clip in flight to drain.

    Overlapping his breath with his words reads as a glitch, which is the exact
    failure presence exists to remove.
    """
    make_pool(tmp_path)
    recorder = Recorder(duration=0.4)
    player = PresencePlayer(
        config_for(tmp_path, sustain_gap_ms=5000, max_drain_ms=3000), play=recorder
    )
    with player.thinking():
        time.sleep(0.05)  # reply lands while the breath is still sounding
    assert recorder.active == 0, "returned while a clip was still playing"


def test_drain_is_bounded_so_a_stuck_player_cannot_hang_the_turn(tmp_path):
    make_pool(tmp_path)
    player = PresencePlayer(
        config_for(tmp_path, sustain_gap_ms=5000, max_drain_ms=200),
        play=Recorder(duration=5.0),
    )
    started = time.perf_counter()
    with player.thinking():
        time.sleep(0.05)
    # Bounded by max_drain_ms, not by the wedged clip.
    assert time.perf_counter() - started < 2.0


def test_clips_never_overlap_each_other(tmp_path):
    make_pool(tmp_path)
    recorder = Recorder(duration=0.15)
    player = PresencePlayer(
        config_for(tmp_path, sustain_gap_ms=300, max_sustain_clips=6), play=recorder
    )
    with player.thinking():
        time.sleep(1.2)
    assert recorder.max_concurrent == 1


def test_thinking_is_a_silent_noop_when_the_pool_is_empty(tmp_path):
    recorder = Recorder()
    player = PresencePlayer(config_for(tmp_path / "nothing"), play=recorder)
    with player.thinking() as track:
        time.sleep(0.05)
    assert track.played == []
    assert recorder.played == []


def test_thinking_is_a_noop_when_on_think_is_off(tmp_path):
    make_pool(tmp_path)
    recorder = Recorder()
    player = PresencePlayer(config_for(tmp_path, on_think=False), play=recorder)
    assert player.available() is True          # still usable for the wake beat
    assert player.thinking_available() is False
    with player.thinking() as track:
        time.sleep(0.05)
    assert track.played == []


def test_track_reports_what_was_actually_heard(tmp_path):
    make_pool(tmp_path)
    player = PresencePlayer(
        config_for(tmp_path, sustain_gap_ms=300, max_sustain_clips=3), play=Recorder()
    )
    with player.thinking() as track:
        time.sleep(1.2)
    assert len(track.played) == 3
    assert all(isinstance(p, Path) for p in track.played)


# ---------------------------------------------------------------------------
# failure is never fatal
# ---------------------------------------------------------------------------

def test_a_broken_player_degrades_to_silence(tmp_path):
    """No audio backend, an unreadable WAV, a locked file - a turn that produces
    a real reply with no breath in front of it is still a good turn."""
    make_pool(tmp_path)

    def explode(path: Path) -> None:
        raise OSError("no audio device")

    player = PresencePlayer(config_for(tmp_path), play=explode)
    assert player.play() is None
    with player.thinking() as track:
        time.sleep(0.2)
    assert track.played == []


def test_a_broken_player_does_not_spin(tmp_path):
    make_pool(tmp_path)
    calls = []

    def explode(path: Path) -> None:
        calls.append(path)
        raise OSError("no audio device")

    player = PresencePlayer(config_for(tmp_path, sustain_gap_ms=300), play=explode)
    with player.thinking():
        time.sleep(0.9)
    # Stops after the first failure rather than hammering a broken device.
    assert len(calls) == 1


def test_stop_is_idempotent(tmp_path):
    make_pool(tmp_path)
    player = PresencePlayer(config_for(tmp_path), play=Recorder())
    track = PresenceTrack(player)
    track.start()
    track.stop()
    track.stop()


# ---------------------------------------------------------------------------
# render helpers
# ---------------------------------------------------------------------------

def test_clip_paths_are_stable_so_a_rebuild_replaces_rather_than_accumulates(tmp_path):
    config = config_for(tmp_path)
    first = clip_path(config, "exhale", 1)
    assert first == clip_path(config, "exhale", 1)
    assert first.name == "exhale_01.wav"
    assert first.parent.name == "exhale"


def test_render_texts_cover_every_kind(tmp_path):
    config = config_for(tmp_path)
    for kind in KINDS:
        assert render_texts(config, kind), f"no source utterances for {kind}"
    assert render_texts(config, "not-a-kind") == []


@pytest.mark.parametrize("kind", KINDS)
def test_default_config_weights_every_kind(kind):
    assert PresenceConfig().weights.get(kind, 0) > 0
