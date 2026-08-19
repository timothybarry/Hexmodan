"""Regression tests for the anti-feedback (self-hearing) machinery.

The failure these guard against: AZMO speaks, the mic picks up his own voice,
Whisper transcribes his own name out of it, the wake word fires, and he answers
himself forever. Each test below pins one of the three independent guards.
"""

from __future__ import annotations

import queue

import pytest

from azmo_mind.config import ListenerConfig
from azmo_mind.listener import EchoGuard, MicStream, similarity, strip_wake


AZMO_LINE = (
    "I am Azmodan, Lord of Sin, and your hesitation is the only wall between "
    "you and the throne you pretend not to want."
)


# ---------------------------------------------------------------------------
# Guard 1: the capture gate
# ---------------------------------------------------------------------------

def _frames(stream: MicStream) -> int:
    return stream._queue.qsize()


def _feed(stream: MicStream, count: int = 3) -> None:
    """Simulate the sounddevice callback delivering frames."""
    for _ in range(count):
        if stream._open_gate.is_set():
            try:
                stream._queue.put_nowait(b"\x00\x00" * 480)
            except queue.Full:
                pass


def test_gate_starts_open():
    stream = MicStream(ListenerConfig())
    assert not stream.gated
    _feed(stream)
    assert _frames(stream) == 3


def test_pause_shuts_the_gate_and_clears_the_buffer():
    stream = MicStream(ListenerConfig())
    _feed(stream, 5)
    stream.pause()
    assert stream.gated
    assert _frames(stream) == 0
    # Audio arriving while AZMO speaks must be discarded, not queued.
    _feed(stream, 10)
    assert _frames(stream) == 0


def test_resume_reopens_the_gate_after_draining():
    stream = MicStream(ListenerConfig())
    stream.pause()
    _feed(stream, 4)
    stream.resume(cooldown_s=0)
    assert not stream.gated
    assert _frames(stream) == 0
    _feed(stream, 2)
    assert _frames(stream) == 2


def test_queue_is_bounded_so_a_long_pause_cannot_grow_without_limit():
    config = ListenerConfig(max_utterance_ms=3000)
    stream = MicStream(config)
    limit = stream._queue.maxsize
    assert limit > 0
    for _ in range(limit + 200):
        try:
            stream._queue.put_nowait(b"\x00\x00" * 480)
        except queue.Full:
            break
    assert _frames(stream) <= limit


def test_next_utterance_returns_none_instead_of_blocking_forever():
    stream = MicStream(ListenerConfig())
    # No audio is ever delivered; the read must give up, not hang the loop.
    assert stream.next_utterance(timeout_s=0.05) is None


def test_close_unblocks_readers():
    stream = MicStream(ListenerConfig())
    stream.close()          # sets the stop event even with no live device
    assert stream.next_utterance(timeout_s=5) is None


# ---------------------------------------------------------------------------
# Guard 1b: the deaf window, and the second sound source (azmo-presence)
#
# Presence added a second thing that makes noise. Everything the gate does for
# his speech it must also do for his breath - and the wake-word breath is the
# awkward one, because it plays during the exact moment the user has been
# invited to speak.
# ---------------------------------------------------------------------------

class _FakeMic:
    """Records the gate calls a deaf window makes."""

    def __init__(self) -> None:
        self.paused = 0
        self.cooldowns: list[float] = []

    def pause(self) -> None:
        self.paused += 1

    def resume(self, cooldown_s: float = 0.0) -> None:
        self.cooldowns.append(cooldown_s)


def _listener_with_fake_mic(config: ListenerConfig):
    from azmo_mind.listener import Listener

    listener = Listener.__new__(Listener)
    listener.config = config
    listener.mic = _FakeMic()
    listener.echo = EchoGuard(window_s=config.echo_guard_window_ms / 1000)
    return listener


def test_deaf_shuts_the_gate_and_reopens_after_the_default_cooldown():
    config = ListenerConfig(post_speech_cooldown_ms=700)
    listener = _listener_with_fake_mic(config)
    with listener.deaf():
        assert listener.mic.paused == 1
        assert listener.mic.cooldowns == []      # still shut inside the block
    assert listener.mic.cooldowns == [0.7]


def test_deaf_accepts_a_shorter_cooldown_for_a_quieter_sound():
    """The wake-word breath must not hold the gate shut for a full reply's tail.

    700 ms is sized for a loud complete reply. Using it after a short quiet
    breath keeps the mic shut while the user is being invited to speak, which
    eats the front of their command - a worse failure than the dead air the
    breath was added to fix.
    """
    config = ListenerConfig(post_speech_cooldown_ms=700)
    listener = _listener_with_fake_mic(config)
    with listener.deaf(cooldown_ms=200):
        pass
    assert listener.mic.cooldowns == [0.2]


def test_deaf_cooldown_override_of_zero_is_honoured_not_treated_as_unset():
    config = ListenerConfig(post_speech_cooldown_ms=700)
    listener = _listener_with_fake_mic(config)
    with listener.deaf(cooldown_ms=0):
        pass
    assert listener.mic.cooldowns == [0.0]


def test_deaf_reopens_the_gate_even_when_the_block_raises():
    """A crash mid-turn must not leave him permanently deaf."""
    config = ListenerConfig()
    listener = _listener_with_fake_mic(config)
    with pytest.raises(RuntimeError):
        with listener.deaf():
            raise RuntimeError("synthesis blew up")
    assert listener.mic.cooldowns, "gate was never reopened"


def test_deaf_arms_the_echo_guard_with_what_he_said():
    config = ListenerConfig()
    listener = _listener_with_fake_mic(config)
    with listener.deaf(spoken=AZMO_LINE):
        pass
    assert listener.echo.is_echo(AZMO_LINE)


def test_presence_wake_cooldown_default_is_shorter_than_the_speech_cooldown():
    """Pins the relationship, not the numbers: a breath's tail is shorter than
    a full reply's, and the default config must reflect that."""
    from azmo_mind.config import PresenceConfig

    assert PresenceConfig().wake_ack_cooldown_ms < ListenerConfig().post_speech_cooldown_ms


# ---------------------------------------------------------------------------
# Guard 3: echo suppression
# ---------------------------------------------------------------------------

def test_similarity_is_one_for_a_verbatim_echo():
    assert similarity(AZMO_LINE, AZMO_LINE) == pytest.approx(1.0)


def test_similarity_ignores_stopword_only_overlap():
    # Shares only "the"/"you"/"and" with AZMO's line: not an echo.
    assert similarity("and the you", AZMO_LINE) == 0.0


def test_echo_guard_rejects_azmo_hearing_himself():
    guard = EchoGuard(window_s=8.0)
    guard.remember(AZMO_LINE)
    # A partial pickup of his own reply, which contains his own wake word.
    assert guard.is_echo("I am Azmodan Lord of Sin and your hesitation")


def test_a_bare_wake_word_is_never_treated_as_an_echo():
    # AZMO says his own name in almost every reply, so the wake word carries no
    # evidence. If it counted toward the echo score, summoning him in the eight
    # seconds after a reply would be silently swallowed - he would appear to
    # ignore every second command. The capture gate and cooldown are the guards
    # for a genuine bare-name reflection, not this one.
    guard = EchoGuard(window_s=8.0, wake_word="Azmodan")
    guard.remember(AZMO_LINE)
    assert not guard.is_echo("Azmodan")
    assert not guard.is_echo("Azmodan.")


def test_his_own_name_does_not_inflate_the_echo_score():
    guard = EchoGuard(window_s=8.0, wake_word="Azmodan")
    guard.remember(AZMO_LINE)
    # Real command: only "Azmodan" overlaps, and that one is excluded.
    assert not guard.is_echo("Azmodan, set a timer for ten minutes")


def test_echo_guard_allows_a_real_command_that_shares_a_few_words():
    guard = EchoGuard(window_s=8.0)
    guard.remember(AZMO_LINE)
    assert not guard.is_echo("Azmodan, what is the weather in Denver tomorrow?")


def test_echo_guard_expires_so_the_user_may_quote_him_later():
    guard = EchoGuard(window_s=8.0)
    guard.remember(AZMO_LINE, now=0.0)
    assert guard.is_echo("I am Azmodan Lord of Sin", now=1.0)
    assert not guard.is_echo("I am Azmodan Lord of Sin", now=100.0)


def test_echo_guard_is_inert_before_azmo_has_spoken():
    guard = EchoGuard()
    assert not guard.is_echo("Azmodan, introduce yourself")


def test_echo_guard_clear_disarms_it():
    guard = EchoGuard()
    guard.remember(AZMO_LINE)
    guard.clear()
    assert not guard.is_echo(AZMO_LINE)


# ---------------------------------------------------------------------------
# The interaction that actually caused the loop
# ---------------------------------------------------------------------------

def test_azmo_self_introduction_would_wake_him_but_the_echo_guard_stops_it():
    # strip_wake finds his name in his own sentence - which is exactly why the
    # echo guard has to exist. Both halves of this are the point.
    assert strip_wake(AZMO_LINE, "Azmodan") is not None
    guard = EchoGuard()
    guard.remember(AZMO_LINE)
    assert guard.is_echo(AZMO_LINE)
