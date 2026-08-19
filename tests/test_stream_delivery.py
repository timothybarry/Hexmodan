"""Overlapping synthesis with generation without letting him stutter.

The design log (2026-07-30) is explicit about the trade this code makes. Pure
overlap - play chunk 1 the instant it exists - is the fastest and the wrong
choice: if the renderer falls behind the playhead he stops mid-sentence, and a
gap inside a line reads as broken where a pause before it reads as deliberate.
Presence already covers the front of the turn, so the front is not where the
risk is.

So playback waits for a prebuffer, and every stall it could not avoid is
counted rather than swallowed. These tests pin both halves: that the prebuffer
is actually respected, and that a stall is still reported honestly when the
renderer loses the race anyway.

Nothing here loads torch, XTTS or an audio device.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from azmo_mind.config import load_config
from azmo_mind.engine import AzmoEngine
from azmo_mind.memory import MemoryStore
from azmo_mind.providers.base import (
    LLMProvider,
    ProviderError,
    ProviderResult,
    SpeechStream,
)
from azmo_mind.providers.mock import MockProvider
from azmo_mind.schemas import AzmoResponse, VoiceDirection
from azmo_mind.speech import SpeechError, StreamedSpeech
from azmo_mind.state import EmotionStateStore
from azmo_mind.streaming import SpeechFieldStreamer

# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

class FakeClone:
    """A clone adapter that writes a marker file instead of synthesising."""

    name = "fake-clone"

    def __init__(self, delay: float = 0.0, fail_on: int | None = None) -> None:
        self.delay = delay
        self.fail_on = fail_on
        self.rendered: list[str] = []
        self.anchors: list[object] = []

    def render_to_file(self, text, voice, out_path, dsp="__default__", anchor=None) -> bool:
        if self.delay:
            time.sleep(self.delay)
        if self.fail_on is not None and len(self.rendered) == self.fail_on:
            raise RuntimeError("synthesis exploded")
        Path(out_path).write_bytes(b"RIFF")
        self.rendered.append(text)
        self.anchors.append(anchor)
        return True


class Speaker:
    """Records playback order, and how long each clip took."""

    def __init__(self, duration: float = 0.0) -> None:
        self.duration = duration
        self.played: list[Path] = []
        self.existed: list[bool] = []

    def __call__(self, path: Path) -> None:
        self.existed.append(path.exists())
        self.played.append(path)
        if self.duration:
            time.sleep(self.duration)


class FragmentProvider(LLMProvider):
    """Streams a canned JSON document a few characters at a time."""

    def __init__(self, response: AzmoResponse, size: int = 5, fail_after: int | None = None):
        self.document = response.model_dump_json()
        self.size = size
        self.fail_after = fail_after

    def generate(self, messages):
        return ProviderResult(
            response=AzmoResponse.model_validate_json(self.document),
            raw_content=self.document,
            metrics={},
        )

    def generate_stream(self, messages) -> SpeechStream:
        def factory(stream: SpeechStream):
            decoder = SpeechFieldStreamer()
            for sent, index in enumerate(range(0, len(self.document), self.size)):
                if self.fail_after is not None and sent >= self.fail_after:
                    raise ProviderError("Ollama stopped answering mid-reply.")
                fragment = self.document[index:index + self.size]
                delta = decoder.feed(fragment)
                if delta:
                    yield delta
            stream.complete(self.generate(messages))

        return SpeechStream(factory)

    def warmup(self):
        return {"ok": True}

    def health(self):
        return {"ok": True}


def build_engine(tmp_path, provider) -> AzmoEngine:
    cfg = load_config("config/azmo.yaml")
    cfg.memory.database_path = tmp_path / "memory.sqlite3"
    cfg.runtime.log_path = tmp_path / "runtime.jsonl"
    return AzmoEngine(
        cfg,
        provider,
        memory=MemoryStore(cfg.memory.database_path),
        state_store=EmotionStateStore(tmp_path / "state.json"),
    )


def reply(speech: str) -> AzmoResponse:
    return AzmoResponse(
        speech=speech,
        emotion="commanding",
        gesture={"name": "loom", "intensity": 0.3, "duration_ms": 1200, "target": "speaker"},
        voice={"preset": "close_ominous", "pace": 0.9},
    )


CHUNKS = ["First clause of the decree.", "Second clause.", "Third and last."]


# ---------------------------------------------------------------------------
# StreamedSpeech
# ---------------------------------------------------------------------------

def test_chunks_are_spoken_in_order():
    adapter = FakeClone()
    speaker = Speaker()
    stream = StreamedSpeech(adapter, VoiceDirection(), prebuffer=2, play=speaker)
    stream.begin(iter(CHUNKS))
    stream.await_prebuffer(5)
    metrics = stream.play()
    assert adapter.rendered == CHUNKS
    assert metrics["chunks"] == 3
    assert metrics["spoken"] is True
    assert len(speaker.played) == 3


def test_playback_does_not_begin_before_the_prebuffer_is_full():
    """The guarantee the whole class exists for."""
    adapter = FakeClone(delay=0.05)
    speaker = Speaker()
    stream = StreamedSpeech(adapter, VoiceDirection(), prebuffer=3, play=speaker)
    stream.begin(iter(CHUNKS))
    ready = stream.await_prebuffer(5)
    assert ready >= 3
    assert speaker.played == []
    stream.play()


def test_a_reply_shorter_than_the_prebuffer_still_plays():
    """Waiting for a third chunk that will never exist would be a deadlock."""
    adapter = FakeClone()
    speaker = Speaker()
    stream = StreamedSpeech(adapter, VoiceDirection(), prebuffer=8, play=speaker)
    stream.begin(iter(["Only this."]))
    assert stream.await_prebuffer(5) == 1
    assert stream.play()["chunks"] == 1


def test_every_chunk_shares_one_gain_anchor():
    """Without this the reply pumps between chunks - see test_dsp_anchor."""
    adapter = FakeClone()
    stream = StreamedSpeech(adapter, VoiceDirection(), prebuffer=1, play=Speaker())
    stream.begin(iter(CHUNKS))
    stream.await_prebuffer(5)
    stream.play()
    assert len(adapter.anchors) == 3
    assert all(a is adapter.anchors[0] for a in adapter.anchors)
    assert adapter.anchors[0] is not None


def test_no_stall_is_reported_when_synthesis_keeps_ahead():
    adapter = FakeClone()
    stream = StreamedSpeech(adapter, VoiceDirection(), prebuffer=3, play=Speaker())
    stream.begin(iter(CHUNKS))
    stream.await_prebuffer(5)
    assert stream.play()["stalls"] == 0


def test_a_stall_is_counted_when_synthesis_falls_behind():
    """A stall must be visible. It is the failure this design trades against."""
    adapter = FakeClone(delay=0.15)
    stream = StreamedSpeech(adapter, VoiceDirection(), prebuffer=1, play=Speaker())
    stream.begin(iter(CHUNKS))
    stream.await_prebuffer(5)
    assert stream.play()["stalls"] > 0


def test_rendered_files_exist_when_played_and_are_gone_afterwards():
    adapter = FakeClone()
    speaker = Speaker()
    stream = StreamedSpeech(adapter, VoiceDirection(), prebuffer=1, play=speaker)
    stream.begin(iter(CHUNKS))
    stream.await_prebuffer(5)
    stream.play()
    assert all(speaker.existed)
    assert not any(p.exists() for p in speaker.played)


def test_a_failure_before_any_audio_raises():
    stream = StreamedSpeech(FakeClone(fail_on=0), VoiceDirection(), prebuffer=1, play=Speaker())
    stream.begin(iter(CHUNKS))
    stream.await_prebuffer(5)
    with pytest.raises(SpeechError):
        stream.play()


def test_a_failure_after_he_has_spoken_is_reported_not_raised():
    """He already said something. Cutting the turn short beats erroring over it."""
    speaker = Speaker()
    stream = StreamedSpeech(FakeClone(fail_on=2), VoiceDirection(), prebuffer=1, play=speaker)
    stream.begin(iter(CHUNKS))
    stream.await_prebuffer(5)
    metrics = stream.play()
    assert metrics["chunks"] == 2
    assert "exploded" in str(metrics["error"])


def test_close_removes_chunks_that_were_never_played():
    adapter = FakeClone()
    stream = StreamedSpeech(adapter, VoiceDirection(), prebuffer=3, play=Speaker())
    stream.begin(iter(CHUNKS))
    stream.await_prebuffer(5)
    pending = list(stream._ready)
    assert pending and all(p.exists() for p in pending)
    stream.close()
    assert not any(p.exists() for p in pending)


def test_a_stream_cannot_be_started_twice():
    stream = StreamedSpeech(FakeClone(), VoiceDirection(), prebuffer=1, play=Speaker())
    stream.begin(iter(["One."]))
    with pytest.raises(SpeechError):
        stream.begin(iter(["Two."]))
    stream.await_prebuffer(5)
    stream.play()


def test_playback_stays_on_the_calling_thread():
    """Half-duplex depends on play() being finished when it returns."""
    caller = threading.get_ident()
    threads: list[int] = []
    stream = StreamedSpeech(
        FakeClone(), VoiceDirection(), prebuffer=1,
        play=lambda path: threads.append(threading.get_ident()),
    )
    stream.begin(iter(CHUNKS))
    stream.await_prebuffer(5)
    stream.play()
    assert threads and all(t == caller for t in threads)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def test_streamed_speech_reconstructs_the_reply(tmp_path):
    text = (
        "You mistake patience for weakness. I have watched empires rot from "
        "the inside while their kings congratulated themselves. Speak plainly, "
        "or do not speak at all."
    )
    engine = build_engine(tmp_path, FragmentProvider(reply(text)))
    turn = engine.respond_stream("Explain yourself.")
    chunks = list(turn.chunks)
    assert len(chunks) > 1
    assert " ".join(chunks).split() == text.split()


def test_finish_applies_the_parts_that_cannot_stream(tmp_path):
    engine = build_engine(tmp_path, FragmentProvider(reply("The throne is not a chair.")))
    turn = engine.respond_stream("Sit down.")
    list(turn.chunks)
    result = turn.finish()
    assert result.response.gesture.name == "loom"
    assert result.metrics["streamed"] is True
    assert result.provider_error is None
    assert engine.config.runtime.log_path.exists()
    assert any("throne" in t["content"] for t in engine.memory.recent_turns(4))


def test_finish_before_the_chunks_are_drained_is_not_silently_wrong(tmp_path):
    """Calling finish() early must fall back, not invent a completed turn."""
    engine = build_engine(tmp_path, FragmentProvider(reply("Half a thought.")))
    turn = engine.respond_stream("Go.")
    result = turn.finish()
    assert result.provider_error is None or "did not" in result.provider_error
    assert result.response.speech


def test_a_failure_before_he_speaks_gets_the_diagnostic_line(tmp_path):
    engine = build_engine(tmp_path, FragmentProvider(reply("Never said."), fail_after=0))
    turn = engine.respond_stream("Speak.")
    chunks = list(turn.chunks)
    assert chunks
    assert "local mind failed" in " ".join(chunks)
    result = turn.finish()
    assert "stopped answering" in (result.provider_error or "")


def test_a_failure_mid_reply_keeps_what_he_said_and_appends_nothing(tmp_path):
    """Bolting a failure notice onto a half-delivered reply would be stranger
    than simply stopping, so the turn ends with his own words."""
    long_reply = (
        "The first legion moved at dusk. The second waited for the signal that "
        "never came. The third is still marching, and it does not know why."
    )
    engine = build_engine(tmp_path, FragmentProvider(long_reply and reply(long_reply), fail_after=30))
    turn = engine.respond_stream("Report.")
    chunks = list(turn.chunks)
    assert chunks
    spoken = " ".join(chunks)
    assert "local mind failed" not in spoken
    result = turn.finish()
    assert result.provider_error
    # Memory and the echo guard must match what actually left the speaker.
    assert result.response.speech.split() == spoken.split()


def test_structured_output_leaking_into_speech_is_never_read_aloud(tmp_path):
    """The guard that AzmoResponse validation normally provides, applied in
    time to matter: streaming speaks before validation runs."""
    leak = 'Kneel before me. "gesture": {"name": "loom", "intensity": 0.9}'
    engine = build_engine(tmp_path, FragmentProvider(reply(leak)))
    turn = engine.respond_stream("Bow.")
    spoken = " ".join(turn.chunks)
    assert "Kneel before me." in spoken
    assert "gesture" not in spoken
    assert "intensity" not in spoken


def test_a_non_streaming_provider_still_works_through_the_streamed_path(tmp_path):
    """MockProvider has no generate_stream, so it uses the base fallback: one
    late delta. Callers never have to branch on whether a provider streams."""
    engine = build_engine(tmp_path, MockProvider())
    turn = engine.respond_stream("Awaken.")
    chunks = list(turn.chunks)
    assert chunks
    result = turn.finish()
    assert result.response.gesture.name == "loom"
    assert result.metrics["streamed"] is True


def test_the_streamed_document_is_still_validated_as_one_turn(tmp_path):
    """Streaming changes when the words arrive, never what the turn is."""
    engine = build_engine(tmp_path, FragmentProvider(reply("Order restored.")))
    turn = engine.respond_stream("Status.")
    list(turn.chunks)
    result = turn.finish()
    assert json.loads(result.raw_model_output)["gesture"]["name"] == "loom"
