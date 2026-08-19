from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from azmo_mind.config import AppConfig
from azmo_mind.memory import MemoryStore
from azmo_mind.motion_link import MotionLink, MotionResult, SimulatedMotionLink
from azmo_mind.prompts import build_system_prompt
from azmo_mind.providers.base import LLMProvider, ProviderError
from azmo_mind.safety import arbitrate
from azmo_mind.schemas import AzmoResponse, EmotionState, sanitize_speech
from azmo_mind.state import EmotionStateStore, update_state
from azmo_mind.streaming import ChunkAccumulator


@dataclass(frozen=True)
class TurnResult:
    response: AzmoResponse
    state: EmotionState
    raw_model_output: str
    metrics: dict[str, Any] = field(default_factory=dict)
    provider_error: str | None = None
    motion: MotionResult | None = None


@dataclass(frozen=True)
class StreamingTurn:
    """A turn split across time: speech first, everything else at the end.

    ``chunks`` yields XTTS-safe pieces of his reply as the model writes them.
    ``finish`` is called once they are exhausted and does the part of a turn
    that cannot be streamed - safety arbitration, motion, memory, emotional
    state and the log.

    ``finish`` is deliberately *not* run on whatever thread consumed the chunks.
    Those writes touch SQLite and the state file, and a renderer thread is the
    wrong place for them; the caller runs it when the turn is genuinely over.
    """

    chunks: Iterator[str]
    finish: Callable[[], TurnResult]


class AzmoEngine:
    def __init__(
        self,
        config: AppConfig,
        provider: LLMProvider,
        memory: MemoryStore | None = None,
        state_store: EmotionStateStore | None = None,
        motion_link: MotionLink | None = None,
    ):
        self.config = config
        self.provider = provider
        self.memory = memory or MemoryStore(config.memory.database_path)
        self.state_store = state_store or EmotionStateStore()
        # The engine owns performance intent; the motion link owns the path to
        # the motor controller. Only a simulator exists until roadmap 0.6/0.7.
        self.motion_link = motion_link or SimulatedMotionLink(
            hardware_enabled=config.motion.hardware_enabled
        )
        self.config.runtime.log_path.parent.mkdir(parents=True, exist_ok=True)

    def warmup(self) -> dict[str, Any]:
        return self.provider.warmup()

    # -- turn assembly ------------------------------------------------------
    def _prepare(self, user_text: str) -> tuple[str, EmotionState, list[dict[str, str]]]:
        """Clean the input and build the prompt. Shared by both turn paths."""
        cleaned = " ".join(user_text.strip().split())
        if not cleaned:
            raise ValueError("Input cannot be empty.")

        previous_state = self.state_store.load()
        state = update_state(previous_state, cleaned)
        relevant_memories = self.memory.retrieve(
            cleaned,
            limit=self.config.memory.retrieved_memories,
        )

        system_prompt = build_system_prompt(self.config, state, relevant_memories)
        recent = self.memory.recent_turns(self.config.memory.recent_turns)
        messages = [{"role": "system", "content": system_prompt}, *recent]
        messages.append({"role": "user", "content": cleaned})
        return cleaned, state, messages

    @staticmethod
    def _fallback_response(speech: str | None = None) -> AzmoResponse:
        """What he says when the local mind did not finish the turn."""
        return AzmoResponse(
            speech=speech or (
                "The vessel is responsive, but the local mind failed to complete this turn. "
                "No motion will follow. Read the diagnostic beneath my words; even Hell records "
                "its failed campaigns."
            ),
            emotion="irritated",
            emotional_intensity=0.25,
            gesture={
                "name": "none",
                "intensity": 0,
                "duration_ms": 600,
                "target": "none",
            },
            voice={"preset": "calm_dark", "pace": 0.9},
            internal_note="Provider failure fallback.",
        )

    def _commit(
        self,
        cleaned: str,
        state: EmotionState,
        response: AzmoResponse,
        raw: str,
        metrics: dict[str, Any],
        provider_error: str | None,
    ) -> TurnResult:
        """Everything a turn does once the words are settled."""
        response = arbitrate(response, cleaned, self.config.motion)
        motion = self.motion_link.send_gesture(response.gesture)

        self.memory.add_turn("user", cleaned)
        self.memory.add_turn("assistant", response.speech)
        self.state_store.save(state)
        self._log(cleaned, response, state, raw, metrics, provider_error, motion)

        return TurnResult(
            response=response,
            state=state,
            raw_model_output=raw,
            metrics=metrics,
            provider_error=provider_error,
            motion=motion,
        )

    def respond(self, user_text: str) -> TurnResult:
        cleaned, state, messages = self._prepare(user_text)

        provider_error: str | None = None
        metrics: dict[str, Any] = {}
        try:
            generated = self.provider.generate(messages)
            response = generated.response
            raw = generated.raw_content
            metrics = generated.metrics
        except ProviderError as exc:
            provider_error = str(exc)
            response = self._fallback_response()
            raw = response.model_dump_json()

        return self._commit(cleaned, state, response, raw, metrics, provider_error)

    def respond_stream(self, user_text: str) -> StreamingTurn:
        """Begin a turn whose speech is delivered while it is still being written.

        Only the ``speech`` field streams. Gesture, voice direction and emotion
        arrive at the end of the document and are applied by ``finish`` - which
        is correct rather than merely convenient: a gesture is a whole command,
        and there is nothing useful to do with the first half of one.
        """
        cleaned, state, messages = self._prepare(user_text)
        stream = self.provider.generate_stream(messages)
        accumulator = ChunkAccumulator(
            limit=self.config.speech.clone_max_chars,
            first_chunk_chars=self.config.speech.stream_first_chunk_chars,
        )
        spoken: list[str] = []
        failure: dict[str, str | None] = {"error": None}

        def emit(chunk: str) -> str | None:
            """Guard a chunk on its way to the speaker.

            ``sanitize_speech`` normally runs inside ``AzmoResponse`` validation,
            which streaming reaches only *after* the words have been said. A
            model that crams its whole structured response into the speech
            string would therefore be read aloud, field names and all - the
            exact failure ``sanitize_speech`` was written to prevent.

            So the same guard runs per chunk here. If a chunk is found to be
            leaking, we speak the recovered prefix and stop: everything after a
            leak marker is JSON, and none of it is his voice.
            """
            clean = sanitize_speech(chunk)
            if clean != chunk:
                failure["error"] = failure["error"] or "structured output leaked into speech"
                return clean.strip() or None
            return chunk

        def chunks() -> Iterator[str]:
            leaked = False
            try:
                for delta in stream:
                    for chunk in accumulator.feed(delta):
                        guarded = emit(chunk)
                        if guarded:
                            spoken.append(guarded)
                            yield guarded
                        if guarded != chunk:
                            leaked = True
                            break
                    if leaked:
                        break
                if not leaked:
                    for chunk in accumulator.flush():
                        guarded = emit(chunk)
                        if guarded:
                            spoken.append(guarded)
                            yield guarded
                        if guarded != chunk:
                            break
            except ProviderError as exc:
                failure["error"] = str(exc)
                # A failure *before* he says anything gets the standard
                # diagnostic line. A failure part-way through does not: he has
                # already spoken, and appending a failure notice to a half-
                # delivered reply would be stranger than simply stopping. The
                # error is still recorded and shown.
                if not spoken:
                    excuse = ChunkAccumulator(limit=self.config.speech.clone_max_chars)
                    line = self._fallback_response().speech
                    # feed() releases as it goes and flush() only returns the
                    # remainder, so both halves are needed or the diagnostic
                    # loses its opening sentences.
                    for chunk in [*excuse.feed(line), *excuse.flush()]:
                        spoken.append(chunk)
                        yield chunk

        def finish() -> TurnResult:
            error = failure["error"]
            if stream.finished:
                result = stream.result
                response = result.response
                raw = result.raw_content
                metrics = dict(result.metrics)
            else:
                # The document never validated. Keep exactly what was said, so
                # memory and the echo guard match what actually left the speaker.
                said = " ".join(" ".join(spoken).split())
                response = self._fallback_response(said or None)
                raw = response.model_dump_json()
                metrics = {}
            metrics["streamed"] = True
            metrics["stream_chunks"] = len(spoken)
            return self._commit(cleaned, state, response, raw, metrics, error)

        return StreamingTurn(chunks=chunks(), finish=finish)

    def _log(
        self,
        user_text: str,
        response: AzmoResponse,
        state: EmotionState,
        raw: str,
        metrics: dict[str, Any],
        provider_error: str | None,
        motion: MotionResult | None = None,
    ) -> None:
        record: dict[str, object] = {
            "user": user_text,
            "response": response.model_dump(),
            "state": state.model_dump(),
            "metrics": metrics,
            "provider_error": provider_error,
        }
        if motion is not None:
            record["motion"] = {
                "command_id": motion.command.id,
                "state": motion.state.value,
                "lifecycle": [s.value for s in motion.lifecycle],
                "reason": motion.reason,
            }
        if self.config.runtime.save_raw_model_output:
            record["raw_model_output"] = raw

        with self.config.runtime.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
