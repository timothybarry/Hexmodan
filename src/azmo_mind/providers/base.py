from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from azmo_mind.schemas import AzmoResponse


class ProviderError(RuntimeError):
    """A recoverable local-model provider failure."""


@dataclass(frozen=True)
class ProviderResult:
    response: AzmoResponse
    raw_content: str
    metrics: dict[str, Any] = field(default_factory=dict)


class SpeechStream:
    """His words as they arrive, plus the validated turn once they stop.

    A turn is two things at different times. The *speech* is wanted as early as
    possible, because it gates the first sound. Everything else - gesture, voice
    direction, emotion, the metrics - is only wanted once, at the end, and is
    not worth streaming: nothing downstream can act on half a gesture.

    So this yields text deltas while the model writes, and exposes ``result``
    afterwards. Reading ``result`` before the stream is exhausted is a bug, and
    raises rather than returning something half-built.
    """

    def __init__(self, factory: Callable[[SpeechStream], Iterator[str]]) -> None:
        self._deltas = factory(self)
        self._result: ProviderResult | None = None

    def __iter__(self) -> Iterator[str]:
        return self._deltas

    def complete(self, result: ProviderResult) -> None:
        """Called by the producer once the whole document has validated."""
        self._result = result

    @property
    def finished(self) -> bool:
        return self._result is not None

    @property
    def result(self) -> ProviderResult:
        if self._result is None:
            raise ProviderError(
                "The reply stream ended without producing a complete turn."
            )
        return self._result


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> ProviderResult:
        raise NotImplementedError

    def generate_stream(self, messages: list[dict[str, str]]) -> SpeechStream:
        """Stream the speech text, then the full turn.

        The default implementation is honest rather than fake: it runs the
        blocking ``generate`` and emits the finished speech as a single delta.
        Callers therefore never need to branch on whether a provider streams -
        they get the same shape either way, and a non-streaming provider simply
        produces its one delta late. Overridden by ``OllamaProvider``.
        """

        def factory(stream: SpeechStream) -> Iterator[str]:
            result = self.generate(messages)
            yield result.response.speech
            stream.complete(result)

        return SpeechStream(factory)

    @abstractmethod
    def warmup(self) -> dict[str, Any]:
        """Load the model and prove that a minimal inference completes."""
        raise NotImplementedError

    @abstractmethod
    def health(self) -> dict[str, object]:
        raise NotImplementedError
