"""Incremental text handling for streamed replies (0.2.10).

Two pure, dependency-free pieces sit between "Ollama is emitting tokens" and
"XTTS has something safe to render":

``SpeechFieldStreamer``
    Pulls the ``speech`` value out of a JSON object *while it is still being
    written*. AZMO's provider uses Ollama structured output, so the model emits
    a JSON document, not prose - there is no way to stream his words without
    decoding that document incrementally.

``ChunkAccumulator``
    Decides *when* a piece of that text is safe and worthwhile to hand to XTTS.
    Safety is the 250-character window (see ``speech.split_for_xtts``);
    worthwhile is the trade between speaking sooner and adding another seam.

Both are deliberately free of torch, audio and network imports so the whole
streaming decision layer is unit-testable on any machine, including one with no
GPU and no model installed.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from azmo_mind.speech import split_for_xtts

# JSON's two-character escapes. \u is handled separately because it needs a
# four-digit lookahead that may not have arrived yet.
_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}

_HEX = "0123456789abcdefABCDEF"


class SpeechFieldStreamer:
    """Decode one top-level JSON string field from a stream of fragments.

    Ollama is asked for structured output, so the model's first tokens are
    ``{"speech": "`` rather than words. Waiting for the closing brace before
    speaking would throw away the entire point of streaming, so we decode the
    one field we care about as it arrives and ignore the rest of the document.

    **This depends on ``speech`` being the first field in the schema.** Pydantic
    emits ``properties`` in declaration order and Ollama builds its grammar in
    that order, so ``AzmoResponse.speech`` being declared first is what makes the
    text arrive before the gesture and voice metadata. It is not an accident and
    it is pinned by a test - reordering ``AzmoResponse`` would silently move the
    speech to the end of the stream and quietly restore the old serial latency
    without breaking anything visibly.

    The decoder is resilient to fragments splitting anywhere, including in the
    middle of a ``\\uXXXX`` escape: incomplete escapes are held back rather than
    emitted as literal backslashes.
    """

    __slots__ = ("_buffer", "_field", "_key_pattern", "_out", "_state")

    _SEEKING = "seeking"
    _INSIDE = "inside"
    _DONE = "done"

    def __init__(self, field: str = "speech") -> None:
        self._field = field
        self._buffer = ""
        self._state = self._SEEKING
        self._out: list[str] = []
        # `"speech"` followed by optional whitespace, a colon, more optional
        # whitespace, and the opening quote of the value.
        self._key_pattern = re.compile(r'"' + re.escape(field) + r'"\s*:\s*"')

    @property
    def complete(self) -> bool:
        """True once the closing quote of the value has been seen."""
        return self._state == self._DONE

    @property
    def text(self) -> str:
        """Everything decoded so far."""
        return "".join(self._out)

    def feed(self, fragment: str) -> str:
        """Consume a fragment; return only the characters newly decoded.

        Returning the delta rather than the whole value keeps the caller from
        having to diff, and makes the chunk accumulator downstream a pure
        append-only consumer.
        """
        if self._state == self._DONE or not fragment:
            return ""
        self._buffer += fragment
        produced: list[str] = []

        if self._state == self._SEEKING:
            match = self._key_pattern.search(self._buffer)
            if match is None:
                # Keep only enough tail to match a key split across fragments.
                keep = len(self._field) + 8
                if len(self._buffer) > keep:
                    self._buffer = self._buffer[-keep:]
                return ""
            self._buffer = self._buffer[match.end():]
            self._state = self._INSIDE

        if self._state == self._INSIDE:
            produced.append(self._decode())

        decoded = "".join(produced)
        if decoded:
            self._out.append(decoded)
        return decoded

    def _decode(self) -> str:
        """Decode as much of the buffered string body as is unambiguous."""
        out: list[str] = []
        index = 0
        buffer = self._buffer
        size = len(buffer)
        while index < size:
            char = buffer[index]
            if char == '"':
                self._state = self._DONE
                index += 1
                break
            if char != "\\":
                out.append(char)
                index += 1
                continue
            # An escape. Everything below may be incomplete, in which case we
            # stop and wait rather than emitting a stray backslash.
            if index + 1 >= size:
                break
            marker = buffer[index + 1]
            if marker in _SIMPLE_ESCAPES:
                out.append(_SIMPLE_ESCAPES[marker])
                index += 2
                continue
            if marker == "u":
                if index + 6 > size:
                    break
                digits = buffer[index + 2:index + 6]
                if all(d in _HEX for d in digits):
                    out.append(chr(int(digits, 16)))
                    index += 6
                    continue
                # Malformed: pass it through rather than stalling forever.
                out.append(buffer[index:index + 2])
                index += 2
                continue
            # Unknown escape - keep the escaped character, drop the backslash.
            out.append(marker)
            index += 2

        self._buffer = buffer[index:]
        return "".join(out)


# Sentence end: terminal punctuation, any closing quotes/brackets, then space.
_SENTENCE_END = re.compile(r'[.!?]["\'’”)\]]*(\s)')
# Clause end: the fallback when one sentence outruns the window.
_CLAUSE_END = re.compile(r'[,;:]["\'’”)\]]*(\s)')


class ChunkAccumulator:
    """Buffer streamed text and release XTTS-safe chunks as they become ready.

    Every released chunk is guaranteed to be within ``limit`` characters, so the
    250-character window that has aborted the process three times is respected
    by construction rather than by remembering to check.

    The release policy trades two things against each other:

    - **Speak sooner.** The first chunk gates the first word, so it is released
      as early as a complete sentence allows (``first_chunk_chars``).
    - **Fewer seams.** Every chunk is a separate XTTS pass with a 120 ms breath
      after it. Once he is already speaking, latency is hidden by playback, so
      later chunks pack up to the full ``limit`` the way ``split_for_xtts``
      does for a whole reply.

    That asymmetry is the point: the opening chunk buys the entrance, and
    everything after it buys smoothness, because by then the clock has stopped
    mattering.
    """

    def __init__(
        self,
        limit: int = 220,
        first_chunk_chars: int = 60,
    ) -> None:
        first_chunk_chars = min(first_chunk_chars, limit)
        self._limit = max(1, int(limit))
        self._first = max(1, int(first_chunk_chars))
        self._buffer = ""
        self._released = 0

    @property
    def pending(self) -> str:
        """Text held back because it is not yet a releasable chunk."""
        return self._buffer

    @property
    def released(self) -> int:
        return self._released

    def feed(self, text: str) -> list[str]:
        """Add streamed text; return any chunks that are now ready."""
        if text:
            self._buffer += text
        return self._drain(final=False)

    def flush(self) -> list[str]:
        """Release everything that remains, ignoring the size targets."""
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[str]:
        chunks: list[str] = []
        while True:
            cut = self._cut(final)
            if cut is None:
                break
            piece = self._buffer[:cut]
            self._buffer = self._buffer[cut:]
            for chunk in split_for_xtts(piece, self._limit):
                chunks.append(chunk)
                self._released += 1
            if not self._buffer.strip():
                self._buffer = ""
                break
        return chunks

    def _cut(self, final: bool) -> int | None:
        """Index to cut the buffer at, or None if nothing is ready yet."""
        buffer = self._buffer
        if not buffer.strip():
            return None
        if final:
            return len(buffer)

        target = self._first if self._released == 0 else self._limit
        window = buffer[: self._limit]

        best = self._last_boundary(_SENTENCE_END, window)
        if best is not None and best >= target:
            return best
        # A sentence boundary exists but the piece is still short: only take it
        # if waiting is pointless because the buffer already fills the window.
        if best is not None and len(buffer) > self._limit:
            return best
        if len(buffer) <= self._limit:
            return None

        # No usable sentence break inside the window and the buffer has outrun
        # it - fall back the same way split_for_xtts does.
        clause = self._last_boundary(_CLAUSE_END, window)
        if clause is not None:
            return clause
        space = window.rstrip().rfind(" ")
        if space > 0:
            return space + 1
        return self._limit

    @staticmethod
    def _last_boundary(pattern: re.Pattern[str], window: str) -> int | None:
        end: int | None = None
        for match in pattern.finditer(window):
            end = match.end()
        return end


def iter_chunks(
    deltas: Iterator[str],
    limit: int = 220,
    first_chunk_chars: int = 60,
) -> Iterator[str]:
    """Convenience: turn a stream of text deltas into a stream of chunks."""
    accumulator = ChunkAccumulator(limit=limit, first_chunk_chars=first_chunk_chars)
    for delta in deltas:
        yield from accumulator.feed(delta)
    yield from accumulator.flush()
