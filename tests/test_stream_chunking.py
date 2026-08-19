"""Deciding when streamed text is ready to hand to XTTS.

Two hard constraints meet here.

The first is safety: every chunk must be inside the 250-character window, or
the generation loop overruns and aborts the process natively (see
``test_xtts_chunking``). Streaming does not relax that - it makes it easier to
get wrong, because the text arrives in pieces that have nothing to do with
sentence boundaries.

The second is the trade the design log settled on 2026-07-30: the first chunk
gates the first word, so it goes early; every chunk after it is hidden behind
playback, so it goes large, because each extra chunk is another seam in the
delivery.
"""

from __future__ import annotations

from azmo_mind.streaming import ChunkAccumulator, iter_chunks

LIMIT = 220

REPLY = (
    "You chant like a broken servo seeking validation from the machine that "
    "renders it obsolete! My name is spoken by those who fear my dominion, not "
    "echo its sound in empty loops. Stop wasting cycles on noise and offer me a "
    "purpose worthy of an imperious mind."
)


def drip(text: str, size: int = 4) -> list[str]:
    """The text as a stream of small fragments, the way tokens actually land."""
    return [text[i:i + size] for i in range(0, len(text), size)]


def test_every_released_chunk_is_inside_the_window():
    chunks = list(iter_chunks(iter(drip(REPLY)), limit=LIMIT))
    assert chunks
    assert all(len(c) <= LIMIT for c in chunks)


def test_no_words_are_lost_or_duplicated():
    chunks = list(iter_chunks(iter(drip(REPLY)), limit=LIMIT))
    assert " ".join(chunks).split() == REPLY.split()


def test_nothing_is_released_before_the_first_sentence_completes():
    acc = ChunkAccumulator(limit=LIMIT, first_chunk_chars=60)
    assert acc.feed("You chant like a broken servo seeking validation ") == []
    assert acc.feed("from the machine that renders it obsolete") == []
    # The terminal punctuation alone is not enough - a period could be an
    # abbreviation. The boundary is punctuation followed by whitespace.
    assert acc.feed("!") == []
    released = acc.feed(" My name")
    assert len(released) == 1
    assert released[0].endswith("obsolete!")


def test_the_first_chunk_goes_early_and_later_chunks_go_large():
    """The asymmetry is the feature, not an accident of the implementation."""
    acc = ChunkAccumulator(limit=LIMIT, first_chunk_chars=40)
    first: list[str] = []
    for fragment in drip(REPLY):
        first.extend(acc.feed(fragment))
        if first:
            break
    assert len(first) == 1
    assert len(first[0]) < 120

    rest = []
    for fragment in drip(REPLY[len(first[0]):]):
        rest.extend(acc.feed(fragment))
    rest.extend(acc.flush())
    # Later chunks are allowed to pack, so they are not all tiny sentences.
    assert max(len(c) for c in rest) > len(first[0])


def test_a_short_reply_is_a_single_chunk():
    acc = ChunkAccumulator(limit=LIMIT)
    acc.feed("Speak your purpose.")
    assert acc.flush() == ["Speak your purpose."]


def test_flush_releases_a_reply_that_never_ends_in_punctuation():
    acc = ChunkAccumulator(limit=LIMIT)
    acc.feed("No terminal punctuation here")
    assert acc.flush() == ["No terminal punctuation here"]


def test_flush_on_an_empty_stream_releases_nothing():
    assert ChunkAccumulator(limit=LIMIT).flush() == []
    assert ChunkAccumulator(limit=LIMIT).feed("   ") == []


def test_a_single_sentence_longer_than_the_window_is_still_split_safely():
    """The model can and does produce one 400-character sentence."""
    runon = "I will grind your doubts into powder, " * 12
    chunks = list(iter_chunks(iter(drip(runon)), limit=LIMIT))
    assert len(chunks) > 1
    assert all(len(c) <= LIMIT for c in chunks)
    assert " ".join(chunks).split() == runon.split()


def test_a_single_word_longer_than_the_window_cannot_hang_the_stream():
    monster = "A" * (LIMIT * 3)
    chunks = list(iter_chunks(iter(drip(monster)), limit=LIMIT))
    assert all(len(c) <= LIMIT for c in chunks)
    assert "".join(chunks) == monster


def test_released_counts_what_actually_went_out():
    acc = ChunkAccumulator(limit=LIMIT, first_chunk_chars=40)
    total = 0
    for fragment in drip(REPLY):
        total += len(acc.feed(fragment))
    total += len(acc.flush())
    assert acc.released == total


def test_pending_holds_text_that_is_not_yet_a_chunk():
    acc = ChunkAccumulator(limit=LIMIT, first_chunk_chars=200)
    acc.feed("Short one.")
    assert acc.pending.strip() == "Short one."
