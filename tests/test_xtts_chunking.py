"""XTTS text chunking.

XTTS v2 generates English inside a 250-character window. Handing it more with
``enable_text_splitting=False`` does not truncate politely - the generation loop
overruns its positional limit and, on Windows, aborts the process natively
(exit code -1073740791 / 0xC0000409). AZMO's replies exceed 250 characters
routinely, so every chunk we hand the model must be provably under the limit.
"""

from __future__ import annotations

import pytest

from azmo_mind.speech import XTTS_CHARACTER_LIMIT, split_for_xtts

LIMIT = 220

# The exact reply that aborted the process in a live session (253 characters).
CRASHING_REPLY = (
    "You chant like a broken servo seeking validation from the machine that "
    "renders it obsolete! My name is spoken by those who fear my dominion, not "
    "echo its sound in empty loops. Stop wasting cycles on noise and offer me a "
    "purpose worthy of an imperious mind."
)


def test_the_reply_that_crashed_really_was_over_the_limit():
    # Guards the premise of this whole module.
    assert len(CRASHING_REPLY) > XTTS_CHARACTER_LIMIT


def test_the_crashing_reply_is_now_split_safely():
    chunks = split_for_xtts(CRASHING_REPLY, LIMIT)
    assert len(chunks) > 1
    assert all(len(c) <= LIMIT for c in chunks)


def test_short_text_is_left_as_a_single_pass():
    text = "Speak your purpose."
    assert split_for_xtts(text, LIMIT) == [text]


def test_no_words_are_lost_or_reordered():
    chunks = split_for_xtts(CRASHING_REPLY, LIMIT)
    assert " ".join(chunks).split() == CRASHING_REPLY.split()


def test_chunks_break_on_sentence_boundaries_where_possible():
    text = ("First sentence here. " * 20).strip()
    for chunk in split_for_xtts(text, LIMIT):
        assert chunk.endswith(".")


def test_a_single_overlong_sentence_breaks_on_clauses():
    text = ", ".join(["a fairly long clause of text here"] * 12) + "."
    chunks = split_for_xtts(text, LIMIT)
    assert all(len(c) <= LIMIT for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_a_run_on_with_no_punctuation_still_gets_wrapped():
    text = "word " * 200
    chunks = split_for_xtts(text, LIMIT)
    assert chunks
    assert all(len(c) <= LIMIT for c in chunks)
    # Hard wrap must never split a word in half.
    assert all(w == "word" for c in chunks for w in c.split())


def test_a_single_word_longer_than_the_limit_is_not_infinite_looped():
    text = "x" * (LIMIT * 3)
    chunks = split_for_xtts(text, LIMIT)
    assert len(chunks) == 1          # cannot split a word; emit it whole
    assert chunks[0] == text


def test_empty_and_whitespace_produce_no_chunks():
    assert split_for_xtts("", LIMIT) == []
    assert split_for_xtts("    \n  ", LIMIT) == []


def test_whitespace_is_normalised():
    assert split_for_xtts("  two   spaces\nhere  ", LIMIT) == ["two spaces here"]


@pytest.mark.parametrize("limit", [40, 80, 120, 180, 220, 230])
def test_every_chunk_respects_the_limit_at_any_setting(limit):
    for text in (CRASHING_REPLY, "word " * 300, "Short. " * 60):
        chunks = split_for_xtts(text, limit)
        oversized = [c for c in chunks if len(c) > limit and " " in c]
        assert not oversized, f"limit={limit} produced {oversized}"


def test_adjacent_short_sentences_are_packed_to_minimise_passes():
    # Each pass is another seam in the delivery, so do not split more than needed.
    text = "One. Two. Three. Four. Five."
    assert split_for_xtts(text, LIMIT) == [text]
