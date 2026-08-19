"""Wake-word matching against what Whisper actually produces.

"Azmodan" is not in Whisper's vocabulary, so it renders the sound as ordinary
English words. Every string in REAL_MISHEARINGS was observed in a live session -
an exact list of spellings can never be complete, which is why matching is
phonetic.
"""

from __future__ import annotations

import pytest

from azmo_mind.listener import WhisperWake, phonetic_key, strip_wake


# Transcripts seen in the wild for someone saying "Azmodan, introduce yourself."
REAL_MISHEARINGS = [
    "As Madam, introduce yourself.",
    "Asmodan, introduce yourself.",
    "Az modern, introduce yourself.",
    "As Modan, introduce yourself.",
    "Azmodon, introduce yourself.",
]


@pytest.mark.parametrize("heard", REAL_MISHEARINGS)
def test_real_mishearings_still_wake_him(heard):
    assert strip_wake(heard, "Azmodan") == "introduce yourself."


def test_phonetic_key_collapses_the_observed_mishearing():
    # This is why the phonetic pass works: vowels carry no information when the
    # model is guessing, and the consonant skeleton survives.
    assert phonetic_key("asmadam") == phonetic_key("azmodan")


def test_phonetic_key_keeps_unrelated_words_apart():
    assert phonetic_key("introduce") != phonetic_key("azmodan")
    assert phonetic_key("yourself") != phonetic_key("azmodan")
    assert phonetic_key("tomorrow") != phonetic_key("azmodan")


def test_bare_mishearing_alone_is_a_wake_with_no_command():
    assert strip_wake("As Madam.", "Azmodan") == ""


def test_wake_word_may_follow_a_filler_word():
    assert strip_wake("Hey Azmodan, report.", "Azmodan") == "report."
    assert strip_wake("Okay As Madam, report.", "Azmodan") == "report."


def test_ordinary_speech_does_not_false_wake():
    for sentence in (
        "What time is it?",
        "I need to introduce myself to the team tomorrow.",
        "Can you turn the lights down a little.",
        "The weather looks good this afternoon.",
        "Let me know when dinner is ready.",
    ):
        assert strip_wake(sentence, "Azmodan") is None, sentence


def test_fuzzy_matching_only_applies_near_the_start():
    # A late fuzzy match would fire on ordinary sentences constantly.
    assert strip_wake(
        "I was reading about a place called as madam last night", "Azmodan"
    ) is None


def test_fuzzy_can_be_disabled_with_a_threshold_of_one():
    assert strip_wake("As Madam, report.", "Azmodan", fuzzy_threshold=1.0) is None
    # The exact-variant list still works with fuzzy off.
    assert strip_wake("Azmodan, report.", "Azmodan", fuzzy_threshold=1.0) == "report."


def test_extra_variants_are_a_user_escape_hatch():
    assert strip_wake("Hazmat on, report.", "Azmodan") is None
    assert strip_wake(
        "Hazmat on, report.", "Azmodan", extra_variants=["hazmat on"]
    ) == "report."


def test_whisper_wake_passes_config_through():
    wake = WhisperWake(
        wake_word="Azmodan", fuzzy_threshold=0.72, extra_variants=["hazmat on"]
    )
    assert wake.command_from("As Madam, report.") == "report."
    assert wake.command_from("Hazmat on, report.") == "report."
    assert wake.command_from("what time is it") is None


def test_always_on_bypasses_matching_entirely():
    wake = WhisperWake(wake_word="Azmodan", always_on=True)
    assert wake.command_from("just tell me the time") == "just tell me the time"
