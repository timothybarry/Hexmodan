from azmo_mind.listener import WhisperWake, listener_available, strip_wake


def test_wake_word_with_command_in_one_breath():
    assert strip_wake("Azmodan, what should I do today?", "Azmodan") == "what should I do today?"


def test_wake_word_only_returns_empty_string():
    assert strip_wake("Azmodan.", "Azmodan") == ""


def test_no_wake_word_returns_none():
    assert strip_wake("What time is it?", "Azmodan") is None


def test_tolerant_to_whisper_misspellings():
    # Whisper spells the unusual name loosely — these should still wake.
    for heard in ("Asmodan, rise.", "As modan rise.", "Azmodon, rise."):
        assert strip_wake(heard, "Azmodan") == "rise."


def test_wake_match_is_case_insensitive_and_trims_punctuation():
    assert strip_wake("  azmodan:  speak now  ", "Azmodan") == "speak now"


def test_whisper_wake_detector_wraps_strip():
    wake = WhisperWake(wake_word="Azmodan")
    assert wake.command_from("Azmodan, report.") == "report."
    assert wake.command_from("nothing relevant") is None


def test_always_on_treats_all_speech_as_command():
    wake = WhisperWake(wake_word="Azmodan", always_on=True)
    assert wake.command_from("just tell me the time") == "just tell me the time"


def test_listener_available_is_bool():
    assert isinstance(listener_available(), bool)
