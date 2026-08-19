"""Decoding the ``speech`` field out of a JSON document as it is written.

AZMO uses Ollama structured output, so the model emits a JSON object rather
than prose. Streaming his words therefore means decoding one string field of a
document that is not finished yet - including when a network fragment splits in
the middle of an escape sequence.

The failure this guards is not a crash. It is AZMO speaking a stray backslash,
half a unicode escape, or a JSON field name aloud, which is exactly the class of
defect ``sanitize_speech`` exists to prevent on the non-streaming path.
"""

from __future__ import annotations

import json

from azmo_mind.schemas import AzmoResponse
from azmo_mind.streaming import SpeechFieldStreamer


def feed_all(fragments: list[str]) -> str:
    streamer = SpeechFieldStreamer()
    return "".join(streamer.feed(f) for f in fragments)


def test_speech_is_declared_first_in_the_schema():
    """Load-bearing: the field order decides when his words become available.

    Pydantic emits ``properties`` in declaration order and Ollama builds its
    grammar from that, so ``speech`` first is what lets the text reach XTTS
    before the gesture and voice metadata exist. Moving it would not break
    anything visibly - it would quietly restore the old serial latency.
    """
    fields = list(AzmoResponse.model_json_schema()["properties"])
    assert fields[0] == "speech"


def test_decodes_a_whole_document_in_one_fragment():
    doc = json.dumps({"speech": "Kneel.", "emotion": "commanding"})
    assert feed_all([doc]) == "Kneel."


def test_decodes_across_arbitrary_fragment_boundaries():
    doc = json.dumps({"speech": "You mistake patience for weakness.", "emotion": "contemptuous"})
    for size in (1, 2, 3, 5, 7, 13):
        fragments = [doc[i:i + size] for i in range(0, len(doc), size)]
        assert feed_all(fragments) == "You mistake patience for weakness."


def test_stops_at_the_end_of_the_speech_value():
    """Everything after the closing quote is structure, and must never be said."""
    doc = json.dumps({"speech": "Enough.", "gesture": {"name": "dismiss"}})
    streamer = SpeechFieldStreamer()
    streamer.feed(doc)
    assert streamer.complete is True
    assert streamer.text == "Enough."
    assert "dismiss" not in streamer.text
    # Anything fed afterwards is ignored rather than appended.
    assert streamer.feed('{"speech": "again"}') == ""


def test_escaped_quote_inside_the_speech_does_not_end_it():
    doc = json.dumps({"speech": 'He called it "mercy". I called it a mistake.', "emotion": "amused"})
    assert feed_all([doc]) == 'He called it "mercy". I called it a mistake.'


def test_fragment_ending_mid_escape_holds_the_backslash_back():
    """A fragment that ends on a lone backslash must emit nothing yet.

    Emitting it would put a literal backslash into his speech, and XTTS would
    render whatever it makes of that.
    """
    streamer = SpeechFieldStreamer()
    streamer.feed('{"speech": "before')
    assert streamer.feed("\\") == ""
    assert streamer.feed('n after"') == "\n after"


def test_fragment_split_inside_a_unicode_escape():
    doc = json.dumps({"speech": "Naïve."}, ensure_ascii=True)
    assert "\\u" in doc
    cut = doc.index("\\u") + 3
    assert feed_all([doc[:cut], doc[cut:]]) == "Naïve."


def test_all_json_escapes_decode():
    text = 'tab\there\nnewline "quoted" back\\slash'
    doc = json.dumps({"speech": text})
    assert feed_all([doc]) == text


def test_key_split_across_fragments_is_still_found():
    doc = json.dumps({"speech": "Found me."})
    cut = 4  # lands inside the `"speech"` key itself
    assert feed_all([doc[:cut], doc[cut:]]) == "Found me."


def test_leading_prose_or_a_fence_before_the_object_is_skipped():
    doc = '```json\n' + json.dumps({"speech": "Still here."}) + '\n```'
    assert feed_all([doc]) == "Still here."


def test_nothing_is_emitted_before_the_field_opens():
    streamer = SpeechFieldStreamer()
    assert streamer.feed('{"') == ""
    assert streamer.feed('spee') == ""
    assert streamer.feed('ch": ') == ""
    assert streamer.complete is False
    assert streamer.feed('"Now."') == "Now."


def test_incomplete_document_yields_what_arrived():
    """A stream cut off mid-sentence still gives us the words he had said."""
    streamer = SpeechFieldStreamer()
    got = streamer.feed('{"speech": "The campaign was already')
    assert got == "The campaign was already"
    assert streamer.complete is False
