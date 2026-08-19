"""Tuning the wake word from data instead of from impressions.

The threshold and the variant list were previously set by feel. This suite pins
the thing that makes tuning trustworthy: the tuner scores the **real** detector,
and its objective treats a false wake as far worse than a missed one.

That asymmetry is not a preference. A missed wake costs one repetition. A false
wake means AZMO answers a conversation he was not part of - and since he says
his own name constantly, that is the first step of the self-trigger loop the
whole half-duplex design exists to prevent.
"""

from __future__ import annotations

from azmo_mind.listener import strip_wake
from azmo_mind.waketrain import (
    Sample,
    append_samples,
    evaluate,
    load_samples,
    mine_variants,
    recommend,
    seed_samples,
    sweep,
)

WAKE = "Azmodan"


def test_evaluate_counts_every_outcome():
    samples = [
        Sample("Azmodan, report.", True),
        Sample("As Madam, report.", True),
        Sample("What time is it?", False),
        Sample("The weather is fine.", False),
    ]
    score = evaluate(samples, WAKE, 0.72)
    assert score.true_wakes == 2
    assert score.false_wakes == 0
    assert score.missed == 0
    assert score.correct_silence == 2
    assert score.recall == 1.0
    assert score.clean is True


def test_evaluate_uses_the_real_detector():
    """The tuner must never reimplement matching, or its numbers become fiction."""
    text = "Az modern, introduce yourself."
    fired = strip_wake(text, WAKE, fuzzy_threshold=0.72) is not None
    score = evaluate([Sample(text, True)], WAKE, 0.72)
    assert (score.true_wakes == 1) is fired


def test_a_missed_summons_is_recorded_with_its_text():
    samples = [Sample("Completely unrelated words here.", True)]
    score = evaluate(samples, WAKE, 0.72)
    assert score.missed == 1
    assert score.missed_texts == ("Completely unrelated words here.",)


def test_a_false_wake_is_recorded_with_its_text():
    samples = [Sample("Azmodan is not being addressed here.", False)]
    score = evaluate(samples, WAKE, 0.72)
    assert score.false_wakes == 1
    assert "Azmodan" in score.false_texts[0]


def test_lowering_the_threshold_never_reduces_wakes():
    """Monotonicity: a looser threshold can only match more, never less."""
    samples = seed_samples()
    scores = sweep(samples, WAKE, thresholds=[0.60, 0.70, 0.80, 0.90])
    wakes = [s.true_wakes for s in scores]
    assert wakes == sorted(wakes, reverse=True)


def test_recommendation_never_accepts_a_false_wake():
    """The whole objective in one assertion."""
    rec = recommend(seed_samples(), WAKE)
    assert rec.score.false_wakes == 0
    assert rec.score.clean is True


def test_recommendation_prefers_the_safer_threshold_on_a_tie():
    """Two settings equal on recorded data are not equal on unrecorded data."""
    samples = [
        Sample("Azmodan, report.", True),
        Sample("What time is it?", False),
    ]
    # Exact spelling matches at every threshold, so every setting ties.
    rec = recommend(samples, WAKE, thresholds=[0.60, 0.70, 0.80])
    assert rec.threshold == 0.80


def test_recommendation_reports_what_it_rejected():
    samples = [
        Sample("Azmodan, report.", True),
        Sample("Ask Adam if he is coming.", False),
    ]
    rec = recommend(samples, WAKE, thresholds=[0.40, 0.72, 0.95])
    # Whatever it picked, anything that false-woke must be listed as rejected.
    for threshold in rec.rejected_for_false_wakes:
        assert evaluate(samples, WAKE, threshold).false_wakes > 0


def test_seed_data_produces_a_usable_recommendation():
    """Day-one value: a real answer before you record anything yourself."""
    rec = recommend(seed_samples(), WAKE)
    assert 0.55 <= rec.threshold <= 1.0
    assert rec.score.true_wakes > 0
    assert rec.reason


def test_when_nothing_is_clean_it_says_so_instead_of_pretending():
    samples = [
        Sample("Azmodan, report.", True),
        # A negative that contains the name verbatim: no threshold can fix this,
        # because exact-spelling matching runs before the phonetic pass.
        Sample("I was telling him about Azmodan yesterday.", False),
    ]
    rec = recommend(samples, WAKE, thresholds=[0.60, 0.80, 1.0])
    assert rec.score.false_wakes > 0
    assert "extra_wake_variants" in rec.reason


def test_mine_variants_proposes_the_span_that_missed():
    samples = [Sample("Hazmat on, introduce yourself.", True)]
    proposals = mine_variants(samples, WAKE, threshold=0.72)
    assert proposals
    assert any("hazmat" in p for p in proposals)


def test_mine_variants_ignores_summons_that_already_work():
    samples = [Sample("Azmodan, report.", True), Sample("As Madam, report.", True)]
    assert mine_variants(samples, WAKE, threshold=0.72) == []


def test_mine_variants_ignores_negatives():
    samples = [Sample("Something entirely different.", False)]
    assert mine_variants(samples, WAKE, threshold=0.72) == []


# ---------------------------------------------------------------------------
# Dataset persistence
# ---------------------------------------------------------------------------

def test_dataset_roundtrip(tmp_path):
    path = tmp_path / "wake.jsonl"
    added = append_samples(
        [Sample("As Madam, report.", True, note="live"), Sample("Hello there.", False)],
        path,
    )
    assert added == 2
    loaded = load_samples(path)
    assert [s.text for s in loaded] == ["As Madam, report.", "Hello there."]
    assert loaded[0].wake is True
    assert loaded[0].note == "live"
    assert loaded[1].wake is False


def test_appending_the_same_transcript_twice_does_not_double_weight_it(tmp_path):
    path = tmp_path / "wake.jsonl"
    append_samples([Sample("As Madam, report.", True)], path)
    added = append_samples([Sample("As Madam, report.", True)], path)
    assert added == 0
    assert len(load_samples(path)) == 1


def test_a_missing_dataset_is_empty_not_an_error(tmp_path):
    assert load_samples(tmp_path / "nope.jsonl") == []


def test_corrupt_lines_are_skipped_rather_than_fatal(tmp_path):
    path = tmp_path / "wake.jsonl"
    path.write_text(
        '{"text": "As Madam.", "wake": true}\n'
        "not json at all\n"
        '{"text": "", "wake": true}\n'
        '{"text": "Hello.", "wake": false}\n',
        encoding="utf-8",
    )
    loaded = load_samples(path)
    assert [s.text for s in loaded] == ["As Madam.", "Hello."]
