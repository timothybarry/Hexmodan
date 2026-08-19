"""Tuning the wake word against transcripts Whisper actually produced.

``listener.py`` decides whether you addressed AZMO using two numbers and a list:
``wake_fuzzy_threshold``, the phonetic key, and ``extra_wake_variants``. Until
now those were set by hand and judged by impression - lower the threshold until
he wakes up, raise it when he interrupts you. That is guesswork on a system
where both failure modes are annoying and neither is rare.

This module makes it a measurement. Collect the transcripts Whisper genuinely
emits, label each one as "I was talking to him" or "I was not", and the correct
threshold is no longer a matter of taste.

**The objective is asymmetric, deliberately.** A missed wake costs you one
repetition. A false wake means AZMO interrupts a conversation he was not part
of, and - because he says his own name constantly - risks the self-trigger loop
that ``EchoGuard`` and the capture gate exist to prevent. So the tuner does not
maximise accuracy or F1. It finds the setting that wakes him as often as
possible **subject to zero false wakes**, and breaks ties toward the more
conservative threshold, because the negatives you did not record are the ones
that will bite.

Pure and dependency-free: no audio, no models. The whole tuner is testable on a
machine with no microphone.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from azmo_mind.listener import phonetic_key, squash, strip_wake

DEFAULT_DATASET = Path("data/wake_samples.jsonl")

# The sweep range. Below ~0.55 the ratio matches almost any short word; at 1.0
# the phonetic pass is disabled entirely and only exact spellings count.
DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(
    round(0.55 + i * 0.01, 2) for i in range(46)
)


@dataclass(frozen=True)
class Sample:
    """One transcript, labelled with whether it was meant for AZMO.

    ``wake=True`` means *you were addressing him* - the detector must fire.
    ``wake=False`` means ordinary speech - the detector must stay silent.
    """

    text: str
    wake: bool
    note: str = ""

    def to_json(self) -> dict[str, object]:
        record: dict[str, object] = {"text": self.text, "wake": self.wake}
        if self.note:
            record["note"] = self.note
        return record


@dataclass(frozen=True)
class Score:
    """How one configuration performed against the whole dataset."""

    threshold: float
    true_wakes: int
    false_wakes: int
    missed: int
    correct_silence: int
    missed_texts: tuple[str, ...] = ()
    false_texts: tuple[str, ...] = ()

    @property
    def positives(self) -> int:
        return self.true_wakes + self.missed

    @property
    def negatives(self) -> int:
        return self.false_wakes + self.correct_silence

    @property
    def recall(self) -> float:
        """Fraction of genuine summons that woke him."""
        return self.true_wakes / self.positives if self.positives else 0.0

    @property
    def clean(self) -> bool:
        """True when nothing ordinary was mistaken for the wake word."""
        return self.false_wakes == 0


def evaluate(
    samples: Sequence[Sample],
    wake_word: str,
    threshold: float,
    extra_variants: Sequence[str] | None = None,
) -> Score:
    """Run the real detector over the dataset at one setting.

    This calls ``listener.strip_wake`` rather than reimplementing the matching,
    so the tuner can never drift away from what the listener actually does -
    the failure that would make every number here a comfortable lie.
    """
    true_wakes = false_wakes = missed = correct_silence = 0
    missed_texts: list[str] = []
    false_texts: list[str] = []

    for sample in samples:
        fired = strip_wake(
            sample.text,
            wake_word,
            fuzzy_threshold=threshold,
            extra_variants=list(extra_variants or []),
        ) is not None
        if sample.wake and fired:
            true_wakes += 1
        elif sample.wake:
            missed += 1
            missed_texts.append(sample.text)
        elif fired:
            false_wakes += 1
            false_texts.append(sample.text)
        else:
            correct_silence += 1

    return Score(
        threshold=threshold,
        true_wakes=true_wakes,
        false_wakes=false_wakes,
        missed=missed,
        correct_silence=correct_silence,
        missed_texts=tuple(missed_texts),
        false_texts=tuple(false_texts),
    )


def sweep(
    samples: Sequence[Sample],
    wake_word: str,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    extra_variants: Sequence[str] | None = None,
) -> list[Score]:
    """Score every threshold, lowest first."""
    return [
        evaluate(samples, wake_word, threshold, extra_variants)
        for threshold in sorted(thresholds)
    ]


@dataclass(frozen=True)
class Recommendation:
    """The tuner's answer, plus the evidence for it."""

    threshold: float
    score: Score
    suggested_variants: tuple[str, ...]
    rejected_for_false_wakes: tuple[float, ...]
    reason: str


def recommend(
    samples: Sequence[Sample],
    wake_word: str,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    extra_variants: Sequence[str] | None = None,
) -> Recommendation:
    """Pick the best threshold: most wakes, zero false wakes, ties to the safer end.

    Ties break **upward** on purpose. Two thresholds that score identically on
    the recorded data are not equally safe on data you have not recorded: the
    higher one demands a closer match, so it has more margin against the next
    ordinary sentence that happens to sound a little like the name.
    """
    scores = sweep(samples, wake_word, thresholds, extra_variants)
    clean = [s for s in scores if s.clean]
    rejected = tuple(s.threshold for s in scores if not s.clean)

    if not clean:
        # Every setting false-wakes. The threshold is not the problem - either a
        # negative sample really does sound like the name, or a variant in the
        # exact-match list is too greedy.
        best = max(scores, key=lambda s: (-s.false_wakes, s.true_wakes, s.threshold))
        return Recommendation(
            threshold=best.threshold,
            score=best,
            suggested_variants=(),
            rejected_for_false_wakes=rejected,
            reason=(
                "No threshold avoids false wakes. Something in the negative "
                "samples matches by spelling, not by sound - check "
                "extra_wake_variants for an entry that is ordinary English."
            ),
        )

    best_recall = max(s.true_wakes for s in clean)
    # Among the settings that tie on recall, take the highest threshold.
    best = max(
        (s for s in clean if s.true_wakes == best_recall),
        key=lambda s: s.threshold,
    )
    return Recommendation(
        threshold=best.threshold,
        score=best,
        suggested_variants=tuple(mine_variants(samples, wake_word, best.threshold)),
        rejected_for_false_wakes=rejected,
        reason=(
            f"Highest threshold that still wakes {best.true_wakes}/"
            f"{best.positives} summons with zero false wakes."
        ),
    )


def mine_variants(
    samples: Sequence[Sample],
    wake_word: str,
    threshold: float,
    max_span: int = 3,
    max_offset: int = 2,
) -> list[str]:
    """Propose ``extra_wake_variants`` entries for summons that still miss.

    For each missed wake, find the opening span that came closest to the name
    and offer it verbatim. These are *suggestions*, never applied automatically:
    the exact-variant list is matched anywhere in a sentence, so an entry that is
    ordinary English ("as modern") would false-wake mid-conversation. That
    judgement stays with a human, which is exactly why ``_wake_variants``
    excludes real phrases by hand.
    """
    target = squash(wake_word)
    target_key = phonetic_key(target)
    proposals: list[str] = []
    seen: set[str] = set()

    for sample in samples:
        if not sample.wake:
            continue
        if strip_wake(sample.text, wake_word, fuzzy_threshold=threshold) is not None:
            continue
        words = sample.text.split()
        best_span, best_ratio = "", 0.0
        for offset in range(min(max_offset, len(words)) + 1):
            for span in range(1, max_span + 1):
                end = offset + span
                if end > len(words):
                    break
                phrase = " ".join(words[offset:end])
                candidate = squash(phrase)
                if not candidate:
                    continue
                ratio = SequenceMatcher(None, candidate, target).ratio()
                if phonetic_key(candidate) == target_key:
                    ratio = max(ratio, 0.99)
                if ratio > best_ratio:
                    best_span, best_ratio = phrase.lower().strip(" ,.;:!?-"), ratio
        if best_span and best_span not in seen:
            seen.add(best_span)
            proposals.append(best_span)
    return proposals


# ---------------------------------------------------------------------------
# Dataset persistence
# ---------------------------------------------------------------------------

def load_samples(path: str | Path = DEFAULT_DATASET) -> list[Sample]:
    """Read a JSONL dataset. A missing file is an empty dataset, not an error."""
    file = Path(path)
    if not file.exists():
        return []
    samples: list[Sample] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        samples.append(
            Sample(
                text=text,
                wake=bool(record.get("wake", False)),
                note=str(record.get("note", "")),
            )
        )
    return samples


def append_samples(
    new: Iterable[Sample], path: str | Path = DEFAULT_DATASET
) -> int:
    """Append samples, skipping ones already present. Returns how many were added.

    De-duplication is on the exact transcript: recording the same sentence twice
    would silently weight it double and bias the tuner toward it.
    """
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    existing = {s.text for s in load_samples(file)}
    added = 0
    with file.open("a", encoding="utf-8") as handle:
        for sample in new:
            if sample.text in existing:
                continue
            existing.add(sample.text)
            handle.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")
            added += 1
    return added


# Transcripts already earned in live sessions and pinned in the test suite. They
# seed a usable dataset before you record anything, so `azmo wake tune` says
# something meaningful on day one.
SEED_WAKES: tuple[str, ...] = (
    "As Madam, introduce yourself.",
    "Asmodan, introduce yourself.",
    "Az modern, introduce yourself.",
    "As Modan, introduce yourself.",
    "Azmodon, introduce yourself.",
    "Azmodan, introduce yourself.",
    "As been in, introduce yourself.",
    "Hey Azmodan, report.",
    "Okay As Madam, report.",
    "As Madam.",
)

SEED_NEGATIVES: tuple[str, ...] = (
    "What time is it?",
    "I need to introduce myself to the team tomorrow.",
    "Can you turn the lights down a little.",
    "The weather looks good this afternoon.",
    "Let me know when dinner is ready.",
    "I was reading about a place called as madam last night",
    "Is my order ready yet?",
    "That is a lot of money for a power supply.",
    "The cat knocked the mug off the table again.",
    "Add milk to the shopping list.",
    "Has my mother called back?",
    "As soon as you can, please.",
    "Ask Adam if he is coming.",
    "The alarm is set for seven.",
)


def seed_samples() -> list[Sample]:
    """The known corpus as labelled samples."""
    return [
        *[Sample(text=t, wake=True, note="seed") for t in SEED_WAKES],
        *[Sample(text=t, wake=False, note="seed") for t in SEED_NEGATIVES],
    ]
