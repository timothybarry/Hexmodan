"""Personality and gesture regression cases.

Evaluation drives **real turns** through a real engine — that is the point, it
measures the thing that actually ships. The consequence is that it also
*writes*: five cases append ten exchanges to the conversation history and push
the persistent emotional state five decay steps from wherever it was.

So a run used to leave AZMO measurably different afterwards, and salted the
memory of whatever conversation you were actually having with lines you never
said. Worse for the eval itself — run it twice and the second run reads the
first run's turns back as recent context, so the cases stop being independent of
each other and of run order.

Runs are therefore sandboxed: a throwaway memory database, state file and log,
discarded when the run ends. Explicit memories are copied in, because retrieval
is part of what the cases exercise and testing it against an empty store would
measure the wrong system.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml

from azmo_mind.config import AppConfig
from azmo_mind.engine import AzmoEngine
from azmo_mind.memory import MemoryStore
from azmo_mind.paths import resolve
from azmo_mind.providers.base import LLMProvider
from azmo_mind.state import EmotionStateStore


@dataclass(frozen=True)
class EvalResult:
    name: str
    passed: bool
    issues: list[str]
    speech: str
    gesture: str


@contextmanager
def isolated_engine(config: AppConfig, provider: LLMProvider):
    """An engine whose memory, state and log live in a temporary directory.

    Yields a fully-formed ``AzmoEngine`` — same config, same provider, same
    prompt — that simply cannot reach the live stores. The sandbox is removed on
    exit whether or not the cases passed.
    """
    with tempfile.TemporaryDirectory(prefix="azmo-eval-") as tmp:
        sandbox = Path(tmp)
        scratch = config.model_copy(deep=True)
        scratch.memory.database_path = sandbox / "memory.sqlite3"
        scratch.runtime.log_path = sandbox / "runtime.jsonl"

        memory = MemoryStore(scratch.memory.database_path)
        # Carry over explicit memories — not conversation turns — so retrieval is
        # exercised against the same facts the live system would see.
        try:
            live = MemoryStore(config.memory.database_path)
            for item in live.list_memories(limit=500):
                memory.add_memory(item.text, item.importance)
        except Exception:  # noqa: BLE001, S110 - no live database yet is fine
            pass

        yield AzmoEngine(
            scratch,
            provider,
            memory=memory,
            state_store=EmotionStateStore(sandbox / "emotion_state.json"),
        )


def run_cases(engine: AzmoEngine, path: str | Path) -> list[EvalResult]:
    data = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    results: list[EvalResult] = []

    for case in data.get("cases", []):
        result = engine.respond(str(case["input"]))
        speech_lower = result.response.speech.lower()
        issues: list[str] = []

        expected = set(case.get("expected_gestures", []))
        if expected and result.response.gesture.name not in expected:
            issues.append(
                f"gesture {result.response.gesture.name!r} not in expected {sorted(expected)}"
            )

        for phrase in case.get("forbidden_phrases", []):
            if str(phrase).lower() in speech_lower:
                issues.append(f"forbidden phrase present: {phrase!r}")

        results.append(
            EvalResult(
                name=str(case["name"]),
                passed=not issues,
                issues=issues,
                speech=result.response.speech,
                gesture=result.response.gesture.name,
            )
        )

    return results
