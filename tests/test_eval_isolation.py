"""`azmo eval` must not disturb the conversation it is measuring.

Regression: cases ran through the live engine, so every run appended ten
exchanges to the real conversation history and pushed the persistent emotional
state five decay steps. Two consequences, both bad: AZMO was measurably
different after you measured him, and a second run read the first run's turns
back as recent context, so the cases were no longer independent of run order.
"""

import json

from azmo_mind.config import load_config
from azmo_mind.evaluation import isolated_engine, run_cases
from azmo_mind.memory import MemoryStore
from azmo_mind.providers.mock import MockProvider
from azmo_mind.state import EmotionStateStore


def _live_config(tmp_path):
    """A config whose 'live' stores point somewhere we can safely inspect."""
    cfg = load_config("config/azmo.yaml")
    cfg.memory.database_path = tmp_path / "live.sqlite3"
    cfg.runtime.log_path = tmp_path / "live.jsonl"
    return cfg


def test_eval_leaves_live_conversation_turns_untouched(tmp_path):
    cfg = _live_config(tmp_path)
    live = MemoryStore(cfg.memory.database_path)
    live.add_turn("user", "something I actually said")
    before = live.recent_turns(50)

    with isolated_engine(cfg, MockProvider()) as engine:
        run_cases(engine, "eval/cases.yaml")

    assert MemoryStore(cfg.memory.database_path).recent_turns(50) == before


def test_eval_does_not_write_the_live_log(tmp_path):
    cfg = _live_config(tmp_path)
    with isolated_engine(cfg, MockProvider()) as engine:
        run_cases(engine, "eval/cases.yaml")
    assert not cfg.runtime.log_path.exists()


def test_eval_state_store_is_not_the_live_one(tmp_path):
    cfg = _live_config(tmp_path)
    live_state_path = EmotionStateStore().path
    with isolated_engine(cfg, MockProvider()) as engine:
        assert engine.state_store.path != live_state_path
        run_cases(engine, "eval/cases.yaml")
        written = json.loads(engine.state_store.path.read_text(encoding="utf-8"))
    # The sandbox state did move — proving the cases really ran and really wrote.
    assert "dominance" in written


def test_every_store_lives_inside_the_sandbox(tmp_path):
    cfg = _live_config(tmp_path)
    with isolated_engine(cfg, MockProvider()) as engine:
        sandbox = engine.config.memory.database_path.parent
        assert engine.config.runtime.log_path.parent == sandbox
        assert engine.state_store.path.parent == sandbox
        assert tmp_path not in engine.config.memory.database_path.parents


def test_sandbox_is_removed_afterwards(tmp_path):
    cfg = _live_config(tmp_path)
    with isolated_engine(cfg, MockProvider()) as engine:
        sandbox = engine.config.memory.database_path.parent
        assert sandbox.exists()
    assert not sandbox.exists()


def test_explicit_memories_are_visible_inside_the_sandbox(tmp_path):
    """Retrieval is part of what the cases exercise, so the facts come along."""
    cfg = _live_config(tmp_path)
    MemoryStore(cfg.memory.database_path).add_memory("Timothy prefers PETG.", 0.9)

    with isolated_engine(cfg, MockProvider()) as engine:
        assert "Timothy prefers PETG." in [m.text for m in engine.memory.list_memories()]


def test_runs_are_independent_of_each_other(tmp_path):
    """Two runs in a row must see the same starting context, not accumulate."""
    cfg = _live_config(tmp_path)
    with isolated_engine(cfg, MockProvider()) as engine:
        first = run_cases(engine, "eval/cases.yaml")
    with isolated_engine(cfg, MockProvider()) as engine:
        second = run_cases(engine, "eval/cases.yaml")
    assert [r.speech for r in first] == [r.speech for r in second]


def test_cases_still_actually_run(tmp_path):
    cfg = _live_config(tmp_path)
    with isolated_engine(cfg, MockProvider()) as engine:
        results = run_cases(engine, "eval/cases.yaml")
    assert len(results) == 5
    assert all(r.speech for r in results)
