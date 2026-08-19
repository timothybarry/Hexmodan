from azmo_mind.config import load_config
from azmo_mind.engine import AzmoEngine
from azmo_mind.memory import MemoryStore
from azmo_mind.providers.mock import MockProvider
from azmo_mind.state import EmotionStateStore


def test_engine_with_mock_provider(tmp_path):
    cfg = load_config("config/azmo.yaml")
    cfg.memory.database_path = tmp_path / "memory.sqlite3"
    cfg.runtime.log_path = tmp_path / "runtime.jsonl"

    engine = AzmoEngine(
        cfg,
        MockProvider(),
        memory=MemoryStore(cfg.memory.database_path),
        state_store=EmotionStateStore(tmp_path / "state.json"),
    )
    result = engine.respond("Awaken.")
    assert result.response.speech
    assert result.response.gesture.name == "loom"
    assert cfg.runtime.log_path.exists()
