from azmo_mind.memory import MemoryStore


def test_memory_roundtrip(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    memory_id = store.add_memory("Timothy prefers restrained theatrical dialogue.")
    results = store.retrieve(
        "What dialogue style does Timothy prefer?",
        limit=5,
    )
    assert memory_id > 0
    assert results
    assert "restrained" in results[0].text
    assert store.delete_memory(memory_id) is True
