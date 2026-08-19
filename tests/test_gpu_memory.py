"""GPU memory parsing and the VRAM budget check."""

from __future__ import annotations

from azmo_mind import gpu


def test_parse_memory_csv_reads_mib_values():
    state = gpu.parse_memory_csv("12288, 7400, 4888")
    assert state.total == 12288
    assert state.used == 7400
    assert state.free == 4888
    assert state.available


def test_parse_memory_csv_tolerates_units_and_blank_lines():
    state = gpu.parse_memory_csv("\n12288 MiB, 7400 MiB, 4888 MiB\n")
    assert state.total == 12288
    assert state.used == 7400


def test_parse_memory_csv_handles_no_data():
    state = gpu.parse_memory_csv("")
    assert not state.available
    assert state.error


def test_budget_rejects_a_9b_at_8k_context_beside_xtts_on_12gb():
    # The configuration that was thrashing: ~7.2 GB resident LLM on a 12 GB card,
    # leaving too little for XTTS plus the Windows desktop.
    ok, message = gpu.vram_budget(total_mb=12288, llm_used_mb=7400)
    assert not ok
    assert "Over budget" in message


def test_budget_accepts_the_same_model_at_reduced_context():
    # Dropping num_ctx 8192 -> 4096 frees roughly a gigabyte of KV cache.
    ok, _message = gpu.vram_budget(total_mb=12288, llm_used_mb=6300)
    assert ok


def test_budget_is_comfortable_on_a_24gb_card():
    ok, message = gpu.vram_budget(total_mb=24576, llm_used_mb=7400)
    assert ok
    assert "spare" in message


def test_budget_message_names_a_remedy_when_over():
    _ok, message = gpu.vram_budget(total_mb=8192, llm_used_mb=6000)
    assert "context_tokens" in message
    assert "clone_device" in message


def test_cpu_voice_frees_the_budget():
    # speech.clone_device: cpu takes XTTS out of VRAM entirely.
    ok, _ = gpu.vram_budget(total_mb=12288, llm_used_mb=7400, voice_mb=0)
    assert ok
