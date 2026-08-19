from azmo_mind.config import load_config


def test_default_config_loads():
    cfg = load_config("config/azmo.yaml")
    assert cfg.provider.model == "qwen3.5:9b"
    assert cfg.motion.hardware_enabled is False


def test_context_stays_within_the_vram_budget():
    # The measured prompt is ~2500 tokens. 8192 reserved a KV cache three times
    # larger than anything used, and that gigabyte is what XTTS needed on a
    # 12 GB card. Raising this again means re-checking docs/PERFORMANCE.md.
    cfg = load_config("config/azmo.yaml")
    assert cfg.provider.context_tokens <= 4096
