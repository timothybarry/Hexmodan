from azmo_mind.config import load_config
from azmo_mind.memory import Memory
from azmo_mind.prompts import (
    VOLATILE_MARKER,
    build_system_prompt,
    static_prefix,
    volatile_suffix,
)
from azmo_mind.schemas import EmotionState


def _memory(text: str, id: int = 1) -> Memory:
    return Memory(id=id, text=text, importance=1.0)


def test_prompt_contains_lore_and_grounding():
    cfg = load_config("config/azmo.yaml")
    prompt = build_system_prompt(cfg, EmotionState(), [])
    assert "Azmodan" in prompt
    assert "Lord of Sin" in prompt
    assert "strategic" in prompt.lower()
    assert "Never invent observations" in prompt
    assert "Return only the structured JSON" in prompt


def test_memories_reach_the_prompt():
    cfg = load_config("config/azmo.yaml")
    prompt = build_system_prompt(cfg, EmotionState(), [_memory("Dana is his sister.")])
    assert "Dana is his sister." in prompt


def test_no_memories_says_so_rather_than_leaving_a_hole():
    cfg = load_config("config/azmo.yaml")
    assert "No relevant stored memories" in build_system_prompt(cfg, EmotionState(), [])


# ---------------------------------------------------------------------------
# Prefix-caching invariants.
#
# Prefix caching reuses work only up to the FIRST byte that differs between two
# prompts. Volatile content placed above the ~6 KB of lore kills the cache
# before the lore is reached, so all of it is re-prefilled every turn. These
# tests are what stop that regression from silently returning.
# ---------------------------------------------------------------------------

def test_prompt_is_exactly_static_prefix_then_volatile_suffix():
    cfg = load_config("config/azmo.yaml")
    state = EmotionState()
    memories = [_memory("Dana is his sister.")]
    prompt = build_system_prompt(cfg, state, memories)
    assert prompt.startswith(static_prefix(cfg))
    assert prompt.endswith(volatile_suffix(state, memories))


def test_static_prefix_is_byte_identical_across_differing_turns():
    """The cacheable half must not move when state or memories change."""
    cfg = load_config("config/azmo.yaml")
    calm = build_system_prompt(cfg, EmotionState(), [])
    roused = build_system_prompt(
        cfg,
        EmotionState(irritation=0.9, dominance=0.95, calculation=0.2),
        [_memory("He distrusts the new PSU."), _memory("Paul builds the legs.", id=2)],
    )
    prefix = static_prefix(cfg)
    assert calm.startswith(prefix)
    assert roused.startswith(prefix)
    # And the divergence begins only at the volatile boundary - not earlier.
    common = 0
    for a, b in zip(calm, roused):
        if a != b:
            break
        common += 1
    assert common >= len(prefix), (
        "prompts diverge before the end of the static prefix, so prefix caching "
        "dies early and the lore is re-prefilled every turn"
    )


def test_volatile_content_sits_after_all_the_lore():
    cfg = load_config("config/azmo.yaml")
    prompt = build_system_prompt(cfg, EmotionState(), [_memory("Dana is his sister.")])
    boundary = prompt.index(VOLATILE_MARKER)
    for heading in ("PERSONALITY REFERENCE", "DIALOGUE REFERENCE", "GESTURE REFERENCE"):
        assert prompt.index(heading) < boundary, f"{heading} must precede volatile content"
    for volatile in ("CURRENT INTERNAL STATE", "RELEVANT MEMORIES", "Dana is his sister."):
        assert prompt.index(volatile) > boundary, f"{volatile} must follow the lore"


def test_static_prefix_does_not_leak_state_or_memories():
    cfg = load_config("config/azmo.yaml")
    prefix = static_prefix(cfg)
    assert "CURRENT INTERNAL STATE" not in prefix
    assert "RELEVANT MEMORIES" not in prefix
    assert VOLATILE_MARKER not in prefix
