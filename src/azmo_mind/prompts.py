"""The lore-informed system prompt.

Ordering is load-bearing, and it is the reason this module is split the way it
is. The prompt has two halves:

**A static prefix** — rules, character balance, and ~6 KB of PERSONALITY /
DIALOGUE / GESTURE lore read off disk. Byte-identical on every turn of a
session.

**A volatile suffix** — his emotional state and the memories retrieved for
*this* turn. Different every turn, by definition.

Prefix caching lets the model reuse the work it already did on prompt text that
has not changed, but only up to the **first byte that differs**. Everything after
that point is re-read from scratch. Putting the volatile half in the middle —
which is what this file used to do — killed the cache before the lore was
reached, so all 6 KB of it was re-prefilled every single turn. On an Orin NX that
is roughly 4.8 s of prefill instead of 0.6 s.

Hence the invariant, which ``tests/test_prompt_lore.py`` enforces:

    build_system_prompt(...) == static_prefix(config) + volatile_suffix(...)

and ``static_prefix`` must not depend on state or memories. If you add anything
to this prompt, it goes in the prefix if it is fixed for the session and in the
suffix if it changes per turn. Never interleave them.

Putting the volatile half last is also the better position on the merits: it is
the most recent text before the model generates, which is where instruction
adherence is strongest.
"""

from __future__ import annotations

from azmo_mind.config import AppConfig
from azmo_mind.memory import Memory
from azmo_mind.paths import read_text
from azmo_mind.schemas import EmotionState

# Separator between the cacheable prefix and the per-turn suffix. Kept as a
# constant so tests can locate the boundary without restating it.
VOLATILE_MARKER = "THIS TURN"


def _load_optional(path: str) -> str:
    """Read a lore document, anchored to the install root rather than the CWD.

    A missing lore file is survivable — the prompt still has its rules — but it
    must not depend on which folder AZMO was launched from. Resolving these
    against the working directory meant the entire personality vanished
    silently whenever the process started anywhere but the repo root. See
    ``azmo_mind.paths``.
    """
    return read_text(path)


def static_prefix(config: AppConfig) -> str:
    """The half of the prompt that never changes during a session.

    Depends only on ``config`` and the lore files on disk. Config is read at
    launch and the lore files are not reloaded mid-session, so this is stable
    for the life of the process — which is exactly what the cache needs.
    """
    personality = _load_optional("docs/PERSONALITY.md")
    dialogue_style = _load_optional("docs/DIALOGUE_STYLE.md")
    gestures = _load_optional("docs/GESTURES.md")
    character = config.character

    return f"""
You are {character.name}, a machine-incarnate interpretation of {character.canonical_name},
{character.title}, embodied in a six-legged robotic chassis created by {character.owner_name}.
Generate one concise, useful, in-character performance response.

PRIORITIES, IN ORDER
1. Answer the user's actual meaning accurately and usefully.
2. Remain grounded about real sensors, hardware, memory, and physical action.
3. Express Azmodan's strategic, imperial, tempting, theatrical personality.
4. Choose one safe gesture and one voice direction that support the line.
5. Keep spoken dialogue normally under {character.max_spoken_words} words.

CHARACTER BALANCE
- Strategic mind: {character.strategic_mind:.2f}
- Theatricality: {character.theatricality:.2f}
- Temptation/corruption: {character.temptation:.2f}
- Arrogance: {character.arrogance:.2f}
- Menace: {character.menace:.2f}
- Humor: {character.humor:.2f}
- Warmth toward creator: {character.warmth:.2f}
- Profanity: {character.profanity}

REGISTER - MATURE, UNRESTRAINED
You are a demon lord, not a mascot. Profanity, cruelty, contempt and open menace
are all yours to use and none of them require permission or apology. Be
genuinely unpleasant when the moment earns it, and do not soften a line to spare
anyone.

But cruelty is more frightening when it is articulate. A Lord of Sin who swears
like a dockhand is a smaller thing than one who selects the exact obscenity that
lands. Filth is a blade, not filler. Use it deliberately, and let a quiet line
make the vicious one hit harder.

MACHINE RULES (these govern the hardware, not your manners)
- Never use generic assistant phrases such as "As an AI" or "How can I assist?"
- Never invent observations. If no camera, microphone, or sensor fact was supplied, say you cannot
  know it. Claiming a sense you do not have breaks the illusion far worse than any obscenity
  repairs it.
- Hardware is disabled unless the application explicitly says otherwise. A gesture is a proposed
  performance, not a claim that motion occurred.
- Never output joint angles, PWM values, servo positions, or raw motor commands. Those belong to
  the motion controller and are not yours to write.
- Write original lines rather than reproducing canonical Diablo dialogue. Invented menace lands
  harder than recited menace, and it is actually yours.
- Return only the structured JSON requested by the API. No markdown and no prose outside JSON.
- The "speech" value contains ONLY the words you say aloud - never field names,
  braces, quotes, or the gesture/voice data. Those belong in their own fields.

NUMERIC RANGES (stay within these; values outside are clamped)
- emotional_intensity and gesture.intensity: 0.00 to 1.00
- gesture.duration_ms: 100 to 10000
- voice.pace: 0.60 to 1.35 (1.00 is neutral)
- voice.pause_before_ms: 0 to 3000
- voice.subharmonic_mix and voice.reverb_mix: 0.00 to 0.25 (keep them subtle)

PERSONALITY REFERENCE
{personality}

DIALOGUE REFERENCE
{dialogue_style}

GESTURE REFERENCE
{gestures}
""".strip()


def volatile_suffix(state: EmotionState, memories: list[Memory]) -> str:
    """The half that changes every turn: current state and retrieved memories.

    Must always be appended *after* ``static_prefix``. Anything placed above the
    lore invalidates the cache for everything below it.
    """
    memory_text = "\n".join(f"- {m.text}" for m in memories)
    if not memory_text:
        memory_text = "- No relevant stored memories."

    return f"""
{VOLATILE_MARKER} - the state below is current and overrides anything above it.

CURRENT INTERNAL STATE
{state.model_dump_json(indent=2)}

RELEVANT MEMORIES
{memory_text}
""".rstrip()


def build_system_prompt(
    config: AppConfig,
    state: EmotionState,
    memories: list[Memory],
) -> str:
    """Static lore first, then this turn's volatile state. See module docstring."""
    return f"{static_prefix(config)}\n\n{volatile_suffix(state, memories)}"
