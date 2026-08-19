# AZMO — design log

Dated decisions, newest first. Each entry records *what was decided and why*,
so a later session inherits the reasoning instead of re-litigating it.

This file is append-only in spirit: don't rewrite old entries when a decision is
superseded — add a new entry that says so and link back. A decision that was
right in July and wrong in September is useful history, not a mistake to erase.

---

## 2026-08-02 — Streaming shipped, and the latency story was wrong

**Present:** Tim, Claude.

### What was built

0.2.10: the LLM streams, his speech is rendered in chunks while the model is
still writing, and playback starts once a **prebuffer** of rendered chunks
exists. This is the design the 2026-07-30 entry specified, built as specified —
including the prebuffer, which was the non-optional part.

It ships **off** (`speech.stream_playback: false`). Two reasons, and neither is
a code problem, so neither can be cleared by more code:

1. **Nobody has heard it.** Whether the chunk seams are audible is an ear
   question, exactly like curating the presence pool.
2. **It runs the LLM and XTTS concurrently.** That is precisely the back-to-back
   GPU load `gpu.stagger_ms` was added to break apart. It should not be switched
   on before the new cooler and PSU are in and the box has proven stable.

### The measurement that changes the picture

Streaming was justified by "the model takes 5–20 s." That number was in
`HANDOFF.md`, it was never measured, and it is wrong. Measured on this box, with
the real ~8.4 KB system prompt (2055 prompt tokens):

| | model load | prefill | generation | wall | streaming head start |
|---|---|---|---|---|---|
| **prefix cache hit** | 0.32 s | **0.06 s** | 2.06 s | **2.46 s** | 1.95 s |
| **prefix cache miss** | 0.33 s | **8.94 s** | 2.27 s | 13.66 s | 2.17 s |

Three things follow, and they matter more than the feature does.

**1. The prefix-caching fix from 0.2.9 is worth ~9 seconds a turn.** 2055 prompt
tokens prefill in 0.06 s on a hit and 8.94 s on a miss. That fix was reasoned
about and shipped without a number attached; this is the number. Anything that
puts volatile content back above the lore costs ~9 s per turn, every turn.

**2. The brain is not the bottleneck. The voice is.** A warm turn is 2.5 s of
thinking. XTTS then renders ~600 characters in several passes, which is far
longer. The turn is dominated by synthesis, not by thought, and every latency
intuition in the docs was built on the opposite assumption.

**3. Streaming's win is therefore not mostly the LLM overlap.** Only ~2 s of a
turn can be overlapped that way — the whole generation phase. The larger win is
the second half of the same mechanism: playback of chunk 1 no longer waits for
chunks 2..N to render. That is also exactly where a stall can happen, which is
why the prebuffer is not optional and why `stalls` is reported per turn.

**Unmeasured, deliberately:** XTTS render time per chunk. Getting it means
loading the voice model onto a GPU behind a seven-year-old PSU and a cooler that
has already caused thermal shutdowns. It is the *first* thing to measure after
the teardown, because it is the number that sets `stream_prebuffer_chunks`.

### The DSP problem streaming created

`apply_azmo_voice` peak-normalises three times per call. Over a whole reply that
is correct and deliberate — `speech.py` runs the DSP once over the concatenation
specifically so loudness is consistent across chunks.

Rendering chunks separately silently breaks that: each chunk normalises to the
same ceiling on its own, so a murmured closing clause is lifted to match a
shouted opening one. The result is loudness pumping at every chunk boundary. It
does not raise, it is not a stutter, and it is squarely in the *audio* column of
the two-evaluations table — the automatable one.

Fixed with `voice_dsp.GainAnchor`: chunk 1 captures the three scale factors and
every later chunk reuses them, which puts the reply back in one gain frame. The
non-streamed path passes `None` and is byte-identical to before, which is pinned
by a test.

Worth recording: the chain's own saturation and compression already squash a
20 dB input difference to about 3 dB, so the defect was less severe than raw
normalisation suggests — but 3 dB flipping chunk to chunk is plainly audible.

### Known limitation: streamed replies lose his chosen delivery

`AzmoResponse` declares `speech` first and `voice` fifth, so a chunk must be
rendered before its `VoiceDirection` exists. Streamed replies therefore render
with the default direction.

This costs less than it sounds like, because two locked settings already
suppress most of that direction on purpose: `heaviness_variation` damps the
preset/mix swing almost to nothing, and `effective_pace` keeps only ~30% of the
model's pace swing. What is actually lost is a few percent of tempo.

The fix is to declare `voice` and `emotion` *before* `speech` in the schema, at
a cost of a few dozen tokens' delay. **Not done, on purpose:** reordering the
fields changes what the model writes, and whether he stays in character is
judged by ear, not by test. It is a question for Tim, not a bug.

### Still open

- Does he sound right streamed? Only listening answers this.
- XTTS render time per chunk, after the teardown. It sets the prebuffer.
- The `gpu:` comment block in `config/azmo.yaml` still states the disproven
  power-transient diagnosis as settled. `HANDOFF.md` corrected the diagnosis on
  2026-07-31; the config did not follow. Still outstanding.

---

## 2026-07-30 — The POC reframe: presence over speed

**Present:** Tim, Claude.

### What changed

The project's stated latency goal has been "≤4 s from you finishing your
sentence to hearing his first word." That number is now **a general goal, not a
requirement**, and the real objective was restated:

> The enemy is dead air, not elapsed time.

A robot that sits inert and silent for six seconds reads as broken. A robot that
audibly or visibly *turns the question over* for eight reads as thinking. If
AZMO fills the gap with contemplation, the gap is allowed to be significantly
longer than four seconds.

This is a change of optimization target, not a relaxation of standards.
Perceived smoothness is now the metric, and it is judged by ear, by the builder,
in live conversation.

### What this project currently is

**A proof of concept.** The deliverable is a convincing enough Azmodan on the
Windows desktop that buying the Jetson Orin NX becomes an obvious decision.
Hardware is downstream of this POC succeeding, not parallel to it.

Legs, servos and the gesture→signal translation layer are **Paul's lead** and
come later. `motion_link.py` already speaks the correct command envelope and can
sit untouched until there is hardware to attach it to.

### Consequences — things that follow from the reframe

**1. Model size pressure now points UP, not down.**

Every argument for shrinking the model was a latency argument. With
contemplation covering the gap, that pressure is gone, and the builder's stated
fear is losing coherent Azmodan — not slow replies.

So `azmo compare` inverts. It stops asking *how small can I get away with* and
starts asking *how large can I afford*. The metric is character fidelity: does he
still feel like the Azmodan in `docs/AZMODAN_LORE.md`, is the personality
coherent under pressure, does he reach for novel imagery or fall back on stock
phrasing.

**2. Streaming overlap gets riskier, not merely less urgent.**

Overlapping the LLM and XTTS trades one long silence *before* he speaks for the
risk of a stall *mid-sentence* — if chunk 3 isn't rendered before playback drains
chunk 2, he stutters.

Under a hard 4 s target that trade is worth taking. Under "smooth" it is not:
a confident pause followed by unbroken delivery reads as *deliberate*, while a
gap in the middle of a line reads as *broken*.

Overlap is therefore still wanted, but **with a prebuffer** — hold 2–3 rendered
chunks before the first word plays. This spends part of the latency win to
guarantee he never stutters. Slower than pure overlap. Better presence.

**3. Two evaluations, not one.**

"Worse" means two unrelated things and they need separate harnesses:

| | Failure | How it's judged |
|---|---|---|
| **Audio** | underwater, low-pass, garbled, uncrisp | automatable — render known-good lines, compare |
| **Character** | incoherent personality, off-lore, stock phrasing | human — read and listen, not scriptable |

Only the first can be automated. Don't pretend otherwise.

### Deferred (explicitly, with reasons)

| Deferred | Why |
|---|---|
| Contemplation **gesture** | Decided: deterministic, chosen from the 8-dimension emotional state at end-of-transcription, no LLM in the motion path. Parked because there is no body — it can be built and unit-tested but not *felt*, so it cannot serve the POC. |
| Hotword engine (openWakeWord) | Solves a battery/always-on problem on a robot that doesn't exist yet. |
| Most of `JETSON_MIGRATION.md` | Optimizes for a machine not yet purchased. The purchase depends on this POC. |
| Motion / Teensy protocol | Paul's lead, post-POC. |

### Accepted — presence via audio

With no body, audio is AZMO's only channel, so dead air is killed with sound.

**A pool of pre-rendered non-verbals**, not a single clip — a single breath on
loop becomes a tic within one session. Both registers were wanted:

- **slow deliberate exhale** — reads as menace and control; he has *decided* to
  consider you
- **low considering growl** — reads as active processing

Fired when thinking starts, and **repeatedly** while thinking continues. A single
sound at t=0 does nothing for the person still waiting at t=9; a contemplation
track that breathes every few seconds is what actually sells presence on a long
turn. That is the whole point of the reframe, and it is why `presence.sustain_*`
exists.

Pre-rendered means near-zero latency at request time: no LLM, no XTTS, no GPU —
just a WAV read. First sound lands in well under a second.

### Corrections to the record

- **`HANDOFF.md` was stale and self-contradictory.** It listed
  `torch.inference_mode()` both as a thing that natively aborts XTTS *and* as a
  0.2.6 mitigation. The code was never wrong: `speech.py::_inference_context`
  correctly uses `no_grad()` and documents why. Stale prose, not a bug.
- **Version drift.** HANDOFF claimed 0.2.6 / 58 tests. Actual: `pyproject.toml`
  is 0.2.8 and the suite is **117 passing** across 19 files, verified by running
  it rather than by reading about it.
- **The prefix-caching problem is real.** `prompts.py::build_system_prompt` put
  `CURRENT INTERNAL STATE` and `RELEVANT MEMORIES` *above* ~6 KB of static
  PERSONALITY / DIALOGUE / GESTURE lore, so the cache died at the first volatile
  byte and everything after it was re-read every turn. Fixed this session.

### Still open

- Does the model hold character better at a larger size? Needs `azmo compare`
  after the PSU lands.
- The 8 unexplained WHEA events (May–Jul 4) that the power-transient story does
  not account for. Suspect RAM/XMP.
- Whether the GPU power cap is still wanted once the Montech 1050 W ATX 3.1 is
  installed (2026-07-31).

---

## Standing constraints (not up for casual revision)

These have each cost real debugging time. Changing one requires a reason and a
test, not a hunch.

- The LLM never owns safety-critical hardware, the serial port, or its own
  emotional state.
- `dsp.use_world: false`. WORLD re-synthesis caused the underwater tone.
- No pitch or formant shifting. The clone is already deep.
- `clone_seed` fixed and non-zero; **re-seed before every chunk**. That
  re-seeding is why our chunking preserves character where XTTS's own splitting
  did not.
- Never pass >~230 chars to XTTS with splitting disabled.
- `no_grad()`, never `inference_mode()`. Never touch `torch.backends.cudnn.*`.
- Warm ears **before** voice.
- `listener.always_on` stays false — it removes the wake word as a guard.
- All `.ps1` / `.bat` files stay pure ASCII.
