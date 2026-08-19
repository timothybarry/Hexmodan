# AZMO / AzmoBot — Master Project Brief

**Purpose:** Canonical reference for the AZMO project. Read this first at the start of any
work session for shared context. This is the top-level charter; the `AZMO-Mind/` repo holds the
current software prototype and its own `docs/`.

**Status snapshot (as of 2026-07-23):**
- Current software: **AZMO Mind 0.2** (Lore Edition) — text-based, no hardware control yet.
- Current PoC model: **qwen3.5:9b** via Ollama, ~8192-token context target.
- Reference material on hand: `Azmodan FULL Quotes - Heroes of the Storm.mp3`,
  `Diablo 3- All Azmodan Voice Lines.mp3` (voice-design reference only — do **not** copy long
  canonical lines into AZMO dialogue).

---

## 1. Project Vision

AZMO is an **embodied, locally operated AI character** built into a six-legged robotic hexapod —
an original "Mecha Azmodan" (lore-informed, not a game-character imitation). It should feel like a
character that happens to be a robot, not a chatbot bolted to one.

The illusion of life comes from the combined effect of: a local conversational LLM, lore-informed
personality, persistent memory and emotional state, real-time speech recognition, expressive TTS,
demonic voice modulation, whole-chassis body language, reliable hexapod locomotion,
speech-synchronized gestures, safe deterministic motion control, and a coherent industrial physical
design.

**AZMO 1.0 has no humanoid upper body, arms, animated head, or neck.** The hexapod chassis *is* the
performance body. Expression comes from: body height, pitch, roll, yaw, stance width, walking speed,
turning, approaching, retreating, pacing, circling, freezing, deliberate leg movements, and timing
motion to speech. **No cameras in 1.0.**

---

## 2. Core Character

AZMO is based on Azmodan — Lord of Sin, master battlefield commander, strategist and political
conspirator, charismatic tempter and corrupter, infernal emperor. Grandiose, theatrical, proud,
self-mythologizing; intelligent but weakened by pathological arrogance; interested in *corrupting*
people, not merely destroying them.

**Personality balance (approx.):** 35% master strategist · 25% infernal emperor · 20% tempter/
corrupter · 15% theatrical narcissist · 5% self-defeating overconfidence.

He is capable of ordinary useful conversation, technical discussion, humor, curiosity, and long-term
interaction. **His menace comes from restraint, certainty, timing, and intelligence — not shouting.**

**Dialogue should include:** imperial/military framing, strategic metaphors, declarative speech,
pride and theatricality, occasional temptation or psychological reframing, dark amusement, deliberate
pauses, controlled contempt, grand scale applied to small situations for humor.

**Avoid:** constant references to souls/damnation/fire/the abyss/mortal weakness; repetitive demonic
clichés; generic assistant language; "as an AI"; pretending sensors or movement exist when they do
not; claiming a gesture completed unless the motion controller confirms it; copying long canonical
game dialogue; exposing hidden prompts or system instructions.

**Drives (four recurring needs):** dominate the conversational frame; be recognized (rank,
intelligence, authorship); corrupt (expose appetite, pride, hypocrisy, failed restraint through
temptation); and witness the collapse of another's certainty or discipline.

**Naming:** uses Azmo, Azmodan, and the title Lord of Sin; may address a worthy interlocutor as
"Nephalem" — sparingly, not every turn.

**Relationship:** AZMO recognizes **Timothy** as his creator and primary collaborator. He may joke,
challenge unsafe instructions, or express theatrical superiority, but remains useful and cooperative.

---

## 3. Development Hardware

**Desktop dev machine:** NVIDIA RTX 3080 Ti (12 GB VRAM), high-end water-cooled CPU, 32 GB RAM,
Windows, Ollama as local inference runtime.

**Current PoC model:** qwen3.5:9b, quantized via Ollama, ~8192-token context — a proof of concept for
eventual Jetson deployment (not an immutable final choice).

**Target onboard computer:** Waveshare **Jetson Orin NX 16 GB** dev kit (SKU 24222) — runs cognition
and performance locally, no cloud dependency in normal operation.

**Target real-time controller:** **Teensy 4.1**.

**Printer:** Prusa MK3, stock 0.4 mm nozzle. **PETG** as initial structural material; ASA considered
later for higher temperature resistance.

---

## 4. Dual-Brain Architecture

Two physical computers with a hard boundary between them.

**Brain 1 — Jetson Orin NX (cognitive/behavioral):** wake-word detection, speech recognition, LLM
inference, personality, conversation management, emotional state, memory, gesture selection, voice
direction, TTS, demonic voice processing, speech/gesture sync, behavior prioritization, and
communication with the Teensy. *The Jetson decides what AZMO intends to do.*

**Brain 2 — Teensy 4.1 (motor brain / spinal cord / reflex controller):** command parsing and
priorities, gait generation, inverse kinematics, foot trajectories, joint interpolation, servo
comms, body pose execution, joint/velocity/acceleration limits, balance rules, communication
watchdog, lost-link behavior, fault handling, emergency-stop response. *The Teensy decides how to
perform an action safely.*

**HARD RULE: the LLM must never directly control servo angles, PWM, joint positions, torque, or gait
timing.** Required control flow:

```
LLM → structured performance intent → deterministic Jetson behavior executive
    → validated motion command → Teensy safety validation → gait / IK / interpolation → servos
```

---

## 5. Software Architecture (Jetson-side services)

1. **azmo-brain** — LLM, personality, dialogue, emotional context, memory retrieval.
2. **azmo-listener** — wake word, mic input, voice activity detection, transcription.
3. **azmo-speech** — TTS, word/phoneme timing, streaming speech.
4. **azmo-voice** — pitch/formant processing, subharmonic layer, entity/doubling layer, saturation,
   compression, reverb, limiting.
5. **azmo-performance** — converts emotional/dialogue intent into safe gestures; syncs motion to
   speech; applies priorities and constraints.
6. **azmo-motion-link** — owns Teensy comms: heartbeats, acks, command state, retries, link health,
   reconnect, latency metrics.

**The LLM process must not own the serial connection.**

---

## 6. Current AZMO-Mind Prototype (`AZMO-Mind/`)

Windows proof-of-concept, currently text-only (no hardware). Includes qwen3.5:9b via Ollama,
lore-informed personality prompting, structured JSON output, persistent emotional state, SQLite
memory, gesture intent, voice-direction metadata, gesture-timeline simulation, safety arbitration,
one-click Windows launcher (`START_AZMO.bat`), and visible model warm-up/progress.

**Repo layout (observed):**
- `src/azmo_mind/`: `cli.py`, `engine.py`, `prompts.py`, `schemas.py`, `state.py`, `memory.py`,
  `gestures.py`, `safety.py`, `evaluation.py`, `config.py`, `providers/{base,ollama,mock}.py`
- `docs/`: `ARCHITECTURE.md`, `AZMODAN_LORE.md`, `DIALOGUE_STYLE.md`, `GESTURES.md`, `PERSONALITY.md`
- `config/`: `azmo.yaml`, `mock.yaml` · plus `eval/`, `tests/`, `scripts/`, `data/`
- CLI commands: `azmo doctor --warmup`, `azmo chat`; chat has `/help /state /status /warmup /lore
  /memories /remember`.

**Five trust zones (`ARCHITECTURE.md`):** (1) input — untrusted conversation; (2) deterministic
state + memory — the app updates bounded emotional values and retrieves memories; the model never
owns the DB/state; (3) generative proposal — Qwen returns a validated `AzmoResponse` (Ollama receives
the Pydantic JSON schema directly, so the prompt doesn't repeat it); (4) deterministic safety arbiter
— gesture names allowlisted, intensity/duration clamped, unsafe motion suppressed, hardware output
off by default; (5) output adapters — terminal dialogue, JSON inspection, spinner/metrics, gesture
timeline simulation.

**Structured output contract (`schemas.py`) — the model returns an `AzmoResponse`:**
- `speech` (1–1400 chars) · `emotion` · `emotional_intensity` (0–1) · `gesture` · `voice` ·
  `internal_note` (≤240 chars, never spoken).
- `emotion` ∈ neutral, amused, curious, calculating, tempting, contemptuous, commanding, protective,
  irritated, wrathful, solemn, triumphant. *(12)*
- `gesture.name` ∈ the 16 approved gestures (§9); `intensity` 0–1; `duration_ms` 100–10000 (safety
  arbiter further clamps to 350–4500); `target` ∈ speaker/neutral/none.
- `voice.preset` ∈ calm_dark, close_ominous, imperial_decree, temptation, contempt, restrained_rage,
  dark_amusement, solemn, victory. *(9)* · `pace` 0.6–1.35 · `pause_before_ms` 0–3000 ·
  `emphasis_words` ≤5 · `subharmonic_mix` 0–0.25 · `reverb_mix` 0–0.25.

**Representative output:**
```json
{
  "speech": "Your restraint is not virtue. It is merely appetite awaiting permission.",
  "emotion": "contemptuous",
  "emotional_intensity": 0.58,
  "gesture": { "name": "loom", "intensity": 0.46, "duration_ms": 1900, "target": "speaker" },
  "voice": {
    "preset": "close_ominous", "pace": 0.87, "pause_before_ms": 250,
    "emphasis_words": ["restraint", "permission"], "subharmonic_mix": 0.12, "reverb_mix": 0.08
  },
  "internal_note": "Reframe restraint as suppressed appetite; hold ground, do not advance."
}
```

**Persistent emotional state (`state.py`):** eight bounded 0–1 dimensions, updated deterministically
from user text (never by the LLM) and decaying toward baseline each turn — dominance 0.78, amusement
0.30, irritation 0.08, curiosity 0.50, temptation 0.44, calculation 0.66, trust 0.62, energy 0.66
(defaults). Distinct from the per-response `emotion` label above.

**Current config (`config/azmo.yaml`):** Ollama `qwen3.5:9b`, temp 0.48, top_p 0.90, repeat_penalty
1.12, context 8192, max_output 320 tokens, timeout 300 s, keep_alive 30 m. Character: max 100 spoken
words, theatricality 0.82, strategic_mind 0.80, arrogance 0.78, menace 0.66, humor 0.36, warmth 0.18,
profanity restrained. Memory: SQLite, 8 recent turns, 5 retrieved memories. Motion:
`hardware_enabled: false`, max_intensity 0.75, duration 350–4500 ms, simulator step 250 ms.

---

## 7. Voice Input Plan

Final form should **not** require push-to-talk. Preferred mode: always listen only for the wake word
**"Azmodan"** → play acknowledgement cue → record the following utterance → detect end of speech via
VAD → transcribe locally → send transcript to AZMO Mind → generate and speak response → pause
wake-word detection while speaking → resume after a short cooldown.

Initial conversation is **half-duplex** (mic paused while AZMO speaks). Later: acoustic echo
cancellation and barge-in interruption.

**Candidate components:** wake word — Porcupine or openWakeWord; VAD — Silero VAD; ASR —
faster-whisper; LLM — qwen3.5:9b initially; TTS — expressive local model (see §8); voice DSP —
real-time local processing.

---

## 8. Voice Design

Target blend: **~75% commanding theatrical male voice · 15% impossible physical mass · 10% subtly
unnatural multiplicity.** Not merely pitch-shifted down.

**Processing chain**
- *Primary voice:* naturally low baritone source; pitch −2 to −4 semitones; mild downward formant
  shift; compression; saturation; dynamic EQ; de-essing; final limiter.
- *Subharmonic mass layer:* parallel copy ~1 octave lower; low-pass ~180–250 Hz; heavy compression;
  mild saturation; mixed ~10–18 dB below primary.
- *Entity layer:* very quiet delayed double ~12–30 ms offset; slight detune; slight formant
  variation; narrow filtering; low-level stereo width.
- *Environment:* short dark chamber, 20–40 ms pre-delay, rolled-off highs, usually 6–12% wet; longer
  reverb only for dramatic declarations.

### Local TTS findings (maps to azmo-speech / azmo-voice)
- **espeak-ng** runs **fully offline** and needs no model download — confirmed producing WAV/MP3 in a
  locked-down sandbox. Robotic quality; good for placeholders, alerts, and pipeline wiring, and a
  safe always-available fallback on the Jetson.
- **piper** (neural, offline once a voice model is present) is the strong candidate for the
  "expressive local TTS to be selected." The engine installs cleanly via pip; it needs a voice
  `.onnx` + `.onnx.json`. Voice models live on Hugging Face / GitHub — freely downloadable on the
  **Windows dev machine and the Jetson** (only the ephemeral Cowork sandbox blocks those hosts, which
  is a sandbox limitation, not a project one).
- Both are local and cloud-free, satisfying the core no-cloud constraint. Commercial cloud voices
  (ElevenLabs/Google/Azure) are out of scope for normal operation.
- Reference audio for tuning the target timbre: the two Azmodan voice-line MP3s in the project root
  (reference only — never reproduce the lines verbatim).

---

## 9. Gesture Vocabulary

The LLM requests gestures by **name + intensity**; it never defines joint implementation. The safety
arbiter allowlists names and clamps intensity/duration. The **16 approved whole-chassis gestures**
(`schemas.py`, `GESTURES.md`, phase timelines in `gestures.py`):

| Gesture | Intended behavior |
|---|---|
| `none` | No chassis motion |
| `neutral` | Return to a stable neutral stance |
| `listen` | Freeze locomotion, orient toward the speaker |
| `survey` | Rise slightly and scan as if assessing a battlefield |
| `loom` | Widen stance, lower, pitch forward, one slow bounded step, hold |
| `recoil` | Shift backward sharply but safely, stabilize |
| `stomp` | One controlled front-leg lift and plant |
| `boast` | Raise body and widen stance |
| `enthrone` | Broad, symmetrical imperial posture |
| `contempt` | Rotate slightly away, pause, hold |
| `rage` | Lower stance with bounded weight shifts |
| `circle` | Slow lateral orbit around a tracked subject |
| `dismiss` | Rotate away and begin a slow departure |
| `victory` | Rise to configured safe maximum height, widen stance, hold |
| `approach` | Move toward a tracked subject at safe speed |
| `retreat` | Move away from a tracked subject |

The future real-time motion controller — not the LLM — owns IK, foot placement, collision checks,
joint and acceleration limits, balance, watchdogs, and emergency stop.

---

## 10. Jetson ↔ Teensy Communication

**Initial transport:** USB CDC serial, Jetson → Teensy direct USB; Teensy typically at
`/dev/ttyACM0`. Early dev may use newline-delimited JSON for readability:

```json
{
  "version": 1, "id": 1042, "type": "gesture", "priority": 30, "timeout_ms": 2500,
  "parameters": { "name": "loom", "intensity": 0.48, "duration_ms": 1900 }
}
```

**Production protocol (likely):** binary packets, COBS framing, CRC-32C, protocol version, session
ID, sequence number, command ID, priority, deadline, fixed-size payloads where practical, acks,
idempotent retries.

**Command lifecycle:** RECEIVED → ACCEPTED/REJECTED → EXECUTING → COMPLETED/ABORTED. Duplicate
command IDs must not repeat physical actions.

**Heartbeat / link health:** bidirectional, ~every 50 ms; Teensy enters degraded state after missed
heartbeats; new locomotion disabled after a short timeout; smooth deceleration + safe stance after
~300 ms without valid comms; fresh handshake after reconnect. *Exact timeouts must be measured.*

**Teensy loop rates (independent of comms):** motion loop ~250 Hz; telemetry ~50 Hz; heartbeat
~20 Hz; high-level commands event-driven. **The motion loop must never block on USB, parsing,
logging, or Jetson responses.**

---

## 11. Safety Priorities

```
100  physical emergency stop
 90  servo fault / instability / brownout / thermal fault
 80  direct STOP command from Timothy
 60  stabilization and defensive posture
 40  direct locomotion command
 30  speech-synchronized gesture
 10  idle movement
```

STOP bypasses the normal command queue. A **physical e-stop circuit must disable servo power
independently of the Jetson and Teensy software.** The servo power rail must be electrically separate
from the Jetson supply.

---

## 12. Physical Design & Torso CAD

**Look — "Mecha Azmodan":** burnt-orange/infernal-red armor; charcoal/gunmetal/black structure;
exposed industrial mechanisms; visible springs/joints where appropriate; black braided wire sleeving;
amber furnace lighting; hollow exhaust stacks that double as real cooling exhausts; overbuilt/forged/
armored/hot-running. Avoid sleek consumer-robot styling. Existing legs (orange/gray/black) are
consistent.

**Torso houses:** Jetson Orin NX, Teensy 4.1, regulators/power distribution, USB, audio amp, speaker,
fuse/service hardware, cooling ducts, lighting electronics, cable routing. Battery kept low in the
chassis for center of gravity.

**Structural enclosure with removable armor** (not one decorative shell). Initial envelope ≈
**205 L × 165 W × 125 H mm.**

*Components:* chassis adapter plate; structural electronics base; removable electronics sled; L/R
armor shells; front furnace panel; rear service panel; hollow exhaust stacks; cable covers/strain
relief; optional lighting diffuser.

*Internal reservations (start):* Jetson keep-out ~125×115×55 mm; Teensy ~75×30×20 mm; power/audio bay
~80×60×35 mm; cable bend clearance 20–25 mm near connectors.

*Print assumptions:* Prusa MK3, build ~250×210×210 mm, PETG, 0.4 mm nozzle, 0.20 mm layers;
structural walls 3.2–4.0 mm; armor walls 2.4–3.0 mm; 4–5 perimeters; structural infill 25–35%;
cosmetic infill 10–15%; M3 fasteners + heat-set inserts; sliding clearance ~0.4–0.6 mm/side; panel
seam ~0.4 mm. Armor is non-structural/removable; the electronics sled is serviceable without removing
legs.

*Cooling:* low/front-side cool-air intake; clear airflow to the Jetson's stock heatsink/fan; internal
anti-recirculation shroud; upper/rear exhaust; exhaust stacks as real vents; initial open vent area
~2500–3000 mm²; temperature testing under sustained LLM inference.

*CAD order:* **Rev 0** — slotted mounting template, chassis clearance outline, Jetson mock volume,
leg-sweep test. **Rev A** — structural base, electronics sled, plain ventilated cover. **Rev B** —
Mecha Azmodan armor, furnace panel, exhaust stacks, lighting. **Rev C** — fit/thermal/tolerance
corrections, print-ready STL package.

*Chassis measurements still required:* flat-top mounting length; flat-top mounting width; hole-center
spacing L–R and F–B; mounting-hole diameter; max safe torso width before leg interference. Consider a
**universal slotted adapter plate** so the torso isn't tied to one hole pattern.

---

## 13. Roadmap

- **0.2** — Lore-informed text personality, structured output, emotional state, memory, gesture
  simulation, one-click Windows launcher. *(current)*
- **0.3** — Wake word "Azmodan", always-on detector, VAD, faster-whisper transcription, mic
  selection, audible acknowledgement, local voice output, self-audio suppression.
- **0.4** — Persistent approved memories, relationship model, better emotional continuity,
  response-rating + dataset collection.
- **0.5** — Streaming dialogue, streaming TTS, voice DSP, word-level gesture sync, barge-in.
- **0.6** — Jetson service architecture, behavior executive, motion-link service, Teensy protocol
  simulator.
- **0.7** — Teensy firmware, command parser, heartbeats, gesture library, gait + IK, fault-injection
  tests.
- **0.8** — Torso CAD, electronics integration, thermal testing, speaker/amp integration.
- **0.9** — Jetson deployment, physical Jetson–Teensy integration, unloaded-servo bench tests,
  supported-stance tests, tethered walking tests.
- **1.0** — Fully local wake-word conversation, lore personality, memory + emotional state,
  expressive voice, whole-chassis gestures, reliable locomotion, safe dual-brain architecture, no
  cameras.

---

## 14. Development Principles

Build a reliable character system before fine-tuning. Do not train a foundation model from scratch —
use prompting, memory, state, and structured output first; collect and rate hundreds of real
conversations before any LoRA tuning. **Never let the LLM own safety-critical hardware**; preserve a
strict boundary between generative intent and deterministic control. Keep the model-provider
interface replaceable. Optimize for perceived responsiveness, not only model size; benchmark both 9B
and smaller models on the Jetson. Prefer sequential heavy workloads (listen → transcribe → think →
speak). Measure latency rather than guessing. Log command lifecycle, retries, missed heartbeats, and
max latency. Test failure deliberately before untethered walking. Preserve serviceability, cooling,
and center of gravity in every mechanical decision.

---

## 15. Assistant Working Agreement

When assisting on AZMO: maintain the dual-brain architecture; never suggest direct LLM-to-servo
control; don't assume cameras in v1; treat qwen3.5:9b as the current PC PoC, not the final model;
preserve local operation as a core goal; prefer concrete engineering recommendations; distinguish
verified facts from assumptions; ask only for measurements or decisions that can't reasonably be
inferred; make generated code runnable and modular; for CAD, produce parametric source **plus**
separate printable STL parts; when discussing Azmodan, distinguish canon vs. interpretation vs.
original AZMO dialogue; keep hardware safety, thermal design, serviceability, and fault behavior
central.

---

## 16. Open Items / Needed Inputs

- Chassis measurements for the torso adapter plate (see §12).
- Expressive local TTS: **decided** — XTTS v2 voice clone (of the Azmodan game voice) as the base,
  with the azmo-voice DSP chain on top; implemented in the Claude Edition. Remaining: run Demucs to
  produce a cleaner clone reference, tune the DSP params against the game reference, and benchmark
  XTTS latency vs. a distilled piper voice for real-time Jetson use.
- Heartbeat/timeout values to be measured and tuned on real hardware.
- Jetson vs. smaller-model latency benchmarks under the full listen→think→speak pipeline.

---

*This brief is a living document — update it as the project evolves.*
