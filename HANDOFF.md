# AZMO — session handoff

**Last verified: 2026-07-31.** Anything below that is not dated has been checked
against the code as of that date.

> **Read `docs/DESIGN_LOG.md` first.** It carries the current goals and the
> reasoning behind them, newest entry first. *This* file is the operational
> detail: hardware state, environment landmines, and the specific traps that
> have each cost a debugging session.
>
> When the two disagree, the design log wins — and then fix this file.

Context for a fresh chat. The full charter is `AZMO_PROJECT_BRIEF.md`; the Jetson
analysis is `Jetson_Orin_Decision.md`.

### Why this file was wrong before

It drifted twice, in ways worth noticing because both are easy to repeat:

- It listed `torch.inference_mode()` **both** as a thing that natively aborts
  XTTS *and* as an applied mitigation. The code was never wrong; the prose was.
  Prose that describes code will rot unless a test pins it.
- It reported 0.2.6 and 58 tests when the repo was 0.2.8 with 117 passing. Both
  numbers were quotable and neither was checked.

Counts in this file are now stated as *run this to find out*, not as facts:

```powershell
azmo --version            # or: python -c "import azmo_mind; print(azmo_mind.__version__)"
python -m pytest -q       # authoritative test count
```

**Trap:** `__version__` reads *installed package metadata*, so after bumping
`pyproject.toml` it keeps reporting the old number until you re-run
`pip install -e .`. Seen on 2026-07-31: `pyproject.toml` at 0.2.9, the import
still saying 0.2.5. `pyproject.toml` is the only place a version is written —
when the two disagree, the install is stale, not the version.

Re-synced on 2026-08-02: both now say 0.2.10. It will drift again at the next
bump — the fix is one command, not vigilance:

```powershell
python -m pip install -e . --no-deps
```

## Where the code lives

- Live working repo (editable install, edits take effect on next launch):
  `AzmoBot Main\AZMO-Mind-Claude\`
- Portable backup zip: `AZMO-Mind-v0.2.5-Claude-Edition.zip`
- Two virtual envs exist. **`.venv312` (Python 3.12) is the real one** — it has the
  voice + listener stack. `.venv` (3.14) is legacy/unused; 3.14 cannot install the
  audio/ML wheels. Launchers prefer `.venv312` automatically.

## What works today (all on the PC)

- **Brain:** qwen3.5:9b via Ollama. Lore-informed Azmodan personality, SQLite
  memory, 8-dimension emotional state, structured JSON output (speech, emotion,
  one allowlisted gesture, voice direction). **A warm turn is ~2.5 s**, measured
  2026-08-02 — see the timing table below. This file previously said "5–20 s per
  turn" and that was never measured.
- **Voice:** XTTS v2 clone of the Azmodan game voice + a custom "azmo-voice" DSP
  chain. **This sounds right — do not redesign it.**
- **Ears (new, barely tested):** `azmo listen` — mic → webrtcvad → faster-whisper
  (small.en, CPU) → brain → voice. Wake word "Azmodan" via tolerant transcript
  matching (pluggable `WakeDetector` so a lean hotword engine can replace it on
  the Jetson).
- **Presence (new, 2026-07-30):** pre-rendered non-verbals — a slow exhale, a low
  growl — played *while he thinks*, and repeated every `sustain_gap_ms` for as
  long as the turn runs. He has no body yet, so audio is the only channel he has
  to look like he is thinking rather than broken. `azmo presence build` renders
  the pool through the same clone engine and seed as his speech; `azmo presence
  test` plays a simulated long think so you can hear whether it loops.
  **The pool ships empty — build it, then curate it by ear.**
- **Motion:** simulated only. `motion_link.py` speaks the real Jetson↔Teensy
  command envelope + lifecycle. No hardware yet, and deliberately not being
  worked on: legs are Paul's lead, post-POC.
- **Streamed delivery (new, 2026-08-02, OFF by default):** the model streams and
  chunks are rendered while it is still writing, with a prebuffer before the
  first word. `speech.stream_playback: false` until you have heard it *and* the
  new cooler and PSU are in — see the section below for why the second condition
  is not caution but a specific hardware interaction.
- `azmo check` = preflight (brain/voice/ears/presence/streaming/GPU).
  `azmo voices` = voice engine diagnostic. `azmo voicetune` = A/B render grid.

## Where a turn's time actually goes (measured 2026-08-02)

Real system prompt (~8.4 KB, 2055 prompt tokens), qwen3.5:9b, this box:

| | model load | prefill | generation | wall |
|---|---|---|---|---|
| prefix cache **hit** | 0.32 s | **0.06 s** | 2.06 s | **2.46 s** |
| prefix cache **miss** | 0.33 s | **8.94 s** | 2.27 s | 13.66 s |

Read three things off this:

1. **The 0.2.9 prompt-ordering fix is worth ~9 s per turn.** It shipped with
   reasoning but no number. This is the number, and it is why
   `tests/test_prompt_lore.py` pinning the ordering matters more than it looks.
2. **The brain is not the bottleneck.** Synthesis is. Latency intuitions in this
   file were built on the opposite assumption and are worth re-checking.
3. **Streaming can only overlap ~2 s of LLM time** — the generation phase. Its
   larger win is that playback stops waiting for every chunk to render.

**Not measured: XTTS render time per chunk.** It needs the voice model on the
GPU, which is the load this machine currently cannot be trusted with. Measure it
first after the teardown: it is the number that sets
`speech.stream_prebuffer_chunks`.

## Streaming: why it is off, and what to check when you turn it on

Two conditions, neither of which more code can clear.

**You have to hear it.** Chunk seams are an ear question, exactly like curating
the presence pool. Build, listen, decide.

**The machine has to be trustworthy first.** Streaming runs the LLM and XTTS
*concurrently*. That is the precise back-to-back GPU load `gpu.stagger_ms` was
added to break apart — streaming does not merely skip that mitigation, it does
the opposite of it. Do not switch this on before the new cooler and PSU are in
and the box has been stable through a real session.

When you do turn it on, the per-turn line reports **stalls**. Zero means the
prebuffer is deep enough. Any stall means he paused mid-sentence, which is the
one thing this is not allowed to do — raise `speech.stream_prebuffer_chunks`.

**Known limitation:** streamed replies render with the default `VoiceDirection`,
because `voice` is declared *after* `speech` in `AzmoResponse` and a chunk has to
be rendered before its direction exists. Two locked settings already suppress
most of that direction (`heaviness_variation`, and `effective_pace` keeping only
~30% of the model's pace swing), so what is actually lost is a few percent of
tempo. Declaring `voice` before `speech` would fix it for a few dozen tokens of
delay — but that changes what the model writes, which is an ear question, so it
is deliberately left open.

**Do not run the DSP per chunk without a `GainAnchor`.** `apply_azmo_voice`
peak-normalises three times per call, so per-chunk DSP lifts a quiet clause to
match a loud one and the reply pumps at every boundary. `voice_dsp.GainAnchor`
holds one gain frame across the reply. `tests/test_dsp_anchor.py` fails if it
stops working.

## The voice settings (locked — preserve these)

Tuned over many iterations to a take the user approved ("C_clean_noworld").
In `config/azmo.yaml`:

- `clone_temperature: 0.26` (low = consistent; higher caused "drunk Englishman"
  takes with wandering accents)
- `clone_seed: 20260726` — fixed and non-zero, so the same line renders
  identically every time. A good take stays good.
- `clone_split_text: false` (one pass per reply; splitting made some sentences
  demonic and others generic)
- `speech.dsp.use_world: false` — **critical.** The WORLD vocoder re-synthesis was
  the cause of the "underwater" sound. Pitch/formant ratios are 1.0 (no shift).
  The clone is already deep; character comes from EQ + grit + sub layers over an
  untouched primary voice.
- `intensity_bias: 0.67`, `heaviness_variation: 0.05` — a *locked register*. High
  variation made him drift "too human" on calm lines.
- Clarity stack: `mud_cut` 330 Hz −4 dB, `clarity` 3 kHz +5 dB, `presence` 7 kHz
  +4 dB, near-dry reverb (0.03–0.07). Guttural grit at −13 dB.

Wanting it *even less* variable = `heaviness_variation: 0.0`. Don't reintroduce
`use_world` or pitch shifting.

### The 250-character trap (cost three crashed sessions — read this)

XTTS v2 generates English inside a **hard 250-character window**. Handing it
more with splitting disabled does not truncate politely: the generation loop
overruns and aborts the whole process natively (Windows exit code
-1073740791 / 0xC0000409, no traceback, nothing catchable). AZMO's ~100-word
replies cross 250 characters routinely.

`speech.clone_split_text: false` stays off *on purpose* — XTTS's internal
splitting re-rolls the sampling per sentence, which is what made some sentences
demonic and others generic. Instead `split_for_xtts()` in `speech.py` chunks the
reply before the model sees it, and **re-seeds before every chunk** so each is
the same sampling roll. That re-seeding is the whole reason our splitting keeps
the character where the model's did not. Chunks are joined with a 120 ms breath
and the DSP runs once over the concatenation.

**Never raise `clone_max_chars` above ~230, and never pass long text with
`enable_text_splitting=False`.**

### Other things that abort XTTS natively (all learned the hard way)

- `torch.inference_mode()` around synthesis. Too strict — XTTS generates through
  HuggingFace code that hard-errors on inference-tagged tensors. Use
  `torch.no_grad()`.
- Touching `torch.backends.cudnn.*` at all. Reading or setting those properties
  forces cuDNN to initialise, and this box has a torch/CTranslate2 cuDNN
  mismatch (`Could not load symbol cudnnGetLibConfig. Error code 127`).
- Warming the voice **before** the ears. torch's cuDNN only works when it
  inherits the copy CTranslate2 already loaded. `cli.listen` has a comment
  saying so — do not "optimise" that order.

`DIAGNOSE_VOICE.bat` locates any future native abort: it prints and flushes a
marker before each step, so the last marker printed is the crash site.

### KNOWN DEFECT: cuDNN version mismatch in `.venv312` (worked around, not fixed)

**This is an environment problem, not a code problem.** Nothing in `azmo_mind`
is at fault, and it is still present.

`torch` and `ctranslate2` (pulled in by `faster-whisper`) both depend on cuDNN,
but only one copy can be loaded per process. The installed cuDNN is **version 8**;
`torch 2.5.1+cu121` was built against **version 9** and calls
`cudnnGetLibConfig`, which does not exist in v8. The loader reports
`Error code 127` (ERROR_PROC_NOT_FOUND) and the process aborts when XTTS reaches
its convolution-heavy vocoder.

Worked around with `speech.clone_disable_cudnn: true`, which makes torch use its
own convolution kernels. Consequences of leaving it:

- Any `pip install` in that venv can reshuffle it. **Snapshot before touching
  anything: `pip freeze > working-env.txt`.**
- `listener.whisper_device: cuda` will hit the same wall from the other side.
  Whisper is on CPU partly for this reason.
- The vocoder runs somewhat slower on the fallback kernels.

**Do not attempt the real fix casually.** The pins in the `clone` extra
(`coqui-tts==0.24.2` + `transformers==4.42.4` + torch 2.5.1+cu121) are fragile
and took real effort to land. When it is worth doing: freeze the environment,
run `DIAGNOSE_VOICE.bat` to capture actual versions, then make **one** change —
most likely either `ctranslate2>=4.5` (which wants cuDNN 9, matching torch) or
pinning `nvidia-cudnn-cu12==9.1.0.70`. Verify with `azmo say` before anything
else. This also matters for the Jetson port, where Whisper must move onto the
GPU — see `docs/JETSON_MIGRATION.md`.

## Half-duplex: why AZMO cannot hear himself (0.2.6)

This was a real feedback loop, not a theoretical one: AZMO says "Azmodan"
constantly, so **any** of his own audio reaching Whisper is a wake word and he
answers himself forever. Three independent guards, all in `listener.py`:

1. **Gate at the capture callback.** `MicStream._open_gate` is checked inside the
   sounddevice callback; while shut, frames are discarded before they are ever
   queued. `Listener.deaf()` wraps the whole think-and-speak phase in `cli.listen`.
2. **`post_speech_cooldown_ms` (700).** The gate stays shut past the end of
   playback — the speaker tail and room reverb arrive *after* `PlaySync()`
   returns. Then the buffer is drained, then the gate reopens.
3. **`EchoGuard`.** Remembers his last reply for `echo_guard_window_ms` and
   discards transcripts whose content words are ≥60% his. Stopwords ignored.

Also: bounded frame queue, timeouts on every blocking read, `min_utterance_ms`
to drop blips, `follow_up_timeout_ms` on the bare-wake-word path, and a
`ListenerError` if the mic dies instead of an infinite silent spin.

**Do not set `listener.always_on: true`** — it removes the wake word as a guard.
`azmo check` warns if it is on.

## Wake-word recognition (the hard part)

"Azmodan" is not in Whisper's vocabulary, so it renders the sound as whatever
ordinary English it resembles. Real transcripts from a live session:
`"As Madam, introduce yourself"`, `"As been in, introduce yourself"`. A list of
exact spellings can never be complete. Current approach, in order:

1. **Bias the decoder** — `whisper_initial_prompt: Azmodan` is passed as
   `hotwords` (faster-whisper >= 1.0.2) or `initial_prompt`, so the model is far
   more likely to emit the name correctly in the first place. `whisper_beam_size: 3`
   also helps materially on rare proper nouns.
2. **Capture the whole word** — `vad_aggressiveness` dropped 2 -> 1 and
   `pre_roll_ms` raised 300 -> 500. A soft "Az" onset often fails to trip the
   VAD, and a decapitated wake word is an unrecognisable one. This is the likely
   cause of the "As been in" mangling.
3. **Match phonetically, not by spelling** — `phonetic_key()` is a Soundex-style
   consonant skeleton; "Azmodan" and "As Madam" both reduce to `A2535`. Plus a
   `difflib` similarity ratio at `wake_fuzzy_threshold` (0.72).

**Important constraint:** the phonetic pass only looks at the first ~3 words of
an utterance (allowing up to 2 filler words, so "Hey Azmodan" works). Fuzzy
matching anywhere in a sentence would false-wake constantly. For the same
reason, `_wake_variants` must only contain spellings that are *not* ordinary
English — "as madam" and "as modern" are deliberately excluded from it, because
that list is matched anywhere in the sentence.

Verified: 0 false wakes across 25 ordinary sentences; all observed manglings
except "as been in" now wake him.

**`azmo hear`** is the diagnostic: transcribes only, shows the closest-matching
span, its similarity score and phonetic key, and whether it woke. No LLM or
voice model loaded, so iterating on these settings is fast. If a specific
mangling keeps slipping through, add it to `listener.extra_wake_variants`.

## The hardware problem: thermal shutdown, not (only) the PSU

**Corrected 2026-07-31. The previous diagnosis in this file was wrong.** It named
GPU power transients tripping the aging Corsair RM750x as *the* cause. That was a
defensible read of ambiguous evidence, and it is not what is happening.

**What is actually happening: the CPU thermally shuts down.** BIOS reports a
thermal shutdown on the following boot. Light loads drive the i7-8700K to 100 C —
exactly its Tjmax. The AIO's tubes are cold while this happens; no coolant is
moving.

**Why the crash report missed it.** `azmo_crash_report.txt` searches the Windows
Event Log for thermal events, finds none, and concludes "not thermal". That
inference does not hold. A THERMTRIP is the CPU's own protection cutting power at
the silicon level — it never reaches the OS to be logged. Windows records only
the aftermath: Kernel-Power 41, "rebooted without cleanly shutting down", which
is the *same* signature a PSU brownout leaves. The absence of thermal events was
never evidence against thermal. **Do not re-derive the old conclusion from that
report.**

**Likely root cause: the AIO is at end of life** (~7 years in service). Coolant
permeates out through the tubing, air accumulates and can stall or cavitate the
pump under load, bearings wear, paste dries. That fits the progression better
than any sudden failure: 8 WHEA "fatal hardware error" events from 2026-05-17
through 07-04 increasing in frequency, then outright thermal shutdowns in late
July. Those WHEA events are probably not a separate mystery — they look like the
same cooler degrading past a threshold.

**The 5.0 GHz OC is a load, not a cause.** Turbo Max x50 at ~1.36 V Vcore (manual
OC or the board's Multi-Core Enhancement) puts the 8700K near 150-200 W all-core
against a 95 W TDP. It has been stable for seven years, so it did not change and
did not cause this — but it is the load a worn cooler can no longer carry, which
is why the machine idles fine and hits Tjmax the moment real work starts.
Dropping to stock is a valid stopgap for working before the swap. It is not a fix.

**Pump RPM could not be read, and that proved nothing.** Neither HWiNFO (Nuvoton
NCT6793D section shows only Chassis1/CPU/Chassis2/Chassis3, all ~1000-1200 RPM =
fan range) nor Corsair Link showed a pump or any Corsair cooler device — Link was
reading motherboard sensors only. Benign explanations exist: a non-`i` Corsair
model has no USB monitoring at all, newer Platinum/Pro/Elite units need iCUE
rather than Link, and the internal USB header may simply be unplugged. Absence of
a reading is not evidence of a dead pump. The decisive tests are physical — touch
the pump housing for vibration, feel both radiator tubes for a gradient.

### The fix, in progress 2026-07-31

Both parts going in during one teardown:

- **New AIO cooler** (delivered). The actual fix. Installing it includes fresh
  paste, so it covers every remaining candidate at once — dead pump, dying pump,
  air in the loop, failed mount, dried paste.
- **Montech Century II 1050 W ATX 3.1 PSU.** Kept deliberately as an *upgrade*,
  not a fix. A 3080 Ti draws 350 W stock with transient excursions toward
  450-500 W, on top of a 150-200 W overclocked CPU; 750 W was thin, and a
  seven-year-old supply has derated (capacitor aging degrades transient response
  first). ATX 3.1 is specified to ride out those excursions.

Doing both at once means the fix cannot be attributed to one part. Deliberate
trade: one teardown, and both components were due regardless.

**After the swap, revisit `gpu:` in `config/azmo.yaml`.** That entire section
(except `empty_cache_after_speech`) exists to mitigate the power-transient
diagnosis, and its in-file comment still states that diagnosis as settled — it
needs the same correction this section just got. With a thermally sane CPU and a
1050 W ATX 3.1 supply, retest `power_limit_watts: null` and a smaller
`stagger_ms`. The cap costs GPU performance for a problem that may never have
been the real one.

**Still open:** whether the WHEA events resolve with the cooler. If they persist,
XMP is next — the G.Skill F4-3200C16 kit runs its XMP profile at DDR4-3200
16-18-18-38 / 1.35 V. Test at JEDEC defaults, one variable at a time, and only
once thermals are known good.

### Mitigations now in the software (0.2.6)

- `azmo_mind/gpu.py` + the `gpu:` config section. `START_AZMO_VOICE.bat` asks for
  admin once and applies `power_limit_watts: 250` (stock 350); it is restored on
  clean exit, on reboot, by `RESTORE_GPU_POWER.bat`, or by `azmo gpu restore`.
  **Always remind the user to restore full power before gaming.**
- `gpu.stagger_ms: 300` between the LLM finishing and XTTS starting — the two
  back-to-back inferences were the sharpest current ramp in the pipeline.
- Heavy components warm one at a time at startup, so the ~2 GB XTTS load no
  longer lands right after the first LLM turn.
- `torch.no_grad()` around synthesis (**not** `inference_mode` — see the section
  above; an earlier revision of this file claimed the opposite here and it was
  wrong. The code in `speech.py::_inference_context` has always been correct and
  documents why), `empty_cache` between turns.
- Whisper pinned to CPU with a thread cap; `OLLAMA_MAX_LOADED_MODELS=1` and
  `OLLAMA_NUM_PARALLEL=1` set by the launcher.
- `speech.engine: sapi` remains the no-GPU dev mode.

## What the user wants next

Goals and their reasoning now live in `docs/DESIGN_LOG.md`. In short: this is a
**POC** — a convincing enough Azmodan on the desktop that buying the Jetson is an
obvious call. Judged by ear. The ≤4 s latency figure is a general goal, not a
requirement: **dead air is the enemy, not elapsed time.**

1. **Install the new AIO cooler and the Montech 1050 W PSU** (one teardown,
   2026-07-31 — see the hardware section above; the cooler is the fix, the PSU is
   an upgrade). Then reconsider whether the power cap is wanted at all
   (`gpu.power_limit_watts: null`) and correct the stale diagnosis in the `gpu:`
   comment block in `config/azmo.yaml`.
2. **Build and curate the presence pool** — `azmo presence build`, then listen
   and delete anything that sounds like words or sounds cut off. Then
   `azmo presence test --seconds 12` and check it does not read as a loop.
3. **Real-world soak test of hands-free mode.** Does he ever self-trigger? Note
   presence adds a new way to fail here: his own breath reaching the mic. Both
   fire points are inside a deaf window by construction, but this has not been
   tested against real speakers in a real room. If the mic catches him, raise
   `post_speech_cooldown_ms` toward 1200 first.
4. **Judge streamed delivery by ear** (built in 0.2.10, shipped off). Turn on
   `speech.stream_playback` *after* the teardown, listen for seams, and watch the
   stall count. While you are there, measure XTTS render time per chunk — it is
   the number that sets the prebuffer, and it is the last unmeasured piece of the
   latency picture.
5. **`azmo compare` to settle model size — upward.** Latency no longer blocks a
   larger model; character coherence is the metric. Now better supported than
   before: a warm turn is 2.5 s, so there is real headroom to spend.
6. Chase the 8 unexplained WHEA events (RAM/XMP), which the power story does not
   account for.

## Hard-won gotchas — don't repeat these

- **All `.ps1` / `.bat` files must be pure ASCII.** An em-dash broke Windows
  PowerShell parsing (`string is missing the terminator`).
- Pinned, fragile ML versions in the `clone` extra:
  `coqui-tts==0.24.2` + `transformers==4.42.4` (newer coqui-tts 0.27.5 is broken —
  it needs both a new and a deleted transformers symbol). torch 2.5.1+cu121 and
  `torchaudio` must be installed **explicitly and first**.
- `webrtcvad` needs a C compiler → use **`webrtcvad-wheels`** instead.
- Temp WAVs must be a **path that does not exist yet** (not `mkstemp`), or Windows
  file locks break SAPI/playback.
- Ollama's structured output does not enforce numeric ranges → `coerce_response_payload`
  clamps out-of-range values; `sanitize_speech` strips leaked JSON so AZMO never
  reads field names aloud; `salvage_embedded_fields` recovers a leaked gesture.
- Config is read at launch only — a running chat window won't see edits.
- **Anything that plays audio must sit inside `listener.deaf()`.** This used to
  be true only of his speech; presence added a second sound source. The thinking
  track is already inside the existing deaf window; the wake-word breath opens
  its own, because it fires while the mic is otherwise live and waiting for your
  command. Without that gate, webrtcvad trips on his own breath and Whisper hands
  back whatever it makes of it *as the command*.
- **Prompt ordering is load-bearing, and now it has a price tag.** `prompts.py`
  is split into `static_prefix()` and `volatile_suffix()` for prefix caching: the
  cache is only reusable up to the first differing byte, so volatile content
  above the lore re-prefills all of it every turn. Measured 2026-08-02: 2055
  prompt tokens prefill in **0.06 s on a hit and 8.94 s on a miss**. Getting this
  wrong costs ~9 s per turn. `tests/test_prompt_lore.py` pins the ordering. If
  you add to the prompt: fixed-for-the-session goes in the prefix,
  changes-per-turn goes in the suffix, never interleaved.
- **Schema field order is load-bearing too (0.2.10).** `AzmoResponse.speech` is
  declared first, and streaming depends on it: Ollama builds its grammar in
  declaration order, so speech-first is what lets the text reach XTTS before the
  gesture and voice metadata. Reordering it would silently restore the old serial
  latency without breaking anything visibly. `tests/test_stream_json.py` pins it.
