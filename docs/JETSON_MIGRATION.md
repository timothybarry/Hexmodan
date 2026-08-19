# Getting AZMO onto the Jetson Orin NX 16 GB

Target from the brief: Waveshare Jetson Orin NX 16 GB (SKU 24222), running the
cognition stack, talking to a Teensy 4.1 over USB CDC.

This is the plan for getting there from a working Windows prototype. Read
`docs/PERFORMANCE.md` first — the bandwidth argument there is the whole basis
for the model-sizing decisions here.

---

## 1. The hardware delta, honestly

| | RTX 3080 Ti (dev PC) | Orin NX 16 GB | Ratio |
|---|---|---|---|
| Memory bandwidth | ~912 GB/s | ~102 GB/s | **9x less** |
| Memory | 12 GB dedicated VRAM | 16 GB **shared** with the OS | — |
| FP16 compute | ~68 TFLOPS | ~25 TFLOPS (sparse-rated higher) | ~3x less |
| Board power | 350 W | 10-25 W | 15x less |
| CPU | i7-8700K, 6 desktop cores | 8x Cortex-A78AE | ~3x less per core |

Two of these matter far more than the others.

**Bandwidth decides LLM speed.** Token generation must read every model weight
from memory for every token produced (see PERFORMANCE.md §2). So:

```
tokens/sec (ceiling) = memory bandwidth / model size
```

| Model | Size at Q4 | Ceiling on Orin NX | Realistic (~65%) |
|---|---|---|---|
| qwen3.5:9b | 5.5 GB | 18 tok/s | **11-13 tok/s** |
| A 3B model | 1.9 GB | 54 tok/s | **32-35 tok/s** |
| A 1.5B model | 1.0 GB | 102 tok/s | 60+ tok/s |

A 150-token reply on the 9B: **12+ seconds of generation alone**. The same reply
on a 3B: about 4 seconds. This is arithmetic, not a benchmark — but benchmark it
anyway, as the brief requires.

**Memory is shared, not dedicated.** The OS, the display stack (if any), the
camera pipeline and your models all draw from the same 16 GB. There is no
separate VRAM pool to overflow into — when it's gone, it's gone.

---

## 2. Memory budget

| Consumer | With 9B | With 3B |
|---|---|---|
| JetPack 6 + Ubuntu, headless | ~2.5 GB | ~2.5 GB |
| LLM weights (Q4) | 5.5 GB | 1.9 GB |
| KV cache at 4096 ctx | 0.7 GB | 0.3 GB |
| TTS | ~3.0 GB (XTTS) | ~0.1 GB (Piper) |
| Whisper small.en | 0.5 GB | 0.5 GB |
| Motion link, behaviour, logging | ~0.5 GB | ~0.5 GB |
| **Total** | **12.7 / 16 GB** | **5.8 / 16 GB** |

The 9B column fits — and is still the wrong choice, because it is slow, leaves
no headroom for the camera work in the roadmap, and gives back nothing you can
hear. Personality here comes from the system prompt, memory store and structured
output, not parameter count. Prove that with `azmo eval` before assuming
otherwise.

---

## 3. Subsystem by subsystem

### Brain (LLM) — straightforward

Ollama has arm64 builds and runs on Jetson, but can silently fall back to CPU if
CUDA isn't wired up correctly — check `azmo doctor` output, and confirm
generation speed matches the table above rather than trusting that it started.
`llama.cpp` built with CUDA for the Jetson is the better-supported fallback, and
`OllamaProvider` is already behind the `LLMProvider` interface, so swapping is a
new provider class and a config line.

**Action:** benchmark 9B vs 3B vs 1.5B on the actual board. Decide on measured
tokens/sec plus an `azmo eval` personality score, not on vibes.

### Ears — needs restructuring, and this is a win

Running `small.en` on the Orin's A78AE cores will be far slower than on the
i7. But the bigger problem is architectural: today Whisper transcribes
*continuously* to detect the wake word. On a battery-powered robot, running a
transcription model around the clock is indefensible.

**Action:** implement a real hotword engine behind the existing `WakeDetector`
interface — openWakeWord or Porcupine, tens of thousands of parameters, ~1% of
one core, always on. Whisper then only runs *after* wake, on one utterance. This
cuts idle power dramatically and improves wake accuracy (a purpose-trained
"Azmodan" model beats phonetic matching on ASR output — see the wake-word notes
in HANDOFF.md for how much work that currently takes).

Whisper itself should move to the GPU. **Note the cuDNN defect in HANDOFF.md** —
on Jetson, cuDNN comes from JetPack rather than pip, so this may resolve itself,
but verify early rather than discovering it during integration.

### Voice — the hard constraint

XTTS is compute-heavy at both stages. Expect **25-50 seconds** per reply on the
Orin against 7-10 on the 3080 Ti. Non-viable as currently used. Three real
options:

**a) XTTS with streaming inference.** XTTS exposes `inference_stream()`, which
yields audio chunks as they are generated instead of returning a finished
waveform. Total time is unchanged, but *time to first sound* could be 3-5
seconds, and playback covers the rest. This preserves the cloned voice exactly,
which is the thing you most want to keep.

> **Design consequence:** the DSP chain currently peak-normalises the whole
> reply at once (`apply_azmo_voice` divides by the maximum absolute sample).
> That is impossible when you're emitting audio before you've generated it.
> Streaming requires switching to a fixed gain with a look-ahead limiter.
> Per-chunk peak normalisation would make loudness pump between chunks — which
> is very likely what made XTTS's own internal splitting sound inconsistent.

**b) Train a Piper voice on the same source audio.** The brief already names
Piper as the intended expressive local TTS, and `PiperSpeech` is implemented.
Piper is a VITS model: ~50 MB, real-time on CPU, trivial on Orin. It cannot
clone from a handful of clips like XTTS — it needs a fine-tuning dataset,
roughly 30-60 minutes of clean single-speaker audio with transcripts. If enough
usable source audio exists, this is the *right* long-term answer: same voice,
a fraction of the cost, and it frees ~3 GB.

**c) espeak-ng.** Already implemented, always available, robotic. Keep it as the
fallback the brief describes, not as the plan.

**Recommended path:** (a) to get running, (b) as the real target. Do not defer
(b) on the assumption that (a) will be fast enough.

### Motion — already designed for this

`motion_link.py` already speaks the Jetson-to-Teensy command envelope and
lifecycle. This is the least risky part of the port. The brief's rule holds
absolutely: **the LLM process must not own the serial connection.**

### Power — a non-issue, and a different problem

None of `gpu.py` applies. Set `gpu.power_limit_watts: null`. The desktop PSU
transient problem cannot occur on a 25 W module.

What replaces it: `nvpmodel` power modes (10 W / 15 W / 25 W) and `jetson_clocks`.
25 W needs the active cooling the brief already reserves space for. If AZMO runs
on battery, inference time is directly battery life — another argument for the
smaller model.

---

## 4. Projected latency

One turn, 3B model plus streaming XTTS, versus today:

| Stage | Windows now | Orin projected |
|---|---|---|
| Wake detection | continuous Whisper | ~0 ms (hotword engine) |
| End-of-speech | 700 ms | 700 ms |
| Transcription | 1-3 s | 1-2 s (GPU) |
| LLM generation | ~2 s | ~4 s |
| Time to first audio | 7-10 s | 3-5 s (streaming) |
| **Perceived total** | **~20 s** | **~9 s** |

Slower silicon, faster experience — because streaming and a hotword engine
attack *waiting*, which is what the user actually perceives. This is the brief's
"optimize for perceived responsiveness, not only model size" made concrete.

---

## 5. Migration order

Stage each step so it can be validated on the PC before the board is involved.

1. **Now, on Windows:** implement LLM streaming and streaming synthesis. Convert
   the DSP to fixed-gain plus limiter. This is the largest code change and it
   belongs where you can debug it. *(Wait for the new PSU — streaming puts the
   LLM and TTS on the GPU concurrently.)*
2. **Now, on Windows:** benchmark a 3B model with `azmo eval` against the 9B.
   Settle the personality question before the hardware forces it.
3. **Now, on Windows:** implement a hotword `WakeDetector`. It is a drop-in
   behind the existing interface and improves the desktop experience too.
4. **On the board, headless:** JetPack 6, NVMe (not SD — model load from SD is
   painful), Ollama or llama.cpp with CUDA verified, `azmo doctor` clean.
5. **On the board:** `azmo check`, then `azmo once`, then `azmo listen`. Measure
   each stage with the per-turn timing already in the listen loop.
6. **Voice decision:** measure streaming XTTS on real hardware. If time-to-first-
   audio exceeds ~5 s, start the Piper training dataset.
7. **Split into services** (brief §5). Until now everything is one process; on
   the robot, a voice crash must not take down the motion link.
8. **Teensy integration** last, per the roadmap.

---

## 6. What the code already gets right

The port is mostly configuration and new adapters, not a rewrite, because the
seams are in the right places:

- `LLMProvider` — swap Ollama for llama.cpp or TensorRT-LLM.
- `SpeechAdapter` — Piper is already implemented; a streaming XTTS adapter slots
  in beside it.
- `WakeDetector` — built specifically so a hotword engine can replace Whisper.
- `MotionLink` — already speaks the real Teensy protocol against a simulator.
- Config is Pydantic-validated, so a `config/azmo-jetson.yaml` is a supported
  deployment rather than a fork.

The two things that will need genuine rework are **streaming synthesis** (which
changes the DSP contract) and **service separation**. Both are worth starting on
the desktop.
