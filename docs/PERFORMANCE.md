# AZMO performance, power, and why the PC crashes

Written to answer four questions:

1. Why does this crash the machine when games on ultra do not?
2. Where is the bottleneck in the pipeline?
3. What can be made more responsive without provoking a crash?
4. Will this run on a Jetson Orin 16 GB?

---

## 1. Why AI crashes a PSU that games do not

This is the least intuitive part, and it is not a question of "how hard" the
work is. Games are a *heavier average* load than AZMO. They are also a much
*gentler* load in the way that matters to a power supply.

**A game is a smooth load.** Rendering a frame spreads work across many
different parts of the GPU - geometry, rasterisation, texture units, shader
cores - and each frame ends at a natural boundary. Power draw is high, but it
is high *continuously*. The current the card pulls looks like a plateau.

**LLM decoding is a square wave.** Generating each token is one large matrix
multiply that lights up every streaming multiprocessor at once, then a short
sequential step where the GPU is nearly idle, then the next token. At ~79
tokens/second, that is the card slamming between near-idle and full tensor-core
load about eighty times a second. XTTS does the same thing, then finishes with
a convolution-heavy vocoder pass that hits different units again.

So the two workloads look like this:

| | Average draw | Rate of change (di/dt) | Idle gaps |
|---|---|---|---|
| Game on ultra | High, steady | Moderate | Every frame |
| LLM + TTS | Moderate | **Extreme** | Thousands per second |

A power supply's over-current protection trips on **peak** current, not
average. Bulk capacitors are what absorb a sudden demand spike before the
regulator can respond - and capacitors are the component that ages worst. An
eight-year-old RM750x can hold a rock-steady 500 W plateau all day and still
trip on a 100 W -> 350 W -> 100 W oscillation, because for a few microseconds
the transient demand is far above the average.

There is no BSOD because the rail collapses faster than Windows can react to
anything. The machine is simply not powered any more.

This is exactly the problem ATX 3.1 was written to fix: it *requires* a supply
to ride out 200% power excursions lasting 100 microseconds. The Montech Century
II 1050 W is specified for this. That is why it is the real fix and everything
here is a mitigation.

**In short: games are a marathon, AI inference is interval sprints. The old PSU
can run a marathon.**

### The second failure mode: memory thrashing

The reported symptom - *fans spin up like an F1 car, Windows loses all
responsiveness, then a hard crash* - is two problems stacked, and the first one
is ours to fix.

A 12 GB card running a 9B model **and** XTTS is at the edge of its memory. When
it goes over, Windows' display driver does not report an error; it starts
**paging GPU memory out to system RAM across the PCIe bus**. That paging is
itself heavy GPU work, so the card pins at maximum load moving memory around
instead of doing anything useful. The desktop stops responding because the
compositor is starved of the same VRAM. Fans go to 100%.

And then that sustained maximum-load thrash - not the inference - is what the
tired PSU finally trips on.

So the fix is not only "use less power". It is **do not let the two models
overcommit the card in the first place**, which is a software problem with a
software fix. See section 4.

---

## 2. Where the time actually goes

Measured from the session logs, for one turn:

| Stage | Time | Notes |
|---|---|---|
| End-of-speech detection | 0.7 s | `end_silence_ms` - pure waiting, by design |
| Whisper transcription | 1-3 s | small.en, int8, CPU |
| **LLM model load** | **5.6-11.4 s** | **A bug. Should be 0. See below.** |
| LLM prompt eval | 1-3 s | ~2500-token prompt, re-read every turn |
| LLM generation | ~2 s | 153 tokens at 79 tok/s |
| Stagger | 0.3 s | Deliberate idle gap between GPU workloads |
| XTTS synthesis | 7-10 s | The single largest real cost |
| DSP chain | ~1 s | pedalboard, CPU |
| Playback | = length of audio | |

**Total: 20-25 seconds** from finishing your sentence to hearing him.

### The model was reloading every single session

`load_duration` appeared in the metrics of *every* logged turn. It should appear
once, at warm-up, and never again - `keep_alive` is set to 30 minutes.

The cause: `warmup()` sent `num_ctx: 4096` while `generate()` sent
`num_ctx: 8192`. Ollama keys a resident model on its runtime options, so those
were two different model instances. The warm-up loaded one, and the first real
turn evicted it and loaded the other. **The warm-up was not just useless, it was
actively harmful** - it doubled the load work and put a multi-gigabyte VRAM
allocation spike at the front of every session.

Fixed: warm-up now uses the same `num_ctx` as generation.

### Ranked by payoff

1. **Model reload** - 6-11 s/turn. Fixed.
2. **Shorter replies** - `max_spoken_words` 100 -> 70 and `max_output_tokens`
   320 -> 200. Every spoken word costs GPU time twice, once to generate and once
   to synthesize. This is the biggest remaining lever and it cuts *power* too.
3. **Smaller context** - `num_ctx` 8192 -> 4096. The measured prompt is ~2500
   tokens, so 8192 reserved a KV cache three times larger than anything used.
   Frees ~1 GB of VRAM, which is the difference between fitting and thrashing.
4. **Streaming synthesis** (not yet implemented, see section 5) - would cut
   *perceived* latency by roughly half.

---

## 3. The VRAM budget

On a 12 GB RTX 3080 Ti:

| Consumer | Before | After |
|---|---|---|
| qwen3.5:9b weights (Q4) | ~5.5 GB | ~5.5 GB |
| KV cache | ~1.5 GB (8192 ctx) | ~0.7 GB (4096 ctx) |
| XTTS v2 weights + activations | ~3.0 GB | ~3.0 GB |
| Windows desktop | ~1.2 GB | ~1.2 GB |
| **Total** | **~11.2 GB of 12** | **~10.4 GB of 12** |

The "before" column has under a gigabyte of slack, and XTTS's activation peak is
variable - a long reply can exceed it. That is when paging starts.

`azmo check` now reports this and warns when it does not fit. If it still says
TIGHT, in order of preference:

1. Lower `provider.context_tokens` further (2048 is fine for this prompt).
2. Use a smaller or more heavily quantized model (a 7B Q4 saves ~1.5 GB).
3. Set `speech.clone_device: cpu` - takes XTTS out of VRAM entirely, at the cost
   of much slower synthesis.

---

## 4. What has been changed

- Warm-up and generation now agree on `num_ctx`, so the model stays resident.
- `context_tokens` 8192 -> 4096; `max_output_tokens` 320 -> 200;
  `max_spoken_words` 100 -> 70.
- `azmo check` reports free VRAM and warns before the budget is exceeded.
- Every turn prints a `Turn: think Xs | voice+play Ys | total Zs` breakdown, and
  says so explicitly if the model reloaded.
- Existing power mitigations: `gpu.power_limit_watts` (250 W of 350 W),
  `gpu.stagger_ms` between the LLM and the voice, VRAM released after each line,
  Whisper pinned to the CPU, `OLLAMA_NUM_PARALLEL=1`.

---

## 5. Not yet done (worth doing after the PSU)

**Stream the LLM and synthesize the first sentence early.** Today the pipeline is
strictly serial: the whole reply is generated, then the whole reply is
synthesized. Ollama can stream tokens. Starting XTTS on sentence one while
sentence two is still being written would cut perceived latency close to half.

It is deliberately *not* done yet, because it means the LLM and XTTS run on the
GPU **at the same time** - precisely the overlap being avoided while the PSU is
marginal. This is the first thing to revisit once the Montech is installed.

**Play chunk 1 while chunk 2 synthesizes.** Same idea, smaller scope, same
caveat. The text chunker already produces the pieces.

---

## 6. Will this run on a Jetson Orin 16 GB?

**The crashes say nothing about the Orin.** They are a desktop power-delivery
problem. An Orin module draws 15-60 W in total and is powered by a barrel jack;
it cannot reproduce this failure. Do not read the PC crashes as a verdict on the
Jetson.

Memory is workable but not roomy. The Orin's 16 GB is **unified** - CPU and GPU
share it, and the OS takes 2-3 GB:

| | Size |
|---|---|
| JetPack + Ubuntu | ~3 GB |
| qwen3.5:9b (Q4) | ~5.5 GB |
| KV cache at 4096 | ~0.7 GB |
| XTTS v2 | ~3 GB |
| Whisper small.en | ~0.5 GB |
| **Total** | **~12.7 GB of 16** |

It fits. Compute is the real constraint. An Orin NX has roughly a tenth of a
3080 Ti's memory bandwidth, and LLM decoding is bandwidth-bound. Expect **8-15
tokens/second against the 79 seen on the desktop** - so a 150-token reply moves
from ~2 s to 10-20 s of generation alone, on top of slower XTTS.

**Recommendation:** do not plan on a 9B model on the Orin. A 3-4B model
(qwen 3B/4B) at Q4 should give 25-40 tok/s and cuts memory to ~2.5 GB, leaving
comfortable room. The personality lives in the system prompt and the memory
store, not in the parameter count, so a smaller model loses less character than
it might seem - it is worth evaluating with `azmo eval` before assuming
otherwise.

Also worth knowing for the Orin:

- XTTS on Orin is slow. Piper (already an implemented engine here) is far
  lighter and a realistic on-robot voice, though it is not the clone.
- `whisper_device: cpu` should become GPU or a dedicated hotword engine - the
  Orin's CPU cores are much weaker than the i7's. The `WakeDetector` interface
  exists for exactly this swap.
- None of the `gpu.py` power machinery applies; `power_limit_watts: null`.
