# Jetson Orin NX 16GB — Buy / Wait Decision Brief

*Prepared for the AZMO project. Ties into AZMO_PROJECT_BRIEF.md §3 (hardware), §5–6
(services), and the roadmap (0.6–1.0 are the Jetson-dependent phases).*

## Bottom line

- The exact kit you're looking at — **Waveshare SKU 24222, Orin NX 16GB, US
  plug — is $1,185.99** (verified on the product page). I under-quoted earlier;
  that $675–$1,105 range covered cheaper 8GB / base configs, not this one.
- **You already own the Teensy 4.1, and optics are deferred** (no cameras in
  1.0). So this Jetson is essentially the single biggest remaining hardware
  purchase, and there's no vision pipeline competing for its memory/compute —
  which helps.
- **Everything in the AZMO stack can run on the Orin NX 16GB.** The catch is
  speed: your 9B LLM will run roughly **8–10× slower** than your RTX 3080 Ti,
  and real-time XTTS voice cloning is the tightest constraint.
- **Nothing you are building right now needs the Jetson.** It is only required
  for the untethered 1.0 build (roadmap 0.9–1.0).
- There is a **~$249 way to de-risk almost everything** before the $1,186 spend
  — see "A cheaper de-risk path" below.

## What it is (specs)

Orin NX 16GB: 1024 Ampere CUDA cores, 32 tensor cores, up to ~100 TOPS (INT8),
**16 GB LPDDR5**, **~102 GB/s** memory bandwidth, 10–25 W. For comparison your
RTX 3080 Ti has ~912 GB/s bandwidth (~9× more) and far more compute — which is
exactly why the desktop feels instant.

**Config note:** get the version that includes the **NVMe SSD (256 GB)**. You
need room for the OS + the 9B model + XTTS (~2 GB) + Whisper; the eMMC alone is
too small. Get **16 GB, not 8 GB** — the 9B (and comfortable headroom for ASR +
TTS sharing memory) needs it.

## Will your stack actually run on it?

- **LLM (qwen 9B via Ollama).** Fits in 16 GB at Q4 (~5.5 GB). But LLM speed
  tracks memory bandwidth, and the Orin NX has ~1/9th of your desktop's. Expect
  roughly **5–10 tok/s vs your ~80 tok/s**. A ~100-word reply that generates in
  ~2 s today would take **~15–25 s** on the Jetson. **This is the headline
  risk.** Mitigation: run a smaller on-device model (7B, or a 3–4B), which means
  re-validating that the Azmodan personality still holds at smaller size. Your
  brief already calls for benchmarking 9B vs smaller models on the Jetson.
- **Voice clone (XTTS).** XTTS supports streaming (~200 ms to first audio chunk,
  RTF ~0.25) *on big GPUs*; on the Orin NX expect it near or slightly above
  real-time. Workable with streaming inference, but the fallback is distilling a
  fast **Piper** voice (Piper runs with very low latency on Orin) at some cost to
  clone fidelity. This is the classic "clone quality vs on-device speed" trade.
- **ASR (faster-whisper), VAD (Silero), wake word, and the azmo-voice DSP
  (pedalboard / pyworld).** All run on Orin (aarch64 wheels or source builds).
  Not a concern.

## Timing vs your own principle

You said: nail the proof of concept before you spend, and earn it. Honest status:

- **Working on PC now:** LLM + cloned Azmodan voice + demonic DSP + gesture
  simulation + safety arbiter + motion-link protocol. (Roadmap 0.2, plus much of
  0.5's voice work.)
- **Not yet on PC:** wake word "Azmodan", VAD, faster-whisper transcription —
  the hands-free *input* half — and streaming. (Roadmap 0.3 / 0.5.)

None of the remaining PC work needs the Jetson. So the highest-value, **$0** next
step is finishing the full hands-free loop on your desktop: say "Azmodan" → it
wakes, listens, transcribes, thinks in character, replies in his cloned voice,
and the gestures simulate. That *is* the proof of concept, complete.

## A cheaper de-risk path (worth serious thought)

The **NVIDIA Jetson Orin Nano Super dev kit is $249** (67 TOPS, **8 GB**). It is
1/5th the price of the Orin NX 16GB kit, and it can answer most of the scary
questions cheaply:

- Does the whole pipeline (Ollama, faster-whisper, XTTS, the DSP) build and run
  on Jetson/aarch64 at all? Yes/no for ~$249.
- What does on-device LLM latency actually feel like for a **7B / 3–4B** model?
- Does the Azmodan personality still hold at a smaller model size?

The catch: **8 GB can't comfortably run the 9B** (the model alone is ~5.5 GB, and
it must share RAM with ASR + TTS + OS). So the Nano Super is a *benchmarking and
smaller-model* platform, not a drop-in for the 9B. But because optics are
deferred, 8 GB stretches further than it otherwise would.

Two ways this plays out, both good:

- If a 3–4B or 7B model on the Nano Super still *feels* like Azmodan, you may
  never need the $1,186 kit — AZMO 1.0 could ship on a $249 brain.
- If it confirms you truly need the 9B, you've spent $249 to buy certainty
  before committing $1,186, and the Nano Super stays useful as a test bench.

This fits your "earn it / prove it before the big spend" instinct better than any
other option here.

## Which kit variant, and the SSD question

The Orin NX module has **no eMMC and no SD slot** — it boots only from an **NVMe
PCIe M.2 SSD** (SATA M.2 does *not* work). So a drive is mandatory; you just
don't have to use Waveshare's.

Waveshare's "Select Kit" options on SKU 24222:

- **BASE-KIT** — module + JETSON-ORIN-IO-BASE carrier board + cooling fan. No SSD,
  no WiFi, no power supply. Cheapest, but you'd source those yourself.
- **DEV-KIT** — everything you actually need, no camera: module + carrier +
  cooling fan + **256 GB NVMe SSD** + dual-band WiFi/BT card + 2 antennas + USB
  cable + Ethernet cable + power supply.
- **DEV-KIT-A / DEV-KIT-B** — the DEV-KIT **plus a camera** (8 MP etc.).

**What you actually need:** the module (16 GB), the carrier board, active
cooling, a power supply, and an NVMe SSD. That's it.

- **Flashing is over USB**, not the network: put the Jetson in recovery mode and
  flash from a host PC. Caveat — the host must be **Ubuntu x86** (NVIDIA SDK
  Manager doesn't run on Windows), so you need an Ubuntu machine/VM for the
  one-time flash, or use Waveshare's pre-flashed image / CLI scripts.
- **WiFi is optional.** The carrier board has **Gigabit Ethernet** — use a wired
  cable during dev to pull models (faster than WiFi anyway). AZMO 1.0 runs fully
  offline, so no network is needed in operation. Only add WiFi if you later want
  wireless remote access to the walking robot.

**The likely savings:** your ~$1,500 is probably a **camera variant (A or B)**.
You've deferred optics — switch the "Select Kit" dropdown to the plain
**DEV-KIT**, or the **BASE-KIT**, and the price drops. That's the real money, not
the SSD.

**On the SSD:** must be **NVMe PCIe, M.2 2280, M-key** (SATA does not work). Size:
128 GB technically works (you only use ~30–40 GB), but it saves ~$10 over 256 GB
and leaves no room for the extra models + conversation datasets your own plan
calls for (LoRA collection, 9B-vs-smaller benchmarking). Get **256 GB minimum,
512 GB for headroom.** The DEV-KIT's bundled 256 GB is fine; if you BYO with the
BASE-KIT, get 512 GB (~$45) — plus a 19 V power supply (~$15). Compare
`BASE-KIT + ~$60` against the plain DEV-KIT price and buy whichever is cheaper.

## Better-value option: Yahboom Orin NX 16GB Super kit (~$790)

The **Yahboom Jetson Orin NX 16GB Super Developer Kit (JetPack 6.2)** is the same
Orin NX 16GB brain for **~$790** (seen $789–839), and it's a better buy than the
Waveshare for AZMO:

- **Complete:** module + carrier board (assembled) + **256 GB M.2 SSD** + WiFi
  card (installed) + power supply + antenna. Nothing to source separately.
- **Pre-flashed:** the SSD ships with the OS/JetPack 6.2 already written — so you
  **skip the Ubuntu-host SDK Manager flash** entirely. That erases the one real
  logistical hurdle (you're on Windows). Boot it and go.
- **"Super" mode:** JetPack 6.2's higher power mode raises Orin NX throughput
  (marketed up to ~157 TOPS). Note: your LLM speed is memory-bandwidth-bound, so
  the ~5–10 tok/s reality for a 9B holds roughly the same; Super helps compute
  more than token rate. Still, free performance.

Same 16 GB Orin NX silicon, **~$400–700 cheaper** than the Waveshare configs, and
it removes the flashing headache. Unless you specifically want Waveshare's
carrier I/O, this is the smarter purchase. (Yahboom is an established Jetson/ROS
robotics vendor, like Waveshare.)

## Carrier board I/O vs what AZMO plugs in

The Yahboom Orin NX carrier board has: **4× USB 3.2 Type-A**, 1× USB-C (data),
**40-pin GPIO header**, Gigabit Ethernet, DisplayPort, 2× MIPI-CSI camera, M.2
Key-M (NVMe, included), M.2 Key-E (WiFi, included), CAN, and 19V/5A DC-in.

Matched to AZMO:

- **Mic + speaker:** the one gap. Jetson carriers have **no analog audio** (no
  3.5 mm jack) — this is normal. Add a **$10 USB audio adapter** (mic-in +
  speaker-out) into a USB-A port, or an **I2S DAC/amp** (e.g., MAX98357) off the
  40-pin header — the latter matches the "audio amplifier + speaker" in your
  torso plan and gives cleaner digital audio. Either is cheap and standard.
- **Teensy link:** plug the Teensy 4.1's USB into any USB-A port → it enumerates
  as `/dev/ttyACM0`, exactly as your brief specifies (USB CDC serial). Or wire it
  to the 40-pin **UART** (pins 8/10, 3.3 V — Teensy is 3.3 V-native) to free a USB
  port. Board supports both.
- **USB headroom:** 4× USB-A easily covers the USB audio adapter + Teensy + a
  keyboard for setup, with ports to spare.
- **Future sensors / e-stop / status LEDs:** the 40-pin header exposes
  I2C/SPI/GPIO/PWM for later.
- **Optics later:** 2× MIPI-CSI camera connectors are already there for when you
  add vision — no board change needed.

**Verdict: the board fits AZMO.** The only extra you must buy is a small USB
audio adapter or an I2S amp for the mic and speaker.

## The decision, two coherent paths

**Path A — Disciplined (matches "earn it").** Finish the hands-free loop on the
PC first. Then buy the Jetson to port and benchmark. Lowest financial risk, and
the kit price is stable — waiting costs you nothing but time.

**Path B — De-risk early.** Buy now, specifically to benchmark the 9B and XTTS
on-device. Those two numbers could force a redesign (smaller model, or Piper
distillation), and you cannot learn them any other way. Defensible if you would
rather face the hard truth now — and remember it is ~$700–1,100, not $1,500.

## Recommendation

**Finish the PC hands-free loop first; and if you want hardware in hand now, make
it the $249 Orin Nano Super — not the $1,186 Orin NX yet.**

1. Over the next couple of weeks, complete the hands-free loop on your desktop
   (wake word → ASR → LLM → cloned voice → gesture sim). $0, and it proves the
   whole experience end to end — the real "earn it" milestone.
2. If the itch to test on real silicon is strong, buy the **$249 Nano Super** to
   benchmark on-device latency and smaller models. Cheap certainty, and it fits
   your budget instinct.
3. When you commit to the Orin NX 16GB, buy the **Yahboom Super kit (~$790,
   complete + pre-flashed)** rather than the pricier Waveshare configs, unless you
   need Waveshare's specific carrier I/O.

**The Yahboom price actually changes the calculus.** At **~$790** for the
complete, pre-flashed real thing, the $249 Nano Super detour is much less
compelling — the final 16 GB brain is only ~$540 more and skips the "will a
smaller model do?" gamble and the flashing hassle. So if budget allows ~$790,
buying the Orin NX 16GB **now** (Yahboom) and porting straight to the final
target is a clean, defensible move — just go in eyes-open about the LLM latency.
Buy the Nano Super only if $790 is out of reach right now and you want to prove
the pipeline for $249 first.

## Questions to answer on-device (the day it arrives)

1. Tokens/sec and personality quality for 9B vs 7B vs 3–4B on the Orin NX.
2. XTTS streaming real-time factor on the Orin NX — conversational, or distill to
   Piper?
3. Full-loop latency: wake → ASR → LLM → TTS → first audio out.

## Sources

- [Waveshare Orin NX 16GB dev kit — SKU 24222, $1,185.99](https://www.waveshare.com/jetson-orin-nx-16g-dev-kit.htm?sku=24222) ·
  [Waveshare wiki](https://www.waveshare.com/wiki/JETSON-ORIN-NX-16G-DEV-KIT)
- [Jetson Orin Nano Super dev kit — $249 (NVIDIA)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/) ·
  [$249 announcement (JetsonHacks)](https://jetsonhacks.com/2024/12/17/jetson-orin-nano-super-developer-kit/)
- [NVIDIA Jetson Orin NX series data sheet](https://developer.nvidia.com/downloads/jetson-orin-nx-series-data-sheet)
- [Running LLMs on Jetson Orin — llama.cpp, Ollama (ProventusNova)](https://proventusnova.com/blog/llm-inference-jetson-orin-llamacpp-ollama/)
- [XTTS model docs (coqui-tts) — streaming / RTF](https://coqui-tts.readthedocs.io/en/latest/models/xtts.html)
- [Piper TTS low-latency on Jetson Orin](https://thomasthelliez.com/blog/running-piper-tts-on-nvidia-jetson-orin-nano-with-low-latency/)
