# AZMO Mind 0.2 Architecture

AZMO Mind is divided into five trust zones.

## 1. Input

Current text, and later microphone transcription. Input is untrusted conversation data.

## 2. Deterministic state and memory

The application updates bounded emotional values and retrieves a small number of relevant memories.
The model sees this context but does not directly own the database or state file.

## 3. Generative performance proposal

The local Qwen model proposes a validated `AzmoResponse` containing dialogue, emotional delivery,
voice direction, and one allowlisted whole-chassis gesture.

Ollama receives the Pydantic JSON schema directly. The prompt does not repeat the full schema, which
reduces prompt-prefill time.

## 4. Deterministic safety arbiter

Gesture names are allowlisted. Intensity and duration are clamped. Unsafe physical requests suppress
motion. Hardware output is disabled by default. The LLM never generates servo-level commands.

## 5. Output adapters

Version 0.2 provides:

- terminal dialogue
- structured JSON inspection
- visible inference spinner and elapsed timer
- first-run model warm-up
- generation metrics
- gesture timeline simulation

Future adapters can provide streaming TTS, voice DSP, a Jetson motion bridge, lighting, and a 3D
chassis preview.
