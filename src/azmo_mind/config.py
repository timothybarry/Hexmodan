from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from azmo_mind.paths import resolve


class ProviderConfig(BaseModel):
    kind: Literal["ollama", "mock"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen3.5:9b"
    timeout_seconds: float = Field(default=300, ge=15, le=1800)
    temperature: float = Field(default=0.48, ge=0, le=2)
    top_p: float = Field(default=0.90, gt=0, le=1)
    repeat_penalty: float = Field(default=1.12, ge=0.5, le=2)
    context_tokens: int = Field(default=8192, ge=2048, le=32768)
    max_output_tokens: int = Field(default=320, ge=64, le=2048)
    keep_alive: str = "30m"
    think: bool = False


class CharacterConfig(BaseModel):
    name: str = "Azmo"
    canonical_name: str = "Azmodan"
    title: str = "Lord of Sin"
    owner_name: str = "Timothy"
    embodiment: Literal["machine_incarnation", "original_inspired"] = "machine_incarnation"
    max_spoken_words: int = Field(default=100, ge=10, le=300)
    theatricality: float = Field(default=0.82, ge=0, le=1)
    strategic_mind: float = Field(default=0.80, ge=0, le=1)
    temptation: float = Field(default=0.62, ge=0, le=1)
    arrogance: float = Field(default=0.78, ge=0, le=1)
    humor: float = Field(default=0.36, ge=0, le=1)
    menace: float = Field(default=0.66, ge=0, le=1)
    warmth: float = Field(default=0.18, ge=0, le=1)
    profanity: Literal["none", "restrained", "free"] = "restrained"


class MemoryConfig(BaseModel):
    database_path: Path = Path("data/azmo_memory.sqlite3")
    recent_turns: int = Field(default=8, ge=2, le=30)
    retrieved_memories: int = Field(default=5, ge=0, le=20)


class VoiceDspConfig(BaseModel):
    """azmo-voice demonic DSP applied after synthesis (brief section 8).

    The voice is dynamic ("blend by moment"): a per-utterance *heaviness* in
    0..1 is computed from ``intensity_bias`` plus the model's ``VoiceDirection``
    (its ``preset`` and ``subharmonic_mix``). Calm presets stay near the light
    end (deep but intelligible "Commander"); declamatory presets like
    ``imperial_decree``/``restrained_rage``/``victory`` push toward the heavy end
    (layered "Legion" voices and a sub "growl" fade in). All DSP params
    interpolate between their *_light and *_heavy ends by that heaviness.

    Pitch and formants are lowered independently by the WORLD vocoder
    (``use_world``, needs the pyworld dep) so the voice sounds physically huge
    rather than sped-down. Without pyworld it degrades to pitch-only shifting;
    without pedalboard it is a transparent pass-through.
    """

    enabled: bool = True
    # WORLD vocoder deepening is OFF by default: the cloned voice is already deep,
    # and re-synthesizing it muddies/"underwaters" the tone. Character comes from
    # EQ + grit + sub layers instead. Enable only for a thin base voice (e.g.
    # SAPI) that genuinely needs pitch/formant lowering.
    use_world: bool = False
    # Constant demonic register. This is the baseline heaviness applied to every
    # line; the clone stays here rather than drifting toward "human" on calm lines.
    intensity_bias: float = Field(default=0.67, ge=0, le=1)
    # How much the model's per-utterance delivery (preset + mix) is allowed to
    # swing heaviness. LOW keeps a consistent register (recommended); 1.0 = full
    # "blend by moment" dynamics.
    heaviness_variation: float = Field(default=0.15, ge=0, le=1)
    # Pitch lowering: none by default — the clone is already deep, and pitch-down
    # muds it. (Lower a touch only if you want less-human at the cost of clarity.)
    pitch_ratio_light: float = Field(default=1.0, ge=0.3, le=1.0)
    pitch_ratio_heavy: float = Field(default=1.0, ge=0.3, le=1.0)
    # Formant lowering: off by default (1.0). Formant-down is the main cause of
    # the underwater tone; only lower it for a thin base voice.
    formant_ratio_light: float = Field(default=1.0, ge=0.5, le=1.0)
    formant_ratio_heavy: float = Field(default=1.0, ge=0.5, le=1.0)
    drive_light_db: float = Field(default=4.0, ge=0, le=24)
    drive_heavy_db: float = Field(default=10.0, ge=0, le=30)
    # Chamber wetness kept LOW for intelligibility — high reverb = washed out.
    reverb_wet_light: float = Field(default=0.03, ge=0, le=0.4)
    reverb_wet_heavy: float = Field(default=0.07, ge=0, le=0.5)
    # Legion (chorused voices) is inherently washy — reserve it for the biggest
    # declarations only.
    legion_threshold: float = Field(default=0.80, ge=0, le=1)
    growl_threshold: float = Field(default=0.45, ge=0, le=1)
    # Guttural throat-rasp: a saturated low-mid band. Louder gain = more growl.
    grit_gain_db: float = Field(default=-13.0, ge=-60, le=6)
    grit_threshold: float = Field(default=0.20, ge=0, le=1)
    grit_drive_db: float = Field(default=17.0, ge=0, le=36)
    # Mud cut: a dip in the low-mids that fights boxiness and the underwater tone.
    mud_cut_hz: float = Field(default=330.0, ge=150, le=800)
    mud_cut_gain_db: float = Field(default=-4.0, ge=-12, le=0)
    mud_cut_q: float = Field(default=1.0, ge=0.3, le=3.0)
    # Clarity: a boost in the speech-intelligibility band (~3 kHz) so words cut
    # through. This is the main "understandability" control.
    clarity_hz: float = Field(default=3000.0, ge=1500, le=6000)
    clarity_gain_db: float = Field(default=5.0, ge=-6, le=12)
    clarity_q: float = Field(default=1.2, ge=0.3, le=3.0)
    # High-end crispness: a presence shelf boost plus a subtle exciter layer.
    presence_hz: float = Field(default=7000.0, ge=2000, le=14000)
    presence_gain_db: float = Field(default=4.0, ge=-12, le=12)
    air_hz: float = Field(default=5000.0, ge=2000, le=12000)
    air_gain_db: float = Field(default=-12.0, ge=-60, le=6)


class SpeechConfig(BaseModel):
    """Local voice output (azmo-speech seed). All engines run offline.

    engine:
      auto   - best available: clone -> piper -> sapi (Windows) -> espeak-ng -> silent
      clone  - XTTS v2 voice clone from clone_reference_path (needs the clone extra + GPU)
      piper  - neural TTS; requires piper_model_path pointing at a .onnx voice
      sapi   - Windows built-in System.Speech (no install required)
      espeak - espeak-ng formant synthesis (robotic; runs anywhere incl. Jetson)
      none   - text only

    The azmo-voice DSP chain (see dsp) is applied after synthesis for the clone
    and piper engines.
    """

    enabled: bool = True
    engine: Literal["auto", "clone", "piper", "sapi", "espeak", "none"] = "auto"
    piper_model_path: Path | None = None
    # A single WAV, or a directory of clean clips (XTTS averages them into a
    # stronger, more stable speaker embedding — recommended).
    clone_reference_path: Path | None = Path("data/voices/azmo_refs")
    clone_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    clone_language: str = "en"
    # "auto" uses the GPU when torch reports one. Force "cpu" if XTTS crashes
    # the process (a cuDNN build mismatch between torch and CTranslate2 shows up
    # as a hard 0xC0000409 abort mid-synthesis): far slower, but it always works.
    clone_device: Literal["auto", "cuda", "cpu"] = "auto"
    # Run XTTS on the GPU but without cuDNN. Fixes the crash where the machine's
    # cuDNN DLLs are version 8 while torch expects version 9: torch asks for
    # `cudnnGetLibConfig`, the loader reports "Error code 127"
    # (ERROR_PROC_NOT_FOUND), and the process aborts a few seconds later when
    # XTTS reaches its convolution-heavy decoder. torch falls back to its own
    # conv kernels - a little slower, still on the GPU, no broken library.
    clone_disable_cudnn: bool = False
    # Load the voice model at startup instead of at the first reply. Keeps the
    # two big GPU loads apart, but means a voice-stack problem takes the session
    # down before you can talk to him. Set false to defer it (and to reach the
    # listening prompt even when the voice is broken).
    warm_on_start: bool = True
    # XTTS generation params. Temperature is the big one: LOW = steadier and
    # consistent (kills the occasional "drunk"/wandering-accent take); higher =
    # more expressive but more variance. repetition_penalty curbs slurring.
    clone_temperature: float = Field(default=0.30, ge=0.1, le=1.2)
    clone_repetition_penalty: float = Field(default=4.0, ge=1.0, le=10.0)
    clone_top_k: int = Field(default=45, ge=1, le=100)
    clone_top_p: float = Field(default=0.80, ge=0.1, le=1.0)
    clone_length_penalty: float = Field(default=1.0, ge=0.1, le=4.0)
    # Render the whole reply in ONE pass rather than sentence-by-sentence, so the
    # cloned voice stays consistent across the response (splitting makes each
    # sentence its own roll, so some land demonic and others drift). AZMO's
    # replies are short (~100 words), so one pass is well within XTTS's limit.
    clone_split_text: bool = False
    # XTTS v2 generates English in a 250-character window. Past that, with
    # splitting off, the loop overruns instead of stopping - garbage audio at
    # best, a native process abort at worst. AZMO's ~100-word replies exceed it
    # routinely, so we chunk on sentence boundaries ourselves and re-seed each
    # chunk, which keeps the voice consistent in a way the model's own internal
    # splitting did not.
    clone_max_chars: int = Field(default=220, ge=40, le=230)
    clone_chunk_gap_ms: int = Field(default=120, ge=0, le=600)
    # --- streamed delivery (0.2.10) ---------------------------------------
    # Render the reply while the model is still writing it, instead of waiting
    # for the whole document. OFF by default, for two reasons, both of which
    # need a human and a working machine to clear:
    #
    #  1. It has never been heard. Whether the chunk seams are audible is a
    #     question for the ear, exactly like the presence pool. Judge it, then
    #     decide.
    #  2. It runs the LLM and XTTS *concurrently*, which is the precise load
    #     pattern gpu.stagger_ms exists to prevent. Leave it off until the new
    #     PSU and cooler are in and the machine has been proven stable.
    stream_playback: bool = False
    # Rendered chunks held before the first word plays. This is the setting
    # that matters. Pure overlap (1) speaks soonest and risks a stall mid-line;
    # a deeper buffer speaks later and guarantees unbroken delivery. The design
    # log is explicit that a gap inside a line is worse than a pause before it,
    # so the default errs deep.
    stream_prebuffer_chunks: int = Field(default=2, ge=1, le=8)
    # How much text the FIRST chunk needs before it is worth rendering. Small =
    # he starts sooner but on a fragment; large = a more natural opening line.
    # Later chunks always pack up to clone_max_chars, because once he is
    # speaking the latency is hidden and only smoothness is left to buy.
    stream_first_chunk_chars: int = Field(default=60, ge=20, le=230)
    # Give up waiting for the prebuffer after this long and start speaking with
    # whatever exists, rather than standing silent because synthesis is slow.
    stream_prebuffer_timeout_ms: int = Field(default=25000, ge=1000, le=120000)
    clone_seed: int = Field(default=0, ge=0)  # 0 = fresh each time; >0 = reproducible
    # Cached speaker latents (computed once from the reference) for a consistent
    # voice and fast startup. Delete this file to recompute after changing clips.
    clone_latent_cache: Path | None = Path("data/voices/azmo_latents.pth")
    sapi_voice_hint: str = "David"
    volume: int = Field(default=100, ge=0, le=100)
    espeak_base_wpm: int = Field(default=150, ge=80, le=300)
    # Global tempo. 1.0 = natural news-anchor pace (the model's pace only nudges
    # it slightly); raise toward 1.2 for brisker, lower for more deliberate.
    speed: float = Field(default=1.05, ge=0.5, le=2.0)
    dsp: VoiceDspConfig = Field(default_factory=VoiceDspConfig)


class PresenceConfig(BaseModel):
    """azmo-presence: the sounds he makes while he is *not* speaking.

    The goal is not speed, it is the absence of dead air. A machine that sits
    silent while it thinks reads as broken; one that audibly turns the question
    over reads as thinking. With no body yet, audio is the only channel he has.

    Clips are **pre-rendered** WAVs in ``clips_path/<kind>/``, so firing one
    costs a file read rather than a model. Build them with
    ``azmo presence build``, or drop in your own - the player does not care
    where they came from.

    ``sustain_gap_ms`` is the setting that matters most. A single sound at the
    start of the turn does nothing for someone still waiting nine seconds later;
    breathing every few seconds for the whole turn is what actually sells
    contemplation, and it is why a long reply is allowed to be long.
    """

    enabled: bool = True
    clips_path: Path = Path("data/presence")
    # Fire while the LLM is working. This is the one that kills dead air.
    on_think: bool = True
    # Fire after a bare wake word, before the command. Played inside its own deaf
    # window, because unlike the thinking track it fires while the mic is live
    # and waiting for you to speak.
    on_wake: bool = True
    # Cooldown after the wake breath, instead of listener.post_speech_cooldown_ms.
    # That 700 ms is sized for a full reply at volume; a short quiet breath has a
    # much shorter tail. Over-waiting here is NOT harmless - the gate is shut
    # while you are being invited to speak, so every extra millisecond is a
    # chance to eat the first word of your command. Raise it only if his breath
    # is actually being transcribed.
    wake_ack_cooldown_ms: int = Field(default=200, ge=0, le=2000)
    # Relative likelihood per clip kind. Zero disables a kind entirely.
    weights: dict[str, float] = Field(
        default_factory=lambda: {"exhale": 1.0, "growl": 1.0}
    )
    # Silence between contemplation sounds while he is still thinking.
    sustain_gap_ms: int = Field(default=2600, ge=300, le=15000)
    # Cap, so a wedged turn becomes silence rather than an endless growl.
    max_sustain_clips: int = Field(default=4, ge=1, le=20)
    # Don't replay a clip until this many others have played. Guards against the
    # same breath becoming a recognisable tic within one session.
    avoid_repeat_window: int = Field(default=3, ge=0, le=50)
    # Longest we wait on exit for the clip in flight to finish. He must never
    # start speaking over his own breath - that reads as a glitch, which is the
    # exact thing presence exists to remove.
    max_drain_ms: int = Field(default=2500, ge=0, le=10000)
    # Source utterances for `azmo presence build`. XTTS speaks text, so a
    # non-verbal is coaxed out of it with breath spellings rather than words.
    # Which spellings actually render convincingly is empirical - render them,
    # listen, keep the good ones.
    # Deliberately more than you will keep. Curation only ever REMOVES clips, so
    # starting thin leaves a pool too small to sound varied. Render these, delete
    # the duds, and aim to end up with at least three per kind.
    exhale_texts: list[str] = Field(
        default_factory=lambda: [
            "Hhhhh...",
            "Hhhaaah...",
            "Hnnnh...",
            "Hhhh. Mm.",
            "Hhhhhaa.",
            "Hnnh. Hhh.",
        ]
    )
    growl_texts: list[str] = Field(
        default_factory=lambda: [
            "Mmmmm...",
            "Hrrrm...",
            "Mmhrrr...",
            "Rrrmm. Hm.",
            "Mmmhm.",
            "Hrrr. Mmm.",
        ]
    )
    # Delivery for rendered clips. A low, close, unhurried register - these are
    # meant to sound involuntary, not declaimed.
    render_preset: str = "close_ominous"
    render_pace: float = Field(default=0.75, ge=0.6, le=1.35)
    # --- variety (0.2.13) --------------------------------------------------
    # speech.clone_seed is fixed so a good spoken take stays good. Presence
    # wants the OPPOSITE trade: the pool exists so no breath becomes a
    # recognisable tic, and one seed across every clip produces twelve files
    # with a single personality. Each clip gets base_seed + index * stride, so
    # the pool is varied but a rebuild still reproduces it exactly.
    # 0 = old behaviour, every clip on the speech seed.
    render_seed_stride: int = Field(default=997, ge=0, le=100000)
    # Overrides speech.clone_temperature for presence renders only. The low
    # speech value (~0.26) exists to kill wandering-accent takes on words;
    # a wordless breath has no accent to wander, and the extra variance is
    # exactly what stops the pool sounding like one sound. null = use speech's.
    render_temperature: float | None = Field(default=0.65, ge=0.1, le=1.2)
    # Envelope shaping applied after the DSP. The tail fade removes the rising
    # chirp XTTS puts on a trailing vowel; the short head fade softens the
    # low-frequency thump the octave/sub layers make of a sharp onset.
    # Asymmetric on purpose: a breath begins fairly abruptly and dies slowly.
    fade_in_ms: int = Field(default=20, ge=0, le=500)
    fade_out_ms: int = Field(default=220, ge=0, le=2000)


class MotionConfig(BaseModel):
    hardware_enabled: bool = False
    max_intensity: float = Field(default=0.75, ge=0, le=1)
    min_duration_ms: int = Field(default=350, ge=100, le=5000)
    max_duration_ms: int = Field(default=4500, ge=500, le=15000)
    simulator_step_ms: int = Field(default=250, ge=50, le=1000)


class RuntimeConfig(BaseModel):
    save_raw_model_output: bool = False
    log_path: Path = Path("data/azmo_runtime.jsonl")
    warmup_on_chat_start: bool = True
    show_generation_metrics: bool = True
    show_gesture_timeline: bool = True


class ListenerConfig(BaseModel):
    """azmo-listener: mic -> VAD -> wake word -> speech-to-text (brief section 7).

    Needs the ``listen`` extra (sounddevice, webrtcvad, faster-whisper).

    The half-duplex settings (``post_speech_cooldown_ms``, ``echo_*``) exist to
    stop the one failure mode that turns this loop into a runaway: AZMO hearing
    his own voice, recognising his own name in it, and answering himself forever.
    """

    enabled: bool = True
    mic_device: int | None = None            # None = default input device
    wake_word: str = "Azmodan"
    always_on: bool = False                   # True = skip wake word, treat all speech as commands
    # "Azmodan" is not a word Whisper knows, so it renders it as whatever
    # ordinary English it sounds like ("As Madam", "Az modern", "As been in").
    # Two defences: bias the decoder with whisper_initial_prompt so it produces
    # the name correctly, and match what it does produce phonetically rather
    # than by exact spelling.
    wake_fuzzy_threshold: float = Field(default=0.72, ge=0.4, le=1.0)
    # Spellings you have actually seen in the transcript log. These are matched
    # exactly, anywhere in the sentence - the escape hatch when fuzzy matching
    # keeps missing one particular mangling.
    extra_wake_variants: list[str] = Field(default_factory=list)
    # faster-whisper: CPU int8 is the reliable default; set device 'cuda' for GPU.
    # Keeping it on the CPU also keeps the GPU free for the LLM and the voice,
    # which is half of the power-spike mitigation.
    whisper_model: str = "small.en"
    whisper_device: Literal["cpu", "cuda", "auto"] = "cpu"
    whisper_compute_type: str = "int8"
    # Leave headroom for the rest of the machine on a 6-core CPU.
    whisper_cpu_threads: int = Field(default=4, ge=1, le=32)
    # Primes the decoder with vocabulary it would otherwise never guess. Used as
    # 'hotwords' where faster-whisper supports it, else as 'initial_prompt'.
    whisper_initial_prompt: str = "Azmodan"
    # 1 is fastest. 3-5 spends a little more CPU for noticeably better accuracy
    # on unusual words - worth it when the wake word keeps being misheard.
    whisper_beam_size: int = Field(default=3, ge=1, le=10)
    language: str = "en"
    # Voice-activity detection + segmentation.
    # 2 was clipping soft word onsets; 1 keeps more of the leading consonant.
    vad_aggressiveness: int = Field(default=1, ge=0, le=3)  # 3 = most aggressive
    # Audio retained from *before* the VAD fired. A soft "Az" often fails to
    # trip the detector, and a decapitated wake word is an unrecognisable one.
    pre_roll_ms: int = Field(default=500, ge=90, le=2000)
    end_silence_ms: int = Field(default=700, ge=200, le=3000)
    max_utterance_ms: int = Field(default=15000, ge=2000, le=60000)
    # Ignore anything shorter than this: coughs, keystrokes, a closing door.
    # Short blips are what Whisper hallucinates wake words out of.
    min_utterance_ms: int = Field(default=350, ge=0, le=3000)
    wake_cooldown_ms: int = Field(default=250, ge=0, le=2000)
    # How long AZMO waits for your command after hearing the wake word alone.
    follow_up_timeout_ms: int = Field(default=8000, ge=1000, le=30000)
    # Half-duplex: silence the mic for this long AFTER playback ends, covering
    # the speaker tail and room reverb before the gate reopens.
    post_speech_cooldown_ms: int = Field(default=700, ge=0, le=5000)
    # Echo suppression: for this long after he speaks, a transcript that mostly
    # repeats his own words is discarded rather than treated as a command.
    echo_guard_window_ms: int = Field(default=8000, ge=0, le=30000)
    echo_similarity_threshold: float = Field(default=0.6, ge=0.1, le=1.0)


class GpuConfig(BaseModel):
    """GPU power behaviour (RTX 3080 Ti on an aging PSU, brief: crash notes).

    The PC's hard reboots were diagnosed as GPU power transients tripping the
    PSU, not a software fault - but the software can still avoid provoking them.
    Two levers:

    - ``power_limit_watts``: a temporary ``nvidia-smi -pl`` cap. Needs an
      elevated process, and Windows resets it on reboot, so gaming performance
      is never permanently affected. Set to null to leave the GPU alone.
    - ``stagger_ms``: an idle gap inserted between the LLM finishing and the
      voice model starting. Back-to-back inference is exactly the load pattern
      that produces the sharpest current ramps; a short pause lets the rail
      settle between them.
    """

    # Temporary cap in watts, or null for "do not touch the GPU".
    # The 3080 Ti's stock limit is 350 W; ~250 W costs little in this workload.
    power_limit_watts: int | None = Field(default=250, ge=100, le=600)
    # Apply the cap automatically at launch when running elevated.
    apply_on_launch: bool = True
    # Put the limit back when AZMO exits (a reboot also resets it).
    restore_on_exit: bool = True
    # Idle gap between the LLM turn and voice synthesis.
    stagger_ms: int = Field(default=300, ge=0, le=5000)
    # Release cached VRAM after each spoken line, so the LLM and XTTS are not
    # both holding peak allocations on a 12 GB card.
    empty_cache_after_speech: bool = True


class AppConfig(BaseModel):
    provider: ProviderConfig
    character: CharacterConfig
    memory: MemoryConfig
    motion: MotionConfig
    runtime: RuntimeConfig
    speech: SpeechConfig = Field(default_factory=SpeechConfig)
    listener: ListenerConfig = Field(default_factory=ListenerConfig)
    gpu: GpuConfig = Field(default_factory=GpuConfig)
    presence: PresenceConfig = Field(default_factory=PresenceConfig)

    @model_validator(mode="after")
    def _anchor_paths(self) -> AppConfig:
        """Anchor every relative path to the install root, not the process CWD.

        These defaults (``data/azmo_memory.sqlite3``, ``data/voices/azmo_refs``,
        ``data/presence``…) read as "next to the project", and that is what they
        mean — but a bare relative ``Path`` means "next to wherever this process
        happened to start". Launch AZMO from another folder and he writes a fresh
        empty memory database somewhere unexpected and finds no voice clips,
        without reporting either. Resolving here means every consumer downstream
        gets an absolute path and none of them have to think about it.

        An absolute path in the YAML is left exactly as written.
        """
        self.memory.database_path = resolve(self.memory.database_path)
        self.runtime.log_path = resolve(self.runtime.log_path)
        self.presence.clips_path = resolve(self.presence.clips_path)
        if self.speech.piper_model_path is not None:
            self.speech.piper_model_path = resolve(self.speech.piper_model_path)
        if self.speech.clone_reference_path is not None:
            self.speech.clone_reference_path = resolve(self.speech.clone_reference_path)
        if self.speech.clone_latent_cache is not None:
            self.speech.clone_latent_cache = resolve(self.speech.clone_latent_cache)
        return self


def load_config(path: str | Path = "config/azmo.yaml") -> AppConfig:
    config_path = resolve(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(data)
