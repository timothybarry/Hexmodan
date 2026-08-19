# Voice clone references

`azmo_reference.wav` is the reference XTTS clones AZMO's voice from.

The bundled file is a ~25s starter assembled from the driest Azmodan game lines
(quick heuristic clean-up, not source-separated). For a noticeably cleaner clone,
regenerate it with Demucs isolation:

    pip install -e ".[prep]"
    python scripts/prepare_reference.py "Diablo 3- All Azmodan Voice Lines.mp3" \
        --out data/voices/azmo_reference.wav --seconds 25

Then `azmo chat` (engine: clone or auto) will speak in that voice, with the
azmo-voice DSP layered on top. See AZMO_PROJECT_BRIEF.md sections 8 and 15 for
the voice design and the likeness/copyright notes.
