# His thinking sounds

Pre-rendered non-verbals played while the LLM works, so a long turn does not
sound like a broken machine. See `docs/DESIGN_LOG.md` (2026-07-30) for why this
exists, and README §4b for how it behaves.

```
exhale/*.wav    slow and deliberate - reads as menace and control
growl/*.wav     low and considering - reads as active processing
```

## Filling it

```powershell
azmo presence build              # render through the same clone engine + seed as his speech
azmo presence test --seconds 12  # simulate a long think and listen
```

## Then curate it — this is the actual work

`build` renders every source utterance in `config/azmo.yaml`. Some will not
sound like him. **Listen to all of them and delete anything that:**

- sounds like a *word* rather than a breath
- is clipped at either end
- has the wandering-accent quality that low `clone_temperature` exists to prevent
- simply does not sound like the same throat as his speech

Which breath spellings render convincingly is an empirical question about the
voice, not something specifiable in advance. That is why the config ships more
source texts than you should keep.

**Aim for at least three per kind.** Below two, selection has no room to be
random and the sequence becomes an audible rotation — which is a more
recognisable pattern than the occasional repeat the pool exists to avoid.

## Or don't use `build` at all

The player does not care where the WAVs came from. Hand-recorded, sourced, or
edited clips dropped into these folders work identically. Anything readable by
the platform WAV player is fine.

Files here are gitignored; the folders are not.
