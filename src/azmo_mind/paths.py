"""Where AZMO's files actually live, independent of where he was launched from.

Every relative path in the config — ``docs/PERSONALITY.md``,
``data/azmo_memory.sqlite3``, ``data/voices/azmo_refs``, ``data/presence`` — used
to be resolved against the **current working directory**. That is correct
exactly once: when a ``.bat`` file has already done ``cd /d "%~dp0"``.

Everywhere else it fails, and it fails *silently*. ``_load_optional`` returns
``""`` for a lore file it cannot find, so running ``azmo`` from another folder
produced a well-formed prompt with the personality, dialogue and gesture
references simply missing. No error, no warning — AZMO just stops being himself
and starts sounding like a generic assistant, which is the single hardest class
of bug to trace because nothing anywhere reports a failure.

It is also a trap waiting on the Jetson: a systemd unit's working directory is
``/`` unless you set one, so the first deployment would have been "he got boring
after we shipped him".

Resolution order:

1. ``$AZMO_HOME`` if set — an explicit override for a deployed layout where the
   code and the data are not siblings.
2. The installation root, found by walking up from this file for a marker. This
   covers the source checkout and the editable install.
3. The current working directory, as a last resort, matching the old behavior.

Absolute paths are always returned untouched, so anything a user configures
explicitly still wins.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# A marker identifies the repo root. `config/azmo.yaml` is listed as well as
# `pyproject.toml` so a deployed tree without packaging metadata still resolves.
_MARKERS = ("pyproject.toml", "config/azmo.yaml")


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """The directory AZMO's relative paths are anchored to."""
    override = os.environ.get("AZMO_HOME")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_dir():
            return candidate.resolve()

    # src/azmo_mind/paths.py -> src/azmo_mind -> src -> <repo root>
    for parent in Path(__file__).resolve().parents:
        if any((parent / marker).exists() for marker in _MARKERS):
            return parent

    return Path.cwd().resolve()


def resolve(path: str | Path) -> Path:
    """Resolve ``path`` against the repo root unless it is already absolute."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def read_text(path: str | Path, default: str = "") -> str:
    """Read a repo-relative text file, returning ``default`` if it is missing.

    Callers use this for the lore documents, where a missing file is survivable
    but must not depend on the working directory.
    """
    resolved = resolve(path)
    if not resolved.is_file():
        return default
    return resolved.read_text(encoding="utf-8")
