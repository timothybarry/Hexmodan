"""Paths anchor to the install root, not the process working directory.

Regression: every relative path resolved against the CWD, and ``_load_optional``
returns ``""`` for a file it cannot find. Running ``azmo`` from anywhere but the
repo root therefore produced a well-formed prompt with the personality, dialogue
and gesture references silently missing — no error, nothing in the logs, AZMO
just quietly stopped being himself.

The Jetson makes this a certainty rather than a risk: a systemd unit's working
directory is ``/`` unless you set one.
"""

import os

from azmo_mind import paths
from azmo_mind.config import load_config
from azmo_mind.prompts import static_prefix


def test_repo_root_contains_the_project_markers():
    root = paths.repo_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "docs" / "PERSONALITY.md").exists()


def test_relative_paths_resolve_against_the_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert paths.resolve("docs/PERSONALITY.md").exists()


def test_absolute_paths_are_left_exactly_as_written(tmp_path):
    target = tmp_path / "somewhere.wav"
    target.write_bytes(b"RIFF")
    assert paths.resolve(target) == target


def test_read_text_returns_the_default_for_a_missing_file():
    assert paths.read_text("docs/DOES_NOT_EXIST.md", default="fallback") == "fallback"


def test_azmo_home_overrides_the_detected_root(tmp_path, monkeypatch):
    paths.repo_root.cache_clear()
    monkeypatch.setenv("AZMO_HOME", str(tmp_path))
    try:
        assert paths.repo_root() == tmp_path.resolve()
    finally:
        monkeypatch.delenv("AZMO_HOME", raising=False)
        paths.repo_root.cache_clear()
    assert "AZMO_HOME" not in os.environ


# ---------------------------------------------------------------------------
# The regression itself
# ---------------------------------------------------------------------------

def test_lore_survives_a_foreign_working_directory(tmp_path, monkeypatch):
    """The failure this whole module exists to prevent."""
    monkeypatch.chdir(tmp_path)
    cfg = load_config("config/azmo.yaml")
    prefix = static_prefix(cfg)
    # Text that exists only inside the lore documents on disk.
    assert "Psychological engine" in prefix, "PERSONALITY.md did not load"
    assert "Imperial declaration" in prefix, "DIALOGUE_STYLE.md did not load"
    assert "Safe Gesture Vocabulary" in prefix, "GESTURES.md did not load"


def test_lore_is_identical_from_any_directory(tmp_path, monkeypatch):
    cfg = load_config("config/azmo.yaml")
    from_root = static_prefix(cfg)
    monkeypatch.chdir(tmp_path)
    assert static_prefix(load_config("config/azmo.yaml")) == from_root


def test_config_paths_are_absolute_after_loading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config("config/azmo.yaml")
    assert cfg.memory.database_path.is_absolute()
    assert cfg.runtime.log_path.is_absolute()
    assert cfg.presence.clips_path.is_absolute()
    if cfg.speech.clone_reference_path is not None:
        assert cfg.speech.clone_reference_path.is_absolute()


def test_data_lands_next_to_the_project_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config("config/azmo.yaml")
    assert cfg.memory.database_path.parent == paths.repo_root() / "data"
    assert tmp_path not in cfg.memory.database_path.parents


def test_emotional_state_does_not_fork_per_directory(tmp_path, monkeypatch):
    """One AZMO, one mood — not one per shortcut."""
    from azmo_mind.state import EmotionStateStore

    from_root = EmotionStateStore().path
    monkeypatch.chdir(tmp_path)
    assert EmotionStateStore().path == from_root
