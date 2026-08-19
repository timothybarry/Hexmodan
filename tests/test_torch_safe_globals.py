"""torch 2.6 refuses coqui's XTTS checkpoint unless its classes are allowlisted.

Regression: torch 2.6 changed `torch.load`'s `weights_only` default to True.
The XTTS checkpoint contains pickled config objects, so every synthesis failed
with "Unsupported global: GLOBAL TTS.tts.configs.xtts_config.XttsConfig".
Because `_speak` catches SpeechError and only prints a warning, the audible
symptom was AZMO breathing and then saying nothing at all.
"""

import sys
import types

import pytest

from azmo_mind import speech


@pytest.fixture(autouse=True)
def _reset():
    speech._xtts_globals_allowed = False
    yield
    speech._xtts_globals_allowed = False


def test_no_torch_installed_is_not_an_error():
    """The clone extra is optional; absence must degrade, not raise."""
    assert speech.allow_xtts_globals() is False


def test_old_torch_without_the_api_is_a_no_op(monkeypatch):
    """torch < 2.6 has no add_safe_globals and needs nothing done."""
    fake = types.ModuleType("torch")
    fake.serialization = types.SimpleNamespace()      # no add_safe_globals
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert speech.allow_xtts_globals() is False


def _fake_tts_modules(monkeypatch, names):
    """Install stand-ins for the coqui modules holding the pickled classes."""
    made = {}
    for module_path, name in speech._XTTS_PICKLED_CLASSES:
        module = made.setdefault(module_path, types.ModuleType(module_path))
        if name in names:
            setattr(module, name, type(name, (), {}))
    for module_path, module in made.items():
        monkeypatch.setitem(sys.modules, module_path, module)


def test_all_four_classes_are_allowlisted(monkeypatch):
    registered = []
    fake = types.ModuleType("torch")
    fake.serialization = types.SimpleNamespace(add_safe_globals=registered.extend)
    monkeypatch.setitem(sys.modules, "torch", fake)
    _fake_tts_modules(monkeypatch, {n for _, n in speech._XTTS_PICKLED_CLASSES})

    assert speech.allow_xtts_globals() is True
    assert {cls.__name__ for cls in registered} == {
        name for _, name in speech._XTTS_PICKLED_CLASSES
    }


def test_a_class_that_moved_does_not_block_the_others(monkeypatch):
    """Coqui has relocated these between releases; partial success still helps."""
    registered = []
    fake = types.ModuleType("torch")
    fake.serialization = types.SimpleNamespace(add_safe_globals=registered.extend)
    monkeypatch.setitem(sys.modules, "torch", fake)
    _fake_tts_modules(monkeypatch, {"XttsConfig"})     # only one still exists

    assert speech.allow_xtts_globals() is True
    assert [cls.__name__ for cls in registered] == ["XttsConfig"]


def test_repeated_calls_register_once(monkeypatch):
    calls = []
    fake = types.ModuleType("torch")
    fake.serialization = types.SimpleNamespace(add_safe_globals=lambda g: calls.append(g))
    monkeypatch.setitem(sys.modules, "torch", fake)
    _fake_tts_modules(monkeypatch, {n for _, n in speech._XTTS_PICKLED_CLASSES})

    speech.allow_xtts_globals()
    speech.allow_xtts_globals()
    speech.allow_xtts_globals()
    assert len(calls) == 1, "the model loads on every turn; registration must not repeat"


def test_registration_failure_is_survivable(monkeypatch):
    def boom(_globals):
        raise RuntimeError("already registered")

    fake = types.ModuleType("torch")
    fake.serialization = types.SimpleNamespace(add_safe_globals=boom)
    monkeypatch.setitem(sys.modules, "torch", fake)
    _fake_tts_modules(monkeypatch, {n for _, n in speech._XTTS_PICKLED_CLASSES})

    assert speech.allow_xtts_globals() is False   # reported, not raised


def test_trusted_load_passes_weights_only_false(monkeypatch):
    seen = {}
    fake = types.ModuleType("torch")
    fake.load = lambda path, **kw: seen.update(path=path, **kw) or "latents"
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert speech.torch_load_trusted("cache.pth", map_location="cpu") == "latents"
    assert seen["weights_only"] is False
    assert seen["map_location"] == "cpu"


def test_trusted_load_falls_back_on_older_torch(monkeypatch):
    def old_signature(path, map_location=None, **kw):
        if "weights_only" in kw:
            raise TypeError("unexpected keyword argument 'weights_only'")
        return "latents"

    fake = types.ModuleType("torch")
    fake.load = old_signature
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert speech.torch_load_trusted("cache.pth", map_location="cpu") == "latents"
