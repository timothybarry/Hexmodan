"""Tests for the GPU power governor.

These must all pass on a machine with no NVIDIA GPU at all - the governor is a
mitigation, never a requirement.
"""

from __future__ import annotations

from azmo_mind import gpu
from azmo_mind.config import GpuConfig


def test_parse_power_csv_reads_all_four_limits():
    state = gpu.parse_power_csv("250.00, 350.00, 100.00, 400.00")
    assert state.current == 250.0
    assert state.default == 350.0
    assert state.minimum == 100.0
    assert state.maximum == 400.0
    assert state.available


def test_parse_power_csv_tolerates_the_watt_suffix():
    state = gpu.parse_power_csv("250.00 W, 350.00 W, 100.00 W, 400.00 W")
    assert state.current == 250.0
    assert state.default == 350.0


def test_parse_power_csv_handles_unsupported_fields():
    state = gpu.parse_power_csv("250.00, [N/A], [Not Supported], 400.00")
    assert state.current == 250.0
    assert state.default is None
    assert state.minimum is None
    assert state.maximum == 400.0


def test_parse_power_csv_handles_empty_output():
    state = gpu.parse_power_csv("")
    assert not state.available
    assert state.error


def test_parse_power_csv_ignores_blank_leading_lines():
    state = gpu.parse_power_csv("\n\n250.00, 350.00, 100.00, 400.00\n")
    assert state.current == 250.0


def test_stagger_accepts_zero_and_does_not_sleep():
    gpu.stagger(0)          # must not raise
    gpu.stagger(-5)


def test_release_vram_is_safe_without_torch_or_a_gpu():
    assert isinstance(gpu.release_vram(), bool)


def test_is_elevated_returns_a_bool():
    assert isinstance(gpu.is_elevated(), bool)


def test_governor_is_a_no_op_when_the_cap_is_disabled():
    messages: list[str] = []
    governor = gpu.PowerGovernor(
        GpuConfig(power_limit_watts=None),
        notify=lambda m, style=None: messages.append(m),
    )
    governor.start()
    governor.stop()
    assert not governor.applied


def test_governor_skips_when_apply_on_launch_is_false():
    governor = gpu.PowerGovernor(GpuConfig(apply_on_launch=False))
    governor.start()
    governor.stop()
    assert not governor.applied


def test_governor_never_raises_without_a_gpu():
    # read_power() returns an error state when nvidia-smi is absent; start/stop
    # must swallow that rather than taking the conversation down with it.
    with gpu.PowerGovernor(GpuConfig()) as governor:
        assert isinstance(governor.applied, bool)


def test_gpu_config_defaults_are_conservative():
    config = GpuConfig()
    assert config.power_limit_watts == 250
    assert config.restore_on_exit is True
    assert config.stagger_ms > 0
