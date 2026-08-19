from azmo_mind.gestures import simulate
from azmo_mind.schemas import GestureCommand


def test_loom_timeline_ends_at_duration():
    command = GestureCommand(
        name="loom",
        intensity=0.5,
        duration_ms=1800,
        target="speaker",
    )
    timeline = simulate(command)
    assert timeline[0].time_ms == 0
    assert timeline[-1].time_ms == 1800
