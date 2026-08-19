from azmo_mind.motion_link import (
    GESTURE_PRIORITY,
    PROTOCOL_VERSION,
    CommandState,
    SimulatedMotionLink,
)
from azmo_mind.schemas import GestureCommand


def test_gesture_completes_with_full_lifecycle():
    link = SimulatedMotionLink()
    result = link.send_gesture(
        GestureCommand(name="loom", intensity=0.4, duration_ms=1500, target="speaker")
    )
    assert result.state == CommandState.COMPLETED
    assert result.lifecycle == [
        CommandState.RECEIVED,
        CommandState.ACCEPTED,
        CommandState.EXECUTING,
        CommandState.COMPLETED,
    ]
    assert result.timeline, "an executed gesture must produce timeline events"


def test_none_gesture_completes_without_executing():
    link = SimulatedMotionLink()
    result = link.send_gesture(GestureCommand(name="none"))
    assert result.state == CommandState.COMPLETED
    assert CommandState.EXECUTING not in result.lifecycle
    assert result.timeline == []


def test_command_envelope_matches_brief_section_10():
    link = SimulatedMotionLink()
    result = link.send_gesture(GestureCommand(name="boast", intensity=0.5, duration_ms=1000))
    command = result.command
    assert command.version == PROTOCOL_VERSION
    assert command.type == "gesture"
    assert command.priority == GESTURE_PRIORITY
    assert command.timeout_ms > command.parameters["duration_ms"]
    wire = command.to_wire()
    for key in ("version", "id", "type", "priority", "timeout_ms", "parameters"):
        assert key in wire


def test_command_ids_are_unique_and_monotonic():
    link = SimulatedMotionLink()
    ids = [link.send_gesture(GestureCommand(name="neutral")).command.id for _ in range(5)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 5


def test_hardware_can_never_be_enabled_on_simulator():
    link = SimulatedMotionLink(hardware_enabled=True)
    health = link.health()
    assert health["hardware_requested"] is True
    assert health["hardware_enabled"] is False
