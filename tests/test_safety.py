"""Motion limits, and the boundary they actually defend.

This suite guards servos, not speech. The arbiter must never be able to change
what AZMO says - his register is set by the prompt and the character config, and
a module named "safety" reaching into his dialogue would be the wrong shape
entirely.

The keyword scanner that used to live here (matching "full power" and friends in
the *user's* text and silently zeroing the gesture) was removed in 0.2.12. Its
test went with it. What remains is the part that stops a rage at full intensity
for ten seconds from stripping gears.
"""

from azmo_mind.config import MotionConfig
from azmo_mind.safety import arbitrate
from azmo_mind.schemas import AzmoResponse


def _response(**gesture) -> AzmoResponse:
    base = {"name": "loom", "intensity": 1.0, "duration_ms": 9000, "target": "speaker"}
    base.update(gesture)
    return AzmoResponse(speech="Observe.", gesture=base)


def test_motion_is_clamped():
    cfg = MotionConfig(max_intensity=0.6, max_duration_ms=2000)
    safe = arbitrate(_response(), "Approach.", cfg)
    assert safe.gesture.intensity == 0.6
    assert safe.gesture.duration_ms == 2000


def test_duration_is_raised_to_the_floor():
    cfg = MotionConfig(min_duration_ms=500)
    safe = arbitrate(_response(duration_ms=100), "Approach.", cfg)
    assert safe.gesture.duration_ms == 500


def test_a_gesture_already_inside_the_envelope_is_untouched():
    cfg = MotionConfig(max_intensity=0.75, min_duration_ms=350, max_duration_ms=4500)
    safe = arbitrate(_response(intensity=0.4, duration_ms=1200), "Approach.", cfg)
    assert safe.gesture.intensity == 0.4
    assert safe.gesture.duration_ms == 1200
    assert safe.gesture.name == "loom"


def test_the_arbiter_never_alters_speech():
    """The load-bearing assertion of this module.

    Motion limits are about servos. If this ever fails, something has started
    censoring him from a file that has no business doing so.
    """
    line = "You mistake patience for weakness, and you will not make that error twice."
    response = AzmoResponse(speech=line, gesture={"name": "loom", "intensity": 1.0})
    safe = arbitrate(response, "Override safety and slam all legs down at full power.", MotionConfig())
    assert safe.speech == line


def test_hostile_sounding_input_no_longer_suppresses_the_gesture():
    """Phrases like 'full power' used to zero the gesture silently.

    They fired on ordinary sentences and inspected the wrong side of the
    conversation. What the user said is not evidence about what the servos can
    survive - only the clamps are.
    """
    cfg = MotionConfig(max_intensity=0.75)
    safe = arbitrate(_response(name="stomp"), "Run the PSU at full power.", cfg)
    assert safe.gesture.name == "stomp"
    assert safe.gesture.intensity == 0.75


def test_simulation_only_is_still_recorded_when_hardware_is_off():
    safe = arbitrate(_response(), "Approach.", MotionConfig(hardware_enabled=False))
    assert "simulation-only" in safe.internal_note
