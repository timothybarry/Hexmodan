from azmo_mind.schemas import EmotionState
from azmo_mind.state import update_state


def test_positive_message_increases_trust():
    state = EmotionState()
    updated = update_state(state, "Thank you, that was awesome.")
    assert updated.trust > state.trust


def test_insult_increases_irritation():
    state = EmotionState()
    updated = update_state(state, "You are useless.")
    assert updated.irritation > state.irritation
