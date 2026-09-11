import numpy as np

from piper_xvla.replay_collector import FutureStateLabeler, ReplayObservation


def observation(value: float) -> ReplayObservation:
    state = np.zeros(20, dtype=np.float32)
    state[0] = value
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    return ReplayObservation(image, image.copy(), state, "put cube in basket")


def test_future_state_labeler_uses_next_measured_state_as_action():
    labeler = FutureStateLabeler()
    assert labeler.push(observation(1.0)) is None

    frame = labeler.push(observation(2.0))
    assert frame is not None
    assert frame.state20[0] == 1.0
    assert frame.action20[0] == 2.0


def test_future_state_labeler_flushes_final_frame_with_its_own_measured_state():
    labeler = FutureStateLabeler()
    labeler.push(observation(3.0))
    final = labeler.flush()
    assert final is not None
    assert final.state20[0] == 3.0
    assert final.action20[0] == 3.0
    assert labeler.flush() is None
