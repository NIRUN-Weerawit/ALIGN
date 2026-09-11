import numpy as np

from piper_xvla.lerobot_factory import piper_lerobot_features


def test_piper_lerobot_features_match_two_hwc_cameras_and_raw_10d_contract():
    features = piper_lerobot_features(height=480, width=640)
    assert features["observation.images.global_rgb"] == {
        "dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channel"]
    }
    assert features["observation.images.wrist_rgb"] == {
        "dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channel"]
    }
    assert features["observation.state"]["shape"] == (10,)
    assert features["action"]["shape"] == (10,)
    assert features["observation.state"]["dtype"] == np.dtype("float32").name
