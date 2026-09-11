"""LeRobot v3 feature schema for raw one-arm Piper replay demonstrations."""
from __future__ import annotations

import numpy as np


def piper_lerobot_features(*, height: int, width: int) -> dict:
    """Return the fixed raw-dataset feature contract for 20 Hz collection.

    Images are stored HWC because OpenCV camera capture provides `uint8` HWC
    frames. State/action retain Piper's raw active-arm 10-D representation;
    X-VLA's inactive-arm padding belongs to the downstream converter.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    image = {"dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channel"]}
    vector = {"dtype": np.dtype("float32").name, "shape": (10,), "names": None}
    return {
        "observation.images.global_rgb": image.copy(),
        "observation.images.wrist_rgb": image.copy(),
        "observation.state": vector.copy(),
        "action": vector.copy(),
    }
