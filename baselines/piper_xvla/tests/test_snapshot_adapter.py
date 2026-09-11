import json
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as R

from piper_xvla.snapshot_adapter import PiperSnapshotAdapter


class FakeCapture:
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.read_calls = 0

    def read(self):
        self.read_calls += 1
        return True, self.frame


def feedback_piper():
    end_pose = SimpleNamespace(
        X_axis=123_456,
        Y_axis=-234_567,
        Z_axis=345_678,
        RX_axis=90_000,
        RY_axis=-45_000,
        RZ_axis=30_000,
    )
    gripper = SimpleNamespace(grippers_angle=12_345)
    return SimpleNamespace(
        end_pose=end_pose,
        gripper=gripper,
        GetArmEndPoseMsgs=lambda: SimpleNamespace(end_pose=end_pose),
        GetArmGripperMsgs=lambda: SimpleNamespace(gripper_state=gripper),
    )


def test_snapshot_converts_bgr_and_copies_raw_piper_feedback_to_padded_xvla_state():
    global_bgr = np.array([[[3, 2, 1], [30, 20, 10]]], dtype=np.uint8)
    wrist_bgr = np.array([[[6, 5, 4], [60, 50, 40]]], dtype=np.uint8)
    piper = feedback_piper()
    adapter = PiperSnapshotAdapter(
        piper=piper,
        global_capture=FakeCapture(global_bgr),
        wrist_capture=FakeCapture(wrist_bgr),
        task="put cube in basket",
    )

    observation = adapter.snapshot()

    np.testing.assert_array_equal(observation.global_rgb, global_bgr[..., ::-1])
    np.testing.assert_array_equal(observation.wrist_rgb, wrist_bgr[..., ::-1])
    assert observation.global_rgb.dtype == np.uint8
    assert observation.wrist_rgb.dtype == np.uint8
    np.testing.assert_allclose(observation.state20[:3], [0.123456, -0.234567, 0.345678])
    expected_rot6d = R.from_euler("xyz", [90, -45, 30], degrees=True).as_matrix()[:, :2].reshape(-1)
    np.testing.assert_allclose(observation.state20[3:9], expected_rot6d)
    np.testing.assert_allclose(observation.state20[9], 0.012345)
    np.testing.assert_array_equal(observation.state20[10:], np.zeros(10, dtype=np.float32))

    piper.end_pose.X_axis = 999_999
    piper.gripper.grippers_angle = 999_999
    np.testing.assert_allclose(observation.state20[:3], [0.123456, -0.234567, 0.345678])
    np.testing.assert_allclose(observation.state20[9], 0.012345)


class OpenableFakeCapture(FakeCapture):
    def __init__(self, frame: np.ndarray):
        super().__init__(frame)
        self.settings = []

    def isOpened(self):
        return True

    def set(self, property_id, value):
        self.settings.append((property_id, value))

    def release(self):
        pass


def test_from_camera_config_opens_configured_global_and_wrist_devices(monkeypatch, tmp_path):
    config_path = tmp_path / "cameras.json"
    config_path.write_text(json.dumps({
        "global_camera": {"device": "/dev/global", "width": 640, "height": 480, "fps": 30},
        "wrist_camera": {"device": "/dev/wrist", "width": 320, "height": 240, "fps": 15},
    }))
    captures = {
        "/dev/global": OpenableFakeCapture(np.zeros((1, 1, 3), dtype=np.uint8)),
        "/dev/wrist": OpenableFakeCapture(np.zeros((1, 1, 3), dtype=np.uint8)),
    }
    fake_cv2 = SimpleNamespace(
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
        VideoCapture=lambda device: captures[device],
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    adapter = PiperSnapshotAdapter.from_camera_config(feedback_piper(), "put cube in basket", config_path)

    assert adapter._global_capture is captures["/dev/global"]
    assert adapter._wrist_capture is captures["/dev/wrist"]
    assert captures["/dev/global"].settings == [(3, 640), (4, 480), (5, 30)]
    assert captures["/dev/wrist"].settings == [(3, 320), (4, 240), (5, 15)]


def test_snapshot_copies_end_pose_before_reading_next_mutable_sdk_envelope():
    piper = feedback_piper()

    def gripper_feedback_after_end_pose_mutation():
        piper.end_pose.X_axis = 999_999
        piper.end_pose.RX_axis = 0
        return SimpleNamespace(gripper_state=piper.gripper)

    piper.GetArmGripperMsgs = gripper_feedback_after_end_pose_mutation
    adapter = PiperSnapshotAdapter(
        piper,
        FakeCapture(np.zeros((1, 1, 3), dtype=np.uint8)),
        FakeCapture(np.zeros((1, 1, 3), dtype=np.uint8)),
        "put cube in basket",
    )

    state20 = adapter.snapshot().state20

    np.testing.assert_allclose(state20[:3], [0.123456, -0.234567, 0.345678])
    expected_rot6d = R.from_euler("xyz", [90, -45, 30], degrees=True).as_matrix()[:, :2].reshape(-1)
    np.testing.assert_allclose(state20[3:9], expected_rot6d)
