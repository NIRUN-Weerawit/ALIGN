import json
import sys
from types import SimpleNamespace

from piper_xvla import align_live_inference as runner
from piper_xvla.schema import piper_state_to_xvla20


def test_connect_observer_does_not_send_piper_initialization(monkeypatch):
    calls = []

    class Piper:
        def __init__(self, can):
            calls.append(("create", can))

        def ConnectPort(self, *, piper_init=True):
            calls.append(("connect", piper_init))

        def isOk(self):
            return True

    monkeypatch.setitem(sys.modules, "piper_sdk", SimpleNamespace(C_PiperInterface_V2=Piper))
    runner._connect_observer("can_slave")
    assert calls == [("create", "can_slave"), ("connect", False)]


def test_runner_writes_idle_status_before_first_prediction(tmp_path, monkeypatch):
    """Startup used to crash on target20 before inference published a chunk."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"gripper_normalization": {"raw_meters_min": -0.0326,
                                                               "raw_meters_max": 0.0474}}))
    guard = tmp_path / "guard.json"
    guard.write_text(json.dumps({"workspace_min_m": [-0.05, -0.01, 0.069],
                                 "workspace_max_m": [0.37, 0.33, 0.5],
                                 "max_position_step_m": 0.15, "max_rotation_step_deg": 35,
                                 "max_gripper_step_normalized": 1, "max_inactive_arm_l2": 0.05,
                                 "max_camera_age_s": 0.25, "max_feedback_age_s": 0.25}))
    output = tmp_path / "status.jsonl"
    measured = piper_state_to_xvla20([0.18, 0.12, 0.23], [0, 0, 0, 1], 0.02)

    class Adapter:
        def read_state20(self):
            return measured.copy()

        def close(self):
            pass

    class Event:
        def __init__(self):
            self.stopped = False

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

        def wait(self, _seconds):
            self.stopped = True

    class Thread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass  # No prediction is published during this one-tick test.

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner, "load_model", lambda *_: (object(), 10))
    monkeypatch.setattr(runner, "_connect_observer", lambda *_: object())
    monkeypatch.setattr(runner.PiperSnapshotAdapter, "from_camera_config", lambda *_: Adapter())
    monkeypatch.setattr(runner.threading, "Event", Event)
    monkeypatch.setattr(runner.threading, "Thread", Thread)

    runner.main(["--checkpoint", str(tmp_path / "unused.pt"), "--calibration-manifest", str(manifest),
                 "--guard-config", str(guard), "--settings", "{}", "--lease", str(tmp_path / "lease.json"),
                 "--output", str(output)])
    row = json.loads(output.read_text().strip())
    assert row["frame"] == 1
    assert row["raw_action7"] is None and row["target_pose"] is None
    assert row["guard_allowed"] is False and row["control_sent"] is False
