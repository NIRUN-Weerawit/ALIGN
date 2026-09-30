from pathlib import Path

from fastapi.testclient import TestClient

from piper_xvla.webui import PiperWebUI, build_app


def _client(tmp_path):
    ui = PiperWebUI(can="can0", dry_run=True, data_dir=tmp_path / "data")
    return ui, TestClient(build_app(ui, tmp_path / "data"))


def test_webui_xvla_diagnostic_status_starts_idle_and_is_read_only(tmp_path):
    _, client = _client(tmp_path)

    status = client.get("/api/xvla/status")

    assert status.status_code == 200
    assert status.json()["status"] == "idle"
    assert status.json()["motion_capability"] == "none"


def test_webui_xvla_checkpoint_picker_uses_selected_local_model(tmp_path, monkeypatch):
    import threading
    import pytest
    import piper_xvla.webui as webui

    root = tmp_path / "outputs"
    older = root / "old_run" / "best.pt"
    newer = root / "new_run" / "best.pt"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    (root / "alias.pt").symlink_to(older)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    (root / "outside.pt").symlink_to(outside)
    linked_folder = tmp_path / "linked_checkpoints"
    linked_folder.mkdir()
    (linked_folder / "best.pt").write_bytes(b"linked")
    (root / "linked_run").symlink_to(linked_folder, target_is_directory=True)
    monkeypatch.setattr(webui, "XVLA_OUTPUTS_ROOT", root)
    module_dir = tmp_path / "piper_xvla"
    module_dir.mkdir()
    (root / "piper_xvla_single_task").mkdir()
    monkeypatch.setattr(webui, "MODULE_DIR", module_dir)

    ui = PiperWebUI(can="can0", dry_run=True, data_dir=tmp_path / "data")
    app = build_app(ui, tmp_path / "data")
    listing = next(route.endpoint for route in app.routes if route.path == "/api/xvla/checkpoints")()
    assert {item["path"] for item in listing["checkpoints"]} == {"old_run/best.pt", "new_run/best.pt", "alias.pt", "linked_run/best.pt"}
    ui.dry_run = False
    with pytest.raises(ValueError, match="absolute path"):
        ui.start_xvla_diagnostic(checkpoint="../outside.pt")

    started = threading.Event()
    selected = []
    def fake_worker(*args):
        selected.append(args[-1])
        started.set()
    monkeypatch.setattr(ui, "_xvla_worker", fake_worker)
    response = ui.start_xvla_diagnostic(frames=0, checkpoint="old_run/best.pt")
    assert started.wait(2)
    assert selected == [older]
    assert response["checkpoint"] == "old_run/best.pt"
    assert ui.xvla_status()["checkpoint"] == "old_run/best.pt"
    with pytest.raises(RuntimeError, match="already running"):
        ui.start_xvla_diagnostic(checkpoint="new_run/best.pt")


def test_webui_xvla_accepts_manual_checkpoint_when_scan_is_empty(tmp_path, monkeypatch):
    import threading
    import pytest
    import piper_xvla.webui as webui

    outputs = tmp_path / "outputs"
    (outputs / "piper_xvla_single_task").mkdir(parents=True)
    checkpoint = tmp_path / "external" / "chosen.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(webui, "XVLA_OUTPUTS_ROOT", outputs)
    module_dir = tmp_path / "piper_xvla"
    module_dir.mkdir()
    monkeypatch.setattr(webui, "MODULE_DIR", module_dir)

    ui = PiperWebUI(can="can0", dry_run=True, data_dir=tmp_path / "data")
    ui.dry_run = False
    assert ui.xvla_checkpoints()["checkpoints"] == []
    with pytest.raises(ValueError, match="absolute path"):
        ui.start_xvla_diagnostic(checkpoint="../external/chosen.pt")
    with pytest.raises(ValueError, match="not found"):
        ui.start_xvla_diagnostic(checkpoint=str(checkpoint.parent / "missing.pt"))

    started = threading.Event()
    selected = []
    def fake_worker(*args):
        selected.append(args[-1])
        started.set()
    monkeypatch.setattr(ui, "_xvla_worker", fake_worker)
    result = ui.start_xvla_diagnostic(frames=0, checkpoint=str(checkpoint))
    assert started.wait(2)
    assert selected == [checkpoint]
    assert result["checkpoint"] == str(checkpoint)


def test_webui_xvla_worker_passes_selected_checkpoint_to_inference(tmp_path, monkeypatch):
    import piper_xvla.webui as webui

    checkpoint = tmp_path / "chosen" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"test")
    output = tmp_path / "diagnostic.jsonl"
    commands = []

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            commands.append(command)

        def wait(self):
            output.write_text('{"guard_allowed": true, "alerts": []}\n')

    monkeypatch.setattr(webui.subprocess, "Popen", FakeProcess)
    ui = PiperWebUI(can="can0", dry_run=True, data_dir=tmp_path / "data")
    ui._xvla_worker(1, output, None, checkpoint)

    assert commands[0][commands[0].index("--checkpoint") + 1] == str(checkpoint)
    assert ui.xvla_status()["status"] == "done"


def test_webui_xvla_status_exposes_latest_predicted_and_measured_pose(tmp_path):
    import json

    ui, client = _client(tmp_path)
    output = tmp_path / "camera_only.jsonl"
    action = [0.1, 0.2, 0.3, 1, 0, 0, 1, 0, 0, 0] + [0] * 10
    current = [0.01, 0.02, 0.03, 1, 0, 0, 1, 0, 0, 0]
    output.write_text(json.dumps({"guard_allowed": False, "predicted_action20": action, "current_active10": current, "alerts": []}) + "\n")
    ui._xvla_state.update(status="running", frames=25, output=str(output))

    status = client.get("/api/xvla/status")

    assert status.status_code == 200
    assert status.json()["latest"]["predicted_pose"]["xyz_m"] == [0.1, 0.2, 0.3]
    assert status.json()["latest"]["measured_pose"]["xyz_m"] == [0.01, 0.02, 0.03]
    assert status.json()["latest"]["guard_allowed"] is False


def test_webui_camera_preview_busy_during_xvla_returns_409_not_asgi_trace(tmp_path):
    ui, client = _client(tmp_path)
    ui._xvla_state["status"] = "running"
    ui.camera_preflight = lambda: (_ for _ in ()).throw(RuntimeError("camera preview is busy; retry after the active camera operation"))

    response = client.get("/api/cameras")

    assert response.status_code == 409
    assert "X-VLA diagnostic owns the cameras" in response.json()["detail"]


def test_webui_mode_endpoint_requires_enable_for_can_and_allows_standby(tmp_path):
    ui, client = _client(tmp_path)

    rejected = client.post("/api/mode", json={"mode": "can"})
    assert rejected.status_code == 200
    assert rejected.json()["ok"] is False
    assert "Enable the arm first" in rejected.json()["error"]
    assert ui.piper.ctrl_mode == 0x02  # initial dry-run teaching mode unchanged

    enabled = client.post("/api/enable")
    assert enabled.status_code == 200 and enabled.json()["ok"] is True

    can = client.post("/api/mode", json={"mode": "can"})
    assert can.status_code == 200 and can.json()["ok"] is True
    assert ui.piper.ctrl_mode == 0x01

    standby = client.post("/api/mode", json={"mode": "standby"})
    assert standby.status_code == 200 and standby.json()["ok"] is True
    assert ui.piper.ctrl_mode == 0x00


def test_webui_does_not_allow_host_offline_replay_mode(tmp_path):
    _, client = _client(tmp_path)
    response = client.post("/api/mode", json={"mode": "offline"})
    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert "pendant-owned" in response.json()["error"]


def test_webui_config_updates_speed_and_persists_camera_settings(tmp_path, monkeypatch):
    import json
    import piper_xvla.webui as webui

    camera_config = tmp_path / "piper_cameras.json"
    camera_config.write_text(json.dumps({
        "global_camera": {"device": "/dev/video14", "width": 640, "height": 480, "fps": 30},
        "wrist_camera": {"device": "/dev/video8", "width": 640, "height": 480, "fps": 30},
    }))
    monkeypatch.setattr(webui, "DEFAULT_CAMERA_CONFIG", camera_config)

    ui = webui.PiperWebUI(can="can0", dry_run=True, data_dir=tmp_path / "data")
    client = TestClient(webui.build_app(ui, tmp_path / "data"))
    result = client.put("/api/config", json={
        "replay_speed": 35,
        "cameras": {
            "global_camera": {"device": "/dev/video20", "width": 1280, "height": 720, "fps": 20},
            "wrist_camera": {"device": "/dev/video21", "width": 640, "height": 480, "fps": 25},
        },
    })
    assert result.status_code == 200
    assert result.json()["replay_speed"] == 35
    assert ui.ctrl.speed == 35
    persisted = json.loads(camera_config.read_text())
    assert persisted["global_camera"]["device"] == "/dev/video20"
    assert persisted["global_camera"]["width"] == 1280
    assert persisted["wrist_camera"]["fps"] == 25


def test_webui_resume_api_reports_exact_attempted_can_payload(tmp_path):
    """A UI Resume click must expose the controller action and its encoded 0x150 bytes."""
    _, client = _client(tmp_path)
    response = client.post("/api/replay", json={"action": "resume"})
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "action": "resume",
        "can_id": "0x150",
        "payload_hex": "0000050000000000",
    }


def test_webui_resume_sends_command_without_feedback_state_gate(tmp_path):
    """Teach feedback is not a reliable pause latch; Resume must reach the controller."""
    from types import SimpleNamespace
    from piper_xvla.webui import PiperWebUI
    from piper_xvla.replay_control import DryRunPiper, PiperReplayController

    class IdlePiper(DryRunPiper):
        def GetArmStatus(self):
            return SimpleNamespace(arm_status=SimpleNamespace(ctrl_mode=0x02, arm_status=0x00, teach_status=0x00, motion_status=0x00, err_code=0))

    ui = PiperWebUI.__new__(PiperWebUI)
    ui.dry_run, ui.piper, ui.speed = True, IdlePiper(), 20
    ui.ctrl = PiperReplayController(ui.piper, replay_speed_percent=20)
    result = ui.do_replay("resume")
    assert result["ok"] is True
    assert ui.piper.calls == [("motion1", 0x00, 0x00, 0x05)]


def test_webui_collection_does_not_trigger_replay_automatically():
    """Collection ownership is recording only; the operator triggers replay separately."""
    import inspect
    from piper_xvla.webui import PiperWebUI

    source = inspect.getsource(PiperWebUI._collect_worker)
    assert "self.ctrl.start_replay()" not in source


def test_webui_live_status_marks_unhealthy_sdk_feedback_unavailable(tmp_path):
    """A dead live SDK must never look like real STANDBY/OFF arm feedback."""
    from types import SimpleNamespace
    from piper_xvla.webui import PiperWebUI

    class DeadPiper:
        def isOk(self): return False
        def GetCanFps(self): return 0.0
        def GetArmStatus(self): return SimpleNamespace(Hz=0.0, arm_status=SimpleNamespace(ctrl_mode=0, teach_status=0, motion_status=0, err_code=0))
        def GetArmJointMsgs(self): return SimpleNamespace(Hz=0.0)
        def GetArmLowSpdInfoMsgs(self): return SimpleNamespace(Hz=0.0)

    ui = PiperWebUI.__new__(PiperWebUI)
    ui.can, ui.dry_run, ui.piper, ui.speed, ui.task = "can0", False, DeadPiper(), 20, ""
    status = ui.snapshot_status()
    assert status["feedback_ok"] is False
    assert status["feedback_state"] == "UNAVAILABLE"
    assert "SDK health" in status["feedback_reason"]


def test_webui_target_limit_keeps_live_feedback_and_recovery_controls_available():
    """A rejected target is controller feedback, not a lost CAN connection."""
    from types import SimpleNamespace
    from piper_xvla.webui import PiperWebUI

    class TargetLimitPiper:
        def isOk(self): return True
        def GetArmStatus(self): return SimpleNamespace(Hz=50.0, arm_status=SimpleNamespace(ctrl_mode=0x01, arm_status=0x04, err_code=0))
        def GetArmJointMsgs(self): return SimpleNamespace(Hz=50.0, joint_state=SimpleNamespace(joint_1=0, joint_2=0, joint_3=0, joint_4=0, joint_5=0, joint_6=0))
        def GetArmLowSpdInfoMsgs(self):
            return SimpleNamespace(Hz=50.0, **{f"motor_{i}": SimpleNamespace(
                foc_status=SimpleNamespace(driver_enable_status=1), motor_temp=30+i, foc_temp=40+i,
            ) for i in range(1, 7)})
        def GetArmEndPoseMsgs(self): return SimpleNamespace(Hz=50.0, end_pose=SimpleNamespace(X_axis=0, Y_axis=0, Z_axis=0, RX_axis=0, RY_axis=0, RZ_axis=0))
        def GetArmGripperMsgs(self): return SimpleNamespace(Hz=50.0, gripper_state=SimpleNamespace(grippers_angle=0))

    ui = PiperWebUI.__new__(PiperWebUI)
    ui.can, ui.dry_run, ui.piper, ui.speed, ui.task = "can0", False, TargetLimitPiper(), 10, ""
    ui._manual_lock = __import__("threading").Lock()
    ui._manual_enabled = False

    status = ui.snapshot_status()

    assert status["feedback_ok"] is True
    assert status["feedback_state"] == "LIVE"
    assert status["command_ok"] is True
    assert "TARGET_LIMIT" in status["command_reason"]
    assert status["motor_temperatures_c"] == [31, 32, 33, 34, 35, 36]
    assert status["driver_temperatures_c"] == [41, 42, 43, 44, 45, 46]


def test_webui_manual_feedback_gate_requires_endpose_and_gripper_streams():
    from types import SimpleNamespace
    from piper_xvla.webui import PiperWebUI

    class PartialFeedbackPiper:
        def isOk(self): return True
        def GetArmStatus(self): return SimpleNamespace(Hz=50.0, arm_status=SimpleNamespace(arm_status=0, err_code=0))
        def GetArmJointMsgs(self): return SimpleNamespace(Hz=50.0)
        def GetArmLowSpdInfoMsgs(self): return SimpleNamespace(Hz=50.0)
        def GetArmEndPoseMsgs(self): return SimpleNamespace(Hz=0.0)
        def GetArmGripperMsgs(self): return SimpleNamespace(Hz=50.0)

    ui = PiperWebUI.__new__(PiperWebUI)
    ui.dry_run, ui.piper = False, PartialFeedbackPiper()

    ok, reason, _ = ui._feedback_health()

    assert ok is False
    assert "end_pose" in reason


def test_webui_reset_reports_failure_when_standby_is_not_observed(tmp_path):
    """Reset cannot claim success solely because the CAN calls did not raise."""
    from piper_xvla.webui import PiperWebUI
    from piper_xvla.replay_control import DryRunPiper, PiperReplayController

    class ResetRejectedPiper(DryRunPiper):
        def MotionCtrl_2(self, *args, **kwargs):
            self.calls.append(("motion2_rejected", args, kwargs))

    ui = PiperWebUI.__new__(PiperWebUI)
    ui.dry_run, ui.piper, ui.speed = True, ResetRejectedPiper(start_ctrl_mode=0x02), 20
    ui.ctrl = PiperReplayController(ui.piper, replay_speed_percent=20)
    result = ui.do_reset()
    assert result["ok"] is False
    assert "standby" in result["error"].lower()


def test_webui_manual_control_api_requires_explicit_arm_then_accepts_a_mark(tmp_path):
    ui, client = _client(tmp_path)

    rejected = client.post("/api/manual/mark", json={"name": "home"})
    assert rejected.status_code == 403

    armed = client.post("/api/manual/arm", json={"confirm": "ARM"})
    assert armed.status_code == 200
    assert armed.json()["ok"] is True

    marked = client.post("/api/manual/mark", json={"name": "home"})
    assert marked.status_code == 200
    assert marked.json()["mark"]["name"] == "home"
    assert marked.json()["mark"]["pose"]["xyz_m"] == [0.0, 0.0, 0.0]


def test_webui_manual_control_can_be_latched_on_and_off_without_timer(tmp_path):
    _, client = _client(tmp_path)

    enabled = client.post("/api/manual/control", json={"enabled": True, "confirm": "ARM"})
    assert enabled.status_code == 200 and enabled.json()["enabled"] is True

    disabled = client.post("/api/manual/control", json={"enabled": False})
    assert disabled.status_code == 200 and disabled.json()["enabled"] is False

    blocked = client.post("/api/manual/gripper", json={"position_m": 0.02})
    assert blocked.status_code == 403


def test_webui_gripper_uses_physical_opening_range_and_streams_vendor_sequence(monkeypatch):
    import threading
    import piper_xvla.webui as webui

    class FakePiper:
        def __init__(self): self.calls = []
        def GripperCtrl(self, *args): self.calls.append(args)

    ui = webui.PiperWebUI.__new__(webui.PiperWebUI)
    ui.dry_run, ui.piper = False, FakePiper()
    ui._manual_enabled = True
    ui._manual_lock = threading.Lock()
    ui._manual_motion_lock = threading.Lock()
    ui._manual_stop_event = threading.Event()
    ui._manual_motion_active = False
    ui._verified_live_feedback = lambda: (True, "live")
    ticks = iter([0.0, 0.01, 0.11])
    monkeypatch.setattr(webui.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(webui.time, "sleep", lambda _: None)

    result = ui.send_manual_gripper({"position_m": 0.04, "effort": 1000, "duration_s": 0.1, "stream_hz": 10})

    assert result["sdk_target"] == 40000
    assert ui.piper.calls == [
        (40000, 1000, 0x02, 0x00),
        (40000, 1000, 0x01, 0x00),
        (40000, 1000, 0x01, 0x00),
    ]


def test_webui_gripper_accepts_calibrated_negative_opening(monkeypatch):
    import threading
    import piper_xvla.webui as webui

    class FakePiper:
        def __init__(self): self.calls = []
        def GripperCtrl(self, *args): self.calls.append(args)

    ui = webui.PiperWebUI.__new__(webui.PiperWebUI)
    ui.dry_run, ui.piper = False, FakePiper()
    ui._manual_enabled = True
    ui._manual_lock = threading.Lock()
    ui._manual_motion_lock = threading.Lock()
    ui._manual_stop_event = threading.Event()
    ui._manual_motion_active = False
    ui._verified_live_feedback = lambda: (True, "live")
    ticks = iter([0.0, 0.01, 0.11])
    monkeypatch.setattr(webui.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(webui.time, "sleep", lambda _: None)

    result = ui.send_manual_gripper({"position_m": -0.04, "effort": 1000, "duration_s": 0.1, "stream_hz": 10})

    assert result["sdk_target"] == -40000
    assert ui.piper.calls[0] == (-40000, 1000, 0x02, 0x00)


def test_webui_jog_computes_relative_target_from_measured_pose(tmp_path):
    _, client = _client(tmp_path)
    assert client.post("/api/manual/control", json={"enabled": True, "confirm": "ARM"}).status_code == 200

    jogged = client.post("/api/manual/jog", json={"axis": "z", "direction": 1, "step_m": 0.002, "speed_percent": 10})

    assert jogged.status_code == 200
    assert jogged.json()["target"]["xyz_m"] == [0.0, 0.0, 0.002]


def test_webui_model_endpose_guard_enforces_ui_per_axis_xyz_step(tmp_path):
    ui, client = _client(tmp_path)
    assert client.post("/api/manual/control", json={"enabled": True, "confirm": "ARM"}).status_code == 200
    ui._manual_pose = lambda: {"xyz_m": [0.1, 0.2, 0.3], "euler_xyz_deg": [0.0, 0.0, 0.0], "gripper_m": 0.0}

    rejected = client.post("/api/manual/endpose", json={
        "x_m": 0.103, "y_m": 0.2, "z_m": 0.3,
        "rx_deg": 0, "ry_deg": 0, "rz_deg": 0,
        "max_axis_step_m": 0.002,
    })
    allowed = client.post("/api/manual/endpose", json={
        "x_m": 0.102, "y_m": 0.198, "z_m": 0.302,
        "rx_deg": 0, "ry_deg": 0, "rz_deg": 0,
        "max_axis_step_m": 0.002,
    })

    assert rejected.status_code == 400
    assert "per-axis" in rejected.json()["detail"]
    assert allowed.status_code == 200


def test_webui_rotation_jog_changes_only_the_requested_measured_euler_axis(tmp_path):
    _, client = _client(tmp_path)
    assert client.post("/api/manual/control", json={"enabled": True, "confirm": "ARM"}).status_code == 200

    jogged = client.post("/api/manual/jog", json={"axis": "rz", "direction": -1, "step_deg": 2.0, "speed_percent": 10})

    assert jogged.status_code == 200
    assert jogged.json()["target"] == {"xyz_m": [0.0, 0.0, 0.0], "euler_xyz_deg": [0.0, 0.0, -2.0]}


def test_webui_joint_target_uses_move_j_and_vendor_joint_limits(tmp_path):
    _, client = _client(tmp_path)
    assert client.post("/api/manual/control", json={"enabled": True, "confirm": "ARM"}).status_code == 200

    accepted = client.post("/api/manual/joints", json={
        "joints_deg": [0, 90, -90, 0, 0, 0], "speed_percent": 10, "duration_s": 0.2, "stream_hz": 10,
    })
    rejected = client.post("/api/manual/joints", json={"joints_deg": [151, 90, -90, 0, 0, 0]})

    assert accepted.status_code == 200
    assert accepted.json()["target"]["joints_deg"] == [0.0, 90.0, -90.0, 0.0, 0.0, 0.0]
    assert rejected.status_code == 400
    assert "joint 1" in rejected.json()["detail"]


def test_webui_marks_can_reorder_and_run_a_single_selected_mark(tmp_path):
    _, client = _client(tmp_path)
    assert client.post("/api/manual/control", json={"enabled": True, "confirm": "ARM"}).status_code == 200
    assert client.post("/api/manual/mark", json={"name": "first"}).status_code == 200
    assert client.post("/api/manual/mark", json={"name": "second"}).status_code == 200

    reordered = client.post("/api/manual/marks/reorder", json={"names": ["second", "first"]})
    one_mark = client.post("/api/manual/trajectory/run", json={"names": ["second"], "speed_percent": 10})

    assert reordered.status_code == 200
    assert [m["name"] for m in reordered.json()["marks"]] == ["second", "first"]
    assert one_mark.status_code == 200
    assert one_mark.json()["marks"] == ["second"]


def test_webui_rejects_unvalidated_manual_speed_above_ten_percent(tmp_path):
    _, client = _client(tmp_path)
    assert client.post("/api/manual/control", json={"enabled": True, "confirm": "ARM"}).status_code == 200

    response = client.post("/api/manual/endpose", json={
        "x_m": 0.0, "y_m": 0.0, "z_m": 0.0,
        "rx_deg": 0.0, "ry_deg": 0.0, "rz_deg": 0.0,
        "speed_percent": 11,
    })

    assert response.status_code == 400
    assert "1–10%" in response.json()["detail"]


def test_webui_cannot_reenable_manual_control_while_prior_command_is_cancelling(tmp_path):
    ui, client = _client(tmp_path)
    ui._manual_motion_active = True
    ui._manual_enabled = False
    ui._manual_stop_event.set()

    response = client.post("/api/manual/control", json={"enabled": True, "confirm": "ARM"})

    assert response.status_code == 403
    assert "cancelling" in response.json()["detail"]
    assert ui._manual_stop_event.is_set()


def test_webui_collection_can_start_while_manual_motion_is_running(tmp_path, monkeypatch):
    import threading

    ui, _ = _client(tmp_path)
    ui.task = "put object in cup"
    ui._manual_motion_active = True
    worker_started = threading.Event()
    monkeypatch.setattr(ui, "_collect_worker", lambda *args: worker_started.set())

    result = ui.start_collect(10)

    assert result["ok"] is True
    assert worker_started.wait(2)
    assert result["state"]["status"] == "opening"


def test_webui_stop_during_opening_finishes_a_single_frame_episode(tmp_path, monkeypatch):
    import threading
    from piper_xvla.collect_session import CollectSession

    ui, _ = _client(tmp_path)
    ui.task = "put object in cup"
    opening = threading.Event()
    finish_opening = threading.Event()

    class FakeSession:
        episode_index = 0
        video_ok = False
        video_path = tmp_path / "review.mp4"
        frames_written = 0

        def snapshot(self):
            return object()

        def write_frame(self, observation):
            self.frames_written += 1

        def finalize(self):
            assert self.frames_written == 1
            return self.frames_written

    def fake_open(*args, **kwargs):
        opening.set()
        assert finish_opening.wait(2)
        return FakeSession()

    monkeypatch.setattr(CollectSession, "open", fake_open)
    ui.start_collect(10)
    assert opening.wait(2)
    stopped = ui.stop_collect()
    assert stopped["state"]["status"] == "stopping"
    assert ui.collection_active()
    finish_opening.set()
    ui._collect_thread.join(3)

    status = ui.collect_status()
    assert status["status"] == "done"
    assert status["frames"] == 1
    assert "stopped by user" in status["message"]


def test_webui_stop_ends_active_recording_before_its_cap(tmp_path, monkeypatch):
    import threading
    from piper_xvla.collect_session import CollectSession

    ui, _ = _client(tmp_path)
    ui.task = "put object in cup"
    first_frame = threading.Event()

    class FakeSession:
        episode_index = 0
        video_ok = False
        video_path = tmp_path / "review.mp4"
        frames_written = 0

        def snapshot(self):
            return object()

        def write_frame(self, observation):
            self.frames_written += 1
            first_frame.set()

        def finalize(self):
            return self.frames_written

    monkeypatch.setattr(CollectSession, "open", lambda *args, **kwargs: FakeSession())
    ui.start_collect(10)
    assert first_frame.wait(2)
    assert ui.stop_collect()["state"]["status"] == "stopping"
    ui._collect_thread.join(3)

    status = ui.collect_status()
    assert status["status"] == "done"
    assert status["frames"] >= 1
    assert status["elapsed_s"] < 10
    assert "stopped by user" in status["message"]


def test_webui_manual_commands_remain_available_during_collection(tmp_path):
    import pytest

    ui, _ = _client(tmp_path)
    ui._collect_state["status"] = "recording"
    ui._manual_enabled = True
    pose = {"x_m": 0, "y_m": 0, "z_m": 0, "rx_deg": 0, "ry_deg": 0, "rz_deg": 0}

    assert ui.send_manual_endpose(pose)["ok"]
    assert ui.send_manual_joints({"joints_deg": [0] * 6})["ok"]
    assert ui.jog_manual_axis({"axis": "x", "direction": 1})["ok"]
    assert ui.send_manual_gripper({"position_m": 0.02})["ok"]
    ui.mark_manual_pose("home")
    assert ui.run_marked_trajectory({"names": ["home"]})["ok"]
    assert ui.collect_status()["status"] == "recording"

    ui._manual_enabled = False
    with pytest.raises(PermissionError, match="manual controls are off"):
        ui.send_manual_endpose(pose)


def test_webui_collect_uses_selected_dataset_folder_for_collection_and_browsing(tmp_path, monkeypatch):
    import asyncio
    import json
    import threading
    import piper_xvla.webui as webui
    from starlette.requests import Request

    ui, client = _client(tmp_path)
    ui.task = "put object in cup"
    selected = tmp_path / "my episodes"
    called = threading.Event()
    args_seen = []

    def fake_worker(window_s, dataset_root, task):
        args_seen.append((window_s, dataset_root, task))
        called.set()

    monkeypatch.setattr(ui, "_collect_worker", fake_worker)
    async def receive():
        return {"type": "http.request", "body": json.dumps({"window_s": 10, "dataset_path": str(selected)}).encode(), "more_body": False}
    request = Request({"type": "http", "method": "POST", "path": "/api/collect/start", "headers": []}, receive)
    start = next(route.endpoint for route in client.app.routes if route.path == "/api/collect/start")
    result = asyncio.run(start(request))

    assert result["ok"] is True
    assert called.wait(2)
    assert args_seen == [(10, selected, ui.task)]
    assert result["state"]["dataset_path"] == str(selected)
    dataset = next(route.endpoint for route in client.app.routes if route.path == "/api/dataset")
    assert dataset() == {"exists": False, "root": str(selected)}

    review_dir = selected / "images" / "review"
    review_dir.mkdir(parents=True)
    (review_dir / "episode-000000_review.mp4").write_bytes(b"video")
    monkeypatch.setattr(webui, "probe_codec", lambda _: "h264")
    list_videos = next(route.endpoint for route in client.app.routes if route.path == "/api/videos")
    videos = list_videos()["videos"]
    assert [video["name"] for video in videos] == ["episode-000000_review.mp4"]


def test_webui_collect_rejects_ambiguous_or_non_dataset_paths(tmp_path):
    import pytest

    ui, _ = _client(tmp_path)
    ui.task = "put object in cup"

    for path in ("relative/dataset", ""):
        with pytest.raises(ValueError, match="absolute"):
            ui.start_collect(10, path)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "other-file").write_text("leave untouched")
    with pytest.raises(ValueError, match="not empty"):
        ui.start_collect(10, str(occupied))
    assert ui.collect_status()["status"] == "idle"


def test_webui_frontend_contains_mode_controls():
    html = (Path(__file__).parents[1] / "webui.html").read_text()
    assert "Standby</button>" in html
    assert "CAN control</button>" in html
    assert "setMode('standby')" in html
    assert "setMode('can')" in html
    assert 'id="btn-replay-resume"' in html
    assert "replay('resume')" in html
    assert "replay('stop')" in html
    assert "Turn control on" in html
    assert "data-jog" in html
    assert 'max="10"' in html
    assert "hold to move" in html
    assert "body.ok === false" in html
    assert "const held = jogHeld" in html
    assert "sendLatestXvlaPose" in html
    assert "stageXvlaPose" not in html
    assert "'/api/manual/endpose'" in html
    assert "Operator-gated" in html
    assert 'id="js-1"' in html
    assert 'id="js-6"' in html
    assert 'id="m-gripper-slider"' in html
    assert "syncControlValue" in html
    assert "gripperNormalizedToMeters" in html
    assert "'/api/manual/gripper'" in html
    assert "jointEngageEnabled" in html
    assert "queueEngagedJointTarget" in html
    assert "Engage continuous sliders" in html
    assert "queueEngagedGripperTarget" in html
    assert "max_position_step_m" in html
    assert 'id="xvla-checkpoint"' in html
    assert "'/api/xvla/checkpoints'" in html
