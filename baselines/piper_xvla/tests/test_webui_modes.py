from pathlib import Path

from fastapi.testclient import TestClient

from piper_xvla.webui import PiperWebUI, build_app


def _client(tmp_path):
    ui = PiperWebUI(can="can0", dry_run=True, data_dir=tmp_path / "data")
    return ui, TestClient(build_app(ui, tmp_path / "data"))


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


def test_webui_frontend_contains_mode_controls():
    html = (Path(__file__).parents[1] / "webui.html").read_text()
    assert "Mode: Standby" in html
    assert "Mode: CAN control" in html
    assert "setMode('standby')" in html
    assert "setMode('can')" in html
