import json
import time

import pytest

from piper_xvla.webui import PiperWebUI, build_app


def test_align_routes_and_default_status_exist_without_hardware(tmp_path):
    ui = PiperWebUI(dry_run=True, data_dir=tmp_path)
    routes = {route.path for route in build_app(ui, tmp_path).routes}
    assert {"/api/align/start", "/api/align/status", "/api/align/stop",
            "/api/align/control/heartbeat", "/api/align/control/stop"} <= routes
    assert ui.align_status()["status"] == "idle"
    with pytest.raises(RuntimeError, match="live Piper"):
        ui.start_align_inference({})


def test_align_lease_requires_guard_approval_and_does_not_reuse_released_session(tmp_path, monkeypatch):
    ui = PiperWebUI(dry_run=True, data_dir=tmp_path)
    ui.dry_run = False
    ui._manual_enabled = True
    ui._align_state = {"status": "running", "output": None}
    ui._xvla_lease_path = tmp_path / "align.lease.json"
    ui._write_xvla_lease(0)
    monkeypatch.setattr(ui, "_require_manual_arm", lambda **_: None)
    monkeypatch.setattr(ui, "align_status", lambda: {"status": "running", "latest": {"guard_allowed": True, "prediction_age_s": 0.01}})
    active = ui.renew_align_control(5, "session-a", True)
    assert active["active"]
    lease = json.loads(ui._xvla_lease_path.read_text())
    assert lease["session_id"] == "session-a" and lease["expires_monotonic_s"] > time.monotonic()
    ui.revoke_xvla_control("session-a")
    with pytest.raises(RuntimeError, match="released"):
        ui.renew_align_control(5, "session-a", False)
    monkeypatch.setattr(ui, "align_status", lambda: {"status": "running", "latest": {"guard_allowed": False, "prediction_age_s": 0.01}})
    with pytest.raises(RuntimeError, match="rejected"):
        ui.renew_align_control(5, "session-b", True)


def test_align_status_keeps_camera_sample_age_when_serving_latest_row(tmp_path):
    ui = PiperWebUI(dry_run=True, data_dir=tmp_path)
    output = tmp_path / "sample.jsonl"
    sampled_at = time.monotonic() - 0.30
    output.write_text(json.dumps({"frame": 1, "timestamp_monotonic_s": sampled_at + 0.1,
                                  "prediction_age_s": 0.1, "inference_completed_monotonic_s": sampled_at + 0.08,
                                  "guard_allowed": True}) + "\n")
    ui._align_state = {"status": "running", "output": str(output)}
    latest = ui.align_status()["latest"]
    assert latest["prediction_age_s"] >= 0.30


def test_align_heartbeat_uses_configured_camera_age_limit(tmp_path, monkeypatch):
    ui = PiperWebUI(dry_run=True, data_dir=tmp_path)
    ui.dry_run = False
    ui._manual_enabled = True
    ui._align_state = {"status": "running", "output": None}
    ui._xvla_lease_path = tmp_path / "align.lease.json"
    ui._write_xvla_lease(0)
    monkeypatch.setattr(ui, "_require_manual_arm", lambda **_: None)
    monkeypatch.setattr(ui, "xvla_guard_config", lambda: {"max_camera_age_s": 0.5})
    latest = {"guard_allowed": True, "prediction_age_s": 0.3}
    monkeypatch.setattr(ui, "align_status", lambda: {"status": "running", "latest": latest})

    assert ui.renew_align_control(5, "session-a", True)["active"]
    latest["prediction_age_s"] = 0.6
    assert ui.renew_align_control(5, "session-a", False)["active"]  # camera pause renews the lease
    latest["prediction_age_s"] = 0.8
    with pytest.raises(RuntimeError, match="max=0.5 s"):
        ui.renew_align_control(5, "session-a", False)


def test_align_heartbeat_pauses_only_for_camera_staleness(tmp_path, monkeypatch):
    ui = PiperWebUI(dry_run=True, data_dir=tmp_path)
    ui.dry_run = False
    ui._manual_enabled = True
    ui._align_state = {"status": "running", "output": None}
    ui._xvla_lease_path = tmp_path / "align.lease.json"
    ui._write_xvla_lease(0)
    monkeypatch.setattr(ui, "_require_manual_arm", lambda **_: None)
    latest = {"guard_allowed": True, "prediction_age_s": 0.01, "alerts": []}
    monkeypatch.setattr(ui, "align_status", lambda: {"status": "running", "latest": latest})

    assert ui.renew_align_control(5, "session-a", True)["active"]
    latest.update(guard_allowed=False, prediction_age_s=0.3,
                  alerts=[{"code": "STALE_CAMERA", "severity": "REJECT"}])
    assert ui.renew_align_control(5, "session-a", False)["active"]
    with pytest.raises(RuntimeError, match="rejected"):
        ui.renew_align_control(5, "session-b", True)
    latest["alerts"].append({"code": "WORKSPACE_VIOLATION", "severity": "REJECT"})
    with pytest.raises(RuntimeError, match="rejected"):
        ui.renew_align_control(5, "session-a", False)
    latest.update(control_blocked=True, control_block_session="session-a",
                  control_block_reason="WORKSPACE_VIOLATION: X outside bounds")
    with pytest.raises(RuntimeError, match="WORKSPACE_VIOLATION: X outside bounds"):
        ui.renew_align_control(5, "session-a", False)
    latest.update(guard_allowed=True, prediction_age_s=0.01, alerts=[])
    assert ui.renew_align_control(5, "session-b", True)["active"]
    assert ui.renew_align_control(5, "session-b", False)["active"]


def test_align_stop_before_first_heartbeat_prevents_late_control(tmp_path, monkeypatch):
    ui = PiperWebUI(dry_run=True, data_dir=tmp_path)
    ui.dry_run = False
    ui._manual_enabled = True
    ui._align_state = {"status": "running", "output": None}
    ui._xvla_lease_path = tmp_path / "align.lease.json"
    ui._write_xvla_lease(0)
    monkeypatch.setattr(ui, "_require_manual_arm", lambda **_: None)
    monkeypatch.setattr(ui, "align_status", lambda: {"status": "running", "latest": {"guard_allowed": True, "prediction_age_s": 0.01}})

    ui.revoke_xvla_control("session-stopped")
    with pytest.raises(RuntimeError, match="released"):
        ui.renew_align_control(5, "session-stopped", True)
    assert not ui.xvla_control_active()
