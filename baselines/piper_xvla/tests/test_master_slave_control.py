from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from piper_xvla.master_slave_control import (
    MASTER_JOINT_PHYSICAL_LIMITS_DEG, SLAVE_JOINT_PHYSICAL_LIMITS_DEG,
    _fresh_gripper_mm, hold_slave_position, map_joint_positions, mapped_gripper_target,
    mapped_joint_target, run_joint_mirror, slave_measured_joints,
)


class FakePiper:
    def __init__(self, joints, command_joints=None, command_hz=0):
        self.joints = joints
        self.command_joints = joints if command_joints is None else command_joints
        self.command_hz = command_hz
        self.enabled = True
        self.mode = 0
        self.targets = []
        self.speed_rates = []
        self.gripper_mm = 0.0
        self.gripper_hz = 50
        self.gripper_timestamp = None
        self.gripper_command_mm = 0.0
        self.gripper_command_hz = 50
        self.gripper_command_timestamp = None
        self.gripper_command_status = 0x01
        self.gripper_command_set_zero = 0
        self.gripper_targets = []
        self.emergency_calls = []

    def isOk(self):
        return True

    def GetArmJointMsgs(self):
        return SimpleNamespace(Hz=50, time_stamp=time.time(), joint_state=SimpleNamespace(**{
            f"joint_{i}": int(value * 1000) for i, value in enumerate(self.joints, 1)
        }))

    def GetArmJointCtrl(self):
        return SimpleNamespace(Hz=self.command_hz, time_stamp=time.time(), joint_ctrl=SimpleNamespace(**{
            f"joint_{i}": int(value * 1000) for i, value in enumerate(self.command_joints, 1)
        }))

    def GetArmGripperMsgs(self):
        timestamp = time.time_ns() if self.gripper_timestamp is None else self.gripper_timestamp
        return SimpleNamespace(Hz=self.gripper_hz, time_stamp=timestamp,
                               gripper_state=SimpleNamespace(grippers_angle=int(self.gripper_mm * 1000)))

    def GetArmGripperCtrl(self):
        timestamp = time.time_ns() if self.gripper_command_timestamp is None else self.gripper_command_timestamp
        return SimpleNamespace(Hz=self.gripper_command_hz, time_stamp=timestamp,
                               gripper_ctrl=SimpleNamespace(grippers_angle=int(self.gripper_command_mm * 1000),
                                                            status_code=self.gripper_command_status,
                                                            set_zero=self.gripper_command_set_zero))

    def GetArmLowSpdInfoMsgs(self):
        return SimpleNamespace(**{f"motor_{i}": SimpleNamespace(foc_status=SimpleNamespace(driver_enable_status=self.enabled)) for i in range(1, 7)})

    def GetArmStatus(self):
        return SimpleNamespace(Hz=50, arm_status=SimpleNamespace(ctrl_mode=self.mode, arm_status=0, err_code=0))

    def EnableArm(self, _):
        self.enabled = True

    def MotionCtrl_2(self, ctrl_mode, **options):
        self.mode = ctrl_mode
        self.speed_rates.append(options.get("move_spd_rate_ctrl"))

    def JointCtrl(self, *target):
        self.targets.append(target)

    def GripperCtrl(self, *target):
        self.gripper_targets.append(target)

    def EmergencyStop(self, code):
        self.emergency_calls.append(code)


def test_mapped_joint_target_applies_offsets_in_degrees():
    master = FakePiper([0, 0, 0, 0, 0, 0], [10, 20, -30, 4, 5, 6], command_hz=50)
    assert mapped_joint_target(master, [1, -2, 3, 0, 0.5, -1]).tolist() == pytest.approx(
        [11, 18, -177 + 144 * 175 / 174 + 3, 4, 1.5, -108 + 168 * 242 / 243 - 1])


def test_mapped_joint_target_caps_offset_at_slave_limit():
    master = FakePiper([0] * 6, [150, 20, -30, 0, 0, 0], command_hz=50)
    assert mapped_joint_target(master, [10, 0, 0, 0, 0, 0])[0] == 153


def test_mirror_stream_bounds_j6_lead_and_stops():
    master = FakePiper([0] * 6, [1.25, 30, -40, 2, -3, 10], command_hz=50)
    slave = FakePiper([0, 30, -40, 0, 0, 0])
    stop = threading.Event()
    updates = []

    def on_update(update):
        updates.append(update)
        stop.set()

    run_joint_mirror(master, slave, [0.25, 0, 0, 0, 0, 0], 5, stop, on_update)
    assert slave.targets == [(1500, 30000, -42230, 2000, -7000, 5000)]
    assert updates[0]["sent"] == 1
    assert updates[0]["desired_j6_deg"] == pytest.approx(-108 + 172 * 242 / 243)
    assert updates[0]["j6_remaining_deg"] == pytest.approx(round(-108 + 172 * 242 / 243 - 5, 2))


def test_mirror_allows_thirty_percent_but_rejects_higher_speed():
    master = FakePiper([0] * 6, [0] * 6, command_hz=50)
    slave = FakePiper([0] * 6)
    stop = threading.Event()
    run_joint_mirror(master, slave, [0] * 6, 30, stop, lambda _: stop.set())
    assert slave.speed_rates == [30, 30]
    with pytest.raises(ValueError, match="1–30 percent"):
        run_joint_mirror(master, slave, [0] * 6, 31, threading.Event())


def test_mirror_sends_master_gripper_command_with_bounded_lead():
    master = FakePiper([0] * 6, [0] * 6, command_hz=50)
    slave = FakePiper([0] * 6)
    master.gripper_command_mm = 64.5
    master.gripper_hz = 0  # Physical feedback is absent on the master CAN bus.
    slave.gripper_mm = 2.0
    stop = threading.Event()
    updates = []

    def on_update(sample):
        updates.append(sample)
        stop.set()

    run_joint_mirror(master, slave, [0] * 6, 5, stop, on_update, mirror_gripper=True)
    assert slave.gripper_targets == [(7000, 1000, 0x01, 0x00)]
    assert updates[0]["master_gripper_mm"] == 64.5
    assert updates[0]["mapped_gripper_mm"] == 48
    assert updates[0]["slave_gripper_mm"] == 2
    assert updates[0]["gripper_target_mm"] == 7
    assert updates[0]["gripper_remaining_mm"] == 41


def test_gripper_physical_ranges_map_proportionally_and_cap_master_overshoot():
    assert mapped_gripper_target(-8.1) == pytest.approx((-8.1, -41.5))
    assert mapped_gripper_target(64.5) == pytest.approx((64.5, 48.0))
    assert mapped_gripper_target(28.2) == pytest.approx((28.2, 3.25))
    assert mapped_gripper_target(71.54) == pytest.approx((64.5, 48.0))

    master = FakePiper([0] * 6, [0] * 6, command_hz=50)
    slave = FakePiper([0] * 6)
    master.gripper_command_mm = 71.54
    slave.gripper_mm = 23.3
    stop = threading.Event()
    updates = []

    def on_update(sample):
        updates.append(sample)
        stop.set()

    run_joint_mirror(master, slave, [0] * 6, 5, stop, on_update, mirror_gripper=True)
    assert slave.gripper_targets == [(28300, 1000, 0x01, 0x00)]
    assert updates[0]["gripper_input_clamped"] is True
    assert updates[0]["bounded_master_gripper_mm"] == 64.5
    assert updates[0]["mapped_gripper_mm"] == 48.0


def test_master_gripper_command_requires_recent_enabled_in_range_values():
    master = FakePiper([0] * 6)
    master.gripper_command_mm = 12.5
    assert _fresh_gripper_mm(master, "master gripper command stream", command=True) == 12.5
    master.gripper_command_timestamp = time.time_ns() - 1_000_000_000
    with pytest.raises(RuntimeError, match="stale/slow"):
        _fresh_gripper_mm(master, "master gripper command stream", command=True)
    master.gripper_command_timestamp = 0
    with pytest.raises(RuntimeError, match="no timestamp"):
        _fresh_gripper_mm(master, "master gripper command stream", command=True)
    master.gripper_command_timestamp = None
    master.gripper_command_mm = 80
    with pytest.raises(RuntimeError, match="80.00 mm"):
        _fresh_gripper_mm(master, "master gripper command stream", command=True)
    master.gripper_command_mm = 12.5
    master.gripper_command_status = 0x00
    with pytest.raises(RuntimeError, match="disabled"):
        _fresh_gripper_mm(master, "master gripper command stream", command=True)
    master.gripper_command_status = 0x01
    master.gripper_command_set_zero = 0xAE
    with pytest.raises(RuntimeError, match="zero-setting"):
        _fresh_gripper_mm(master, "master gripper command stream", command=True)


def test_slave_gripper_feedback_guard_uses_its_physical_range():
    slave = FakePiper([0] * 6)
    slave.gripper_mm = -41.5
    assert _fresh_gripper_mm(slave, "slave gripper feedback") == -41.5
    slave.gripper_mm = 51
    with pytest.raises(RuntimeError, match="51.00 mm"):
        _fresh_gripper_mm(slave, "slave gripper feedback")


def test_mirror_refuses_another_sender_on_slave_bus():
    master = FakePiper([0, 10, -10, 0, 0, 0], command_hz=50)
    slave = FakePiper([0, 10, -10, 0, 0, 0], command_hz=50)
    with pytest.raises(RuntimeError, match="already has joint commands"):
        run_joint_mirror(master, slave, [0] * 6, 5, threading.Event())
    assert slave.targets == []


def test_mirror_refuses_large_initial_position_error():
    master = FakePiper([0, 100, -10, 0, 0, 0], command_hz=50)
    slave = FakePiper([0, 10, -10, 0, 0, 0])
    with pytest.raises(RuntimeError, match="start poses differ"):
        run_joint_mirror(master, slave, [0] * 6, 5, threading.Event())
    assert slave.targets == []


def test_mirror_uses_configured_start_and_tracking_gap():
    master = FakePiper([0] * 6, [25, 0, 0, 0, 0, 0], command_hz=50)
    slave = FakePiper([0] * 6)
    with pytest.raises(RuntimeError, match="more than 20°"):
        run_joint_mirror(master, slave, [0] * 6, 5, threading.Event())
    stop = threading.Event()
    run_joint_mirror(master, slave, [0] * 6, 5, stop, lambda _: stop.set(), max_joint_gap_deg=30)
    assert slave.targets == [(25000, 0, -2000, 0, -4000, 5000)]

    master.command_joints[0] = 15
    slave.joints[0] = 0
    def move_feedback_away(_sample):
        slave.joints[0] = -11
    with pytest.raises(RuntimeError, match="tracking error exceeds 25°"):
        run_joint_mirror(master, slave, [0] * 6, 5, threading.Event(), move_feedback_away,
                         max_joint_gap_deg=25)


@pytest.mark.parametrize("gap", [0, 4.9, 60.1, float("nan"), float("inf")])
def test_mirror_rejects_invalid_joint_gap(gap):
    with pytest.raises(ValueError, match="max joint gap must be 5–60 degrees"):
        run_joint_mirror(FakePiper([0] * 6), FakePiper([0] * 6), [0] * 6, 5,
                         threading.Event(), max_joint_gap_deg=gap)


def test_observed_encoder_zero_shift_and_alignment_offsets_allow_stream():
    master_joints = [0.47, -1.547, 0.861, 2.314, 14.429, -46.072]
    slave_joints = [0, -1.416, -0.789, 1.153, 18.202, 6.293]
    master = FakePiper([0] * 6, master_joints, command_hz=50)
    slave = FakePiper(slave_joints)
    mapped = map_joint_positions(master_joints, [0] * 6)[1]
    offsets = [float(np.clip(slave_value, *SLAVE_JOINT_PHYSICAL_LIMITS_DEG[i])) - mapped[i]
               for i, slave_value in enumerate(slave_joints)]
    stop = threading.Event()
    run_joint_mirror(master, slave, offsets, 5, stop, lambda _: stop.set())
    assert len(slave.targets) == 1
    assert slave.targets[0][5] == 6293


def test_master_joint_overshoot_is_capped_but_bad_slave_feedback_still_stops():
    master = FakePiper([0] * 6, [0, -2.1, 0, 0, 0, 0], command_hz=50)
    assert mapped_joint_target(master, [0] * 6)[1] == -1
    slave = FakePiper([0, -3.1, 0, 0, 0, 0])
    with pytest.raises(RuntimeError, match="slave joint feedback exceeds configured limits"):
        slave_measured_joints(slave)


def test_small_slave_j5_feedback_overshoot_does_not_stop_mirror():
    master = FakePiper([0] * 6, [0, 0, 0, 0, 74, 0], command_hz=50)
    slave = FakePiper([0, 0, -2, 0, 70.02, 0])
    stop = threading.Event()
    run_joint_mirror(master, slave, [0] * 6, 5, stop, lambda _: stop.set())
    assert slave.targets[0][4] == 70000
    assert hold_slave_position(slave)[4] == 70
    assert slave.targets[-1][4] == 70000
    slave.joints[4] = 72.1
    with pytest.raises(RuntimeError, match=r"J5=72\.10° outside feedback \[-80, 72\]°"):
        slave_measured_joints(slave)


def test_all_six_joints_map_endpoints_and_cap_master_overshoot():
    source_lo = [limits[0] for limits in MASTER_JOINT_PHYSICAL_LIMITS_DEG]
    source_hi = [limits[1] for limits in MASTER_JOINT_PHYSICAL_LIMITS_DEG]
    slave_lo = [limits[0] for limits in SLAVE_JOINT_PHYSICAL_LIMITS_DEG]
    slave_hi = [limits[1] for limits in SLAVE_JOINT_PHYSICAL_LIMITS_DEG]
    source_mid = [(lo + hi) / 2 for lo, hi in MASTER_JOINT_PHYSICAL_LIMITS_DEG]
    slave_mid = [(lo + hi) / 2 for lo, hi in SLAVE_JOINT_PHYSICAL_LIMITS_DEG]
    assert map_joint_positions(source_lo, [0] * 6)[2].tolist() == pytest.approx(slave_lo)
    assert map_joint_positions(source_mid, [0] * 6)[2].tolist() == pytest.approx(slave_mid)
    assert map_joint_positions(source_hi, [0] * 6)[2].tolist() == pytest.approx(slave_hi)
    assert map_joint_positions([value - 20 for value in source_lo], [0] * 6)[2].tolist() == pytest.approx(slave_lo)
    assert map_joint_positions([value + 20 for value in source_hi], [0] * 6)[2].tolist() == pytest.approx(slave_hi)
    assert map_joint_positions(source_hi, [20] * 6)[2].tolist() == pytest.approx(slave_hi)
    assert map_joint_positions(source_lo, [-20] * 6)[2].tolist() == pytest.approx(slave_lo)


def test_master_overshoot_stays_clamped_without_stopping_mirror():
    source_hi = [limits[1] + 10 for limits in MASTER_JOINT_PHYSICAL_LIMITS_DEG]
    slave_hi = [limits[1] for limits in SLAVE_JOINT_PHYSICAL_LIMITS_DEG]
    master = FakePiper([0] * 6, source_hi, command_hz=50)
    slave = FakePiper(slave_hi)
    stop = threading.Event()
    updates = []

    def on_update(sample):
        updates.append(sample)
        if len(updates) == 2:
            stop.set()

    run_joint_mirror(master, slave, [0] * 6, 5, stop, on_update)
    assert slave.targets == [tuple(int(value * 1000) for value in slave_hi)] * 2
    assert updates[-1]["clamped_master_joints"] == [f"J{i}" for i in range(1, 7)]


def test_j6_does_not_block_start_and_catches_up_with_five_degree_lead():
    master = FakePiper([0] * 6, [0, 0, 0, 0, 0, -162.1], command_hz=50)
    slave = FakePiper([0, 0, 0, 0, 0, 134.0])
    stop = threading.Event()
    updates = []

    def on_update(sample):
        updates.append(sample)
        stop.set()

    run_joint_mirror(master, slave, [0] * 6, 5, stop, on_update)
    assert slave.targets[0][5] == 129000
    assert updates[0]["desired_j6_deg"] == pytest.approx(-108.0)
    assert updates[0]["j6_remaining_deg"] == pytest.approx(-237.0)


def test_webui_mirror_sets_and_restores_all_sdk_joint_limits(monkeypatch):
    import piper_xvla.webui as webui
    from piper_xvla.webui import PiperWebUI

    class LimitedPiper(FakePiper):
        def __init__(self):
            super().__init__([0] * 6)
            self.sdk_limits = {f"j{i}": (-2.09439, 2.09439) for i in range(1, 7)}
        def GetSDKJointLimitParam(self, name): return self.sdk_limits[name]
        def SetSDKJointLimitParam(self, name, lo, hi): self.sdk_limits[name] = (lo, hi)

    ui = PiperWebUI(can="can0", dry_run=True)
    slave = LimitedPiper()
    ui.piper = slave
    ui._status_pipers["can_master"] = FakePiper([0] * 6)
    ui._master_slave_state = {"status": "starting", "sent": 0}
    observed = {}

    def run_stub(_, worker_slave, *args, **kwargs):
        observed["mirror_gripper"] = kwargs["mirror_gripper"]
        observed["max_joint_gap_deg"] = kwargs["max_joint_gap_deg"]
        observed["active_limits"] = dict(worker_slave.sdk_limits)

    monkeypatch.setattr(webui, "run_joint_mirror", run_stub)
    ui._master_slave_worker("can_master", "can0", [0] * 6, 5, 30)
    assert observed["mirror_gripper"] is True
    assert observed["max_joint_gap_deg"] == 30
    for index, limits in enumerate(SLAVE_JOINT_PHYSICAL_LIMITS_DEG, 1):
        assert observed["active_limits"][f"j{index}"] == pytest.approx(tuple(np.deg2rad(limits)))
    assert all(limit == (-2.09439, 2.09439) for limit in slave.sdk_limits.values())


def test_webui_mirror_supports_sdk_without_joint_limit_methods(monkeypatch):
    import piper_xvla.webui as webui
    from piper_xvla.webui import PiperWebUI

    ui = PiperWebUI(can="can0", dry_run=True)
    ui.piper = FakePiper([0] * 6)
    ui._status_pipers["can_master"] = FakePiper([0] * 6)
    ui._master_slave_state = {"status": "starting", "sent": 0}
    called = []
    monkeypatch.setattr(webui, "run_joint_mirror", lambda *args, **kwargs: called.append(kwargs["mirror_gripper"]))

    ui._master_slave_worker("can_master", "can0", [0] * 6, 5)

    assert called == [True]
    assert ui.master_slave_status()["status"] == "stopped"
    assert ui.master_slave_status()["sdk_joint_limits"] == "SDK limit API unavailable"


def test_webui_passes_configured_joint_gap_to_worker(monkeypatch):
    import piper_xvla.webui as webui
    from piper_xvla.webui import PiperWebUI

    ui = PiperWebUI(dry_run=True)
    monkeypatch.setattr(ui, "_require_manual_arm", lambda: None)
    monkeypatch.setattr(ui, "collection_active", lambda: False)
    monkeypatch.setattr(ui, "xvla_status", lambda: {"status": "idle"})
    monkeypatch.setattr(ui, "discover_can_devices", lambda: [
        {"name": "can_master", "state": "UP"}, {"name": "can0", "state": "UP"},
    ])

    class DormantThread:
        def __init__(self, *, target, args, **_):
            self.args = args
        def start(self):
            pass

    monkeypatch.setattr(webui.threading, "Thread", DormantThread)
    payload = {"master_can": "can_master", "slave_can": "can0", "speed_percent": 5,
               "max_joint_gap_deg": 32.5}
    with pytest.raises(ValueError, match="5–60 degrees"):
        ui.start_master_slave_control({**payload, "max_joint_gap_deg": 61})
    state = ui.start_master_slave_control(payload)
    assert state["max_joint_gap_deg"] == 32.5
    assert ui._master_slave_thread.args[-1] == 32.5


def test_hold_replaces_last_target_with_measured_slave_pose():
    slave = FakePiper([1.5, 20, -30, 0, 0, 0])
    slave.mode = 1
    assert hold_slave_position(slave) == [1.5, 20, -30, 0, 0, 0]
    assert slave.targets == [(1500, 20000, -30000, 0, 0, 0)]


def test_hold_clamps_feedback_within_slave_physical_range():
    slave = FakePiper([0, -1.5, -0.5, 0, 0, 0])
    slave.mode = 1
    assert hold_slave_position(slave) == [0, -1, -2, 0, 0, 0]
    assert slave.targets == [(0, -1000, -2000, 0, 0, 0)]


def test_mirror_stops_after_heartbeat_expires():
    master = FakePiper([0, 20, -30, 0, 0, 0], command_hz=50)
    slave = FakePiper([0, 20, -30, 0, 0, 0])
    lease = {"valid": True}

    def on_update(_sample):
        lease["valid"] = False

    with pytest.raises(RuntimeError, match="heartbeat expired"):
        run_joint_mirror(master, slave, [0] * 6, 5, threading.Event(), on_update,
                         lease_ok=lambda: lease["valid"])
    assert len(slave.targets) == 1


def test_stop_during_feedback_read_prevents_joint_send():
    stop = threading.Event()
    master = FakePiper([0, 20, -30, 0, 0, 0], command_hz=50)
    slave = FakePiper([0, 20, -30, 0, 0, 0])
    original = master.GetArmJointCtrl
    reads = 0

    def read_and_stop():
        nonlocal reads
        reads += 1
        if reads == 2:
            stop.set()
        return original()

    master.GetArmJointCtrl = read_and_stop
    run_joint_mirror(master, slave, [0] * 6, 5, stop)
    assert slave.targets == []


def test_webui_stop_holds_slave_pose():
    from piper_xvla.webui import PiperWebUI

    ui = PiperWebUI(dry_run=True)
    slave = FakePiper([1, 20, -30, 0, 0, 0])
    slave.mode = 1
    ui._master_slave_slave_piper = slave
    ui._master_slave_state = {"status": "running", "sent": 1}

    result = ui.stop_master_slave_control()

    assert result["ok"] is True
    assert ui._master_slave_stop.is_set()
    assert slave.targets == [(1000, 20000, -30000, 0, 0, 0)]


def test_webui_stop_requests_estop_when_hold_feedback_is_unavailable():
    from piper_xvla.webui import PiperWebUI

    ui = PiperWebUI(dry_run=True)
    slave = FakePiper([1, 20, -30, 0, 0, 0])
    slave.mode = 1
    slave.GetArmJointMsgs = lambda: SimpleNamespace(Hz=0, time_stamp=0)
    ui._master_slave_slave_piper = slave
    ui._master_slave_state = {"status": "running", "sent": 1}

    result = ui.stop_master_slave_control()

    assert result["ok"] is False
    assert ui.master_slave_status()["status"] == "error"
    assert slave.emergency_calls == [1]


def test_webui_heartbeat_requires_current_unexpired_session():
    from piper_xvla.webui import PiperWebUI

    ui = PiperWebUI(dry_run=True)
    ui._master_slave_state = {"status": "running", "session_id": "current"}
    ui._master_slave_lease_expires = time.monotonic() + 0.1
    with pytest.raises(PermissionError, match="does not match"):
        ui.renew_master_slave_control("old")
    assert ui.renew_master_slave_control("current")["lease_ms"] == 2000
    ui._master_slave_lease_expires = time.monotonic() - 0.1
    with pytest.raises(RuntimeError, match="expired"):
        ui.renew_master_slave_control("current")


def test_can_discovery_returns_names_and_usb_ports(monkeypatch, tmp_path):
    from piper_xvla.webui import PiperWebUI
    import piper_xvla.webui as webui

    root = tmp_path / "class" / "net"
    root.mkdir(parents=True)
    for name, usb_port in (("can_slave", "1-1.1:1.0"), ("can_master", "1-3:1.0")):
        target = tmp_path / "devices" / "usb1" / usb_port / "net" / name
        target.mkdir(parents=True)
        (target / "type").write_text("280\n")
        (target / "flags").write_text("0x40081\n")
        (root / name).symlink_to(target, target_is_directory=True)
    (root / "eth0").mkdir()
    (root / "eth0" / "type").write_text("1\n")
    monkeypatch.setattr(webui, "CAN_SYSFS_ROOT", root)
    assert PiperWebUI.discover_can_devices() == [
        {"name": "can_master", "state": "UP", "usb_port": "1-3:1.0"},
        {"name": "can_slave", "state": "UP", "usb_port": "1-1.1:1.0"},
    ]


def test_device_status_reads_master_command_stream_without_sending_commands(monkeypatch, tmp_path):
    from piper_xvla.webui import PiperWebUI

    ui = PiperWebUI(can="can_slave", dry_run=True, data_dir=tmp_path)
    master = FakePiper([0] * 6, [1, 20, -30, 4, 5, 6], command_hz=50)
    master.GetArmJointMsgs = lambda: SimpleNamespace(Hz=0)
    ui._status_pipers["can_master"] = master
    monkeypatch.setattr(ui, "discover_can_devices", lambda: [
        {"name": "can_master", "state": "UP", "usb_port": ""},
        {"name": "can_slave", "state": "UP", "usb_port": ""},
    ])

    status = ui.device_status("can_master")

    assert status["feedback_state"] == "LIVE"
    assert status["joint_source"] == "commanded"
    assert status["joints_deg"] == [1, 20, -30, 4, 5, 6]
    assert master.targets == []


def test_device_status_reports_each_joint_motor_and_driver_temperature(monkeypatch, tmp_path):
    from piper_xvla.webui import PiperWebUI

    ui = PiperWebUI(can="can0", dry_run=True, data_dir=tmp_path)
    slave = FakePiper([0] * 6)
    slave.GetArmLowSpdInfoMsgs = lambda: SimpleNamespace(Hz=50, **{
        f"motor_{i}": SimpleNamespace(motor_temp=30+i, foc_temp=40+i) for i in range(1, 7)
    })
    ui.piper = slave
    monkeypatch.setattr(ui, "discover_can_devices", lambda: [
        {"name": "can0", "state": "UP", "usb_port": ""},
    ])

    status = ui.device_status("can0")

    assert status["motor_temperatures_c"] == [31, 32, 33, 34, 35, 36]
    assert status["driver_temperatures_c"] == [41, 42, 43, 44, 45, 46]


def test_observer_connect_supports_both_sdk_signatures():
    from piper_xvla.webui import _connect_piper_observer

    class CurrentSDK:
        def __init__(self): self.calls = []
        def ConnectPort(self, piper_init=True): self.calls.append(piper_init)

    class LegacySDK:
        def __init__(self): self.calls = 0
        def ConnectPort(self): self.calls += 1

    current, legacy = CurrentSDK(), LegacySDK()
    _connect_piper_observer(current)
    _connect_piper_observer(legacy)
    assert current.calls == [False]
    assert legacy.calls == 1
