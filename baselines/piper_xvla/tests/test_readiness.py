from types import SimpleNamespace

from piper_xvla.readiness import assess_piper_readiness


def healthy_piper():
    return SimpleNamespace(
        isOk=lambda: True,
        GetCanFps=lambda: 120.0,
        GetArmStatus=lambda: SimpleNamespace(Hz=100.0, arm_status=0, ctrl_mode=0x07, teach_status=0, motion_status=0, err_code=0),
        GetArmEndPoseMsgs=lambda: SimpleNamespace(Hz=100.0),
        GetArmJointMsgs=lambda: SimpleNamespace(Hz=100.0),
        GetArmGripperMsgs=lambda: SimpleNamespace(Hz=50.0),
    )


def test_ready_when_connection_is_healthy_and_all_required_streams_are_live():
    report = assess_piper_readiness(healthy_piper(), min_stream_hz=10.0)
    assert report.ready is True
    assert report.errors == []


def test_not_ready_when_can_or_sensor_stream_is_missing():
    piper = healthy_piper()
    piper.GetCanFps = lambda: 0.0
    piper.GetArmJointMsgs = lambda: SimpleNamespace(Hz=2.0)

    report = assess_piper_readiness(piper, min_stream_hz=10.0)
    assert report.ready is False
    assert any("CAN" in message for message in report.errors)
    assert any("joint" in message for message in report.errors)


def test_not_ready_when_arm_reports_fault_or_emergency_stop():
    piper = healthy_piper()
    piper.GetArmStatus = lambda: SimpleNamespace(Hz=100.0, arm_status=1, ctrl_mode=0x07, teach_status=0, motion_status=0, err_code=0)

    report = assess_piper_readiness(piper, min_stream_hz=10.0)
    assert report.ready is False
    assert any("arm_status" in message for message in report.errors)
