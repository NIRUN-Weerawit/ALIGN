from types import SimpleNamespace

from piper_xvla.readiness import assess_piper_readiness


def test_teaching_controller_mode_is_reported_even_when_drag_teach_status_is_disabled():
    status = SimpleNamespace(arm_status=0, ctrl_mode=0x02, teach_status=0, motion_status=0, err_code=0)
    piper = SimpleNamespace(
        isOk=lambda: True,
        GetCanFps=lambda: 100.0,
        GetArmStatus=lambda: SimpleNamespace(Hz=100.0, arm_status=status),
        GetArmEndPoseMsgs=lambda: SimpleNamespace(Hz=100.0),
        GetArmJointMsgs=lambda: SimpleNamespace(Hz=100.0),
        GetArmGripperMsgs=lambda: SimpleNamespace(Hz=100.0),
    )
    report = assess_piper_readiness(piper, min_stream_hz=10.0)
    assert report.ready is True
    assert any("teaching control mode" in warning for warning in report.warnings)
