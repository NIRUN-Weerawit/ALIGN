from types import SimpleNamespace

from piper_xvla.readiness import assess_piper_readiness


def test_accepts_v2_sdk_status_wrapper_with_nested_arm_status():
    nested = SimpleNamespace(arm_status=0, ctrl_mode=0, teach_status=0, motion_status=0, err_code=0)
    piper = SimpleNamespace(
        isOk=lambda: True,
        GetCanFps=lambda: 100.0,
        GetArmStatus=lambda: SimpleNamespace(Hz=100.0, arm_status=nested),
        GetArmEndPoseMsgs=lambda: SimpleNamespace(Hz=100.0),
        GetArmJointMsgs=lambda: SimpleNamespace(Hz=100.0),
        GetArmGripperMsgs=lambda: SimpleNamespace(Hz=100.0),
    )

    report = assess_piper_readiness(piper, min_stream_hz=10.0)
    assert report.ready is True
    assert report.arm_status == 0
