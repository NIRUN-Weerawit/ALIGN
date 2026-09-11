from types import SimpleNamespace

import pytest

from piper_xvla.replay_control import PiperReplayController


class FakePiper:
    def __init__(self, ctrl_mode=0x02):
        self.ctrl_mode = ctrl_mode
        self.calls = []

    def MotionCtrl_1(self, emergency_stop=0, track_ctrl=0, grag_teach_ctrl=0):
        self.calls.append(("motion1", emergency_stop, track_ctrl, grag_teach_ctrl))

    def MotionCtrl_2(self, ctrl_mode=0, move_mode=0, move_spd_rate_ctrl=0,
                     is_mit_mode=0, residence_time=0, installation_pos=0):
        self.calls.append(("motion2", ctrl_mode, move_mode, move_spd_rate_ctrl,
                           is_mit_mode, residence_time, installation_pos))

    def EmergencyStop(self, command):
        self.calls.append(("estop", command))

    def GetArmStatus(self):
        return SimpleNamespace(ctrl_mode=self.ctrl_mode)

    def GetArmJointMsgs(self):
        # Stationary arm: settle check returns immediately.
        class _J:
            joint_state = SimpleNamespace(**{f"joint_{i}": 0 for i in range(1, 7)})
        return _J()


def test_replay_sends_move_to_start_then_execute():
    # The pendant replay button does 0x07 (move to start) then 0x03 (execute).
    # A bare 0x03 only works if the arm was already primed at the start pose.
    piper = FakePiper(ctrl_mode=0x02)
    PiperReplayController(piper, settle_s=0).start_replay()
    assert piper.calls == [
        ("motion1", 0x00, 0x00, 0x07),
        ("motion1", 0x00, 0x00, 0x03),
    ]


def test_replay_does_not_change_control_mode():
    # Drag-teach reproduction runs from the controller's current mode; no 0x151 frames.
    piper = FakePiper(ctrl_mode=0x02)
    PiperReplayController(piper, settle_s=0).start_replay()
    assert not any(c[0] == "motion2" for c in piper.calls)


def test_resume_and_move_to_start_send_documented_commands_without_status_gate():
    piper = FakePiper(ctrl_mode=0x02)
    control = PiperReplayController(piper)
    control.resume_replay()
    control.move_to_start()
    assert piper.calls == [
        ("motion1", 0x00, 0x00, 0x05),
        ("motion1", 0x00, 0x00, 0x07),
    ]


def test_discard_clears_only_current_trajectory():
    piper = FakePiper()
    PiperReplayController(piper).discard_current_recording()
    assert piper.calls == [("motion1", 0x00, 0x03, 0x00)]


def test_emergency_stop_never_moves_the_arm_back_to_start():
    piper = FakePiper()
    PiperReplayController(piper).emergency_stop()
    assert piper.calls == [("estop", 0x01)]


def test_recovery_in_teaching_mode_resumes_then_moves_to_start():
    piper = FakePiper(ctrl_mode=0x02)
    PiperReplayController(piper).recover_and_move_to_start()
    assert piper.calls == [
        ("estop", 0x02),
        ("motion1", 0x00, 0x00, 0x07),
    ]


def test_recovery_in_pendant_selected_offline_mode_moves_to_start():
    piper = FakePiper(ctrl_mode=0x07)
    PiperReplayController(piper).recover_and_move_to_start()
    assert piper.calls == [
        ("estop", 0x02),
        ("motion1", 0x00, 0x00, 0x07),
    ]
