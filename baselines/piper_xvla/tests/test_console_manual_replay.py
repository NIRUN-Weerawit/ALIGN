from types import SimpleNamespace

from piper_xvla.console import PiperConsole


class FakePiper:
    def __init__(self):
        self.calls = []

    def GetArmStatus(self):
        return SimpleNamespace(ctrl_mode=0x02, teach_status=0, motion_status=0, err_code=0)

    def MotionCtrl_2(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_console_rejects_host_offline_mode_and_directs_user_to_pendant(capsys):
    piper = FakePiper()
    PiperConsole(piper).cmd_mode("offline")
    assert piper.calls == []
    assert "pendant replay-mode button" in capsys.readouterr().out


def test_console_workflow_prints_pendant_then_host_collect_sequence(capsys):
    piper = FakePiper()
    PiperConsole(piper).cmd_workflow()
    output = capsys.readouterr().out
    assert "Pendant: record and save" in output
    assert "Host: collect" in output
    assert piper.calls == []


def test_console_task_set_and_get(capsys):
    console = PiperConsole(FakePiper())
    console.cmd_task("put the cube in the basket")
    out = capsys.readouterr().out
    assert "task set to" in out
    assert console.task == "put the cube in the basket"
    console.cmd_task("")  # no arg -> show current
    assert "put the cube in the basket" in capsys.readouterr().out


def test_console_collect_window_defaults_and_bounds():
    assert PiperConsole._collect_window_seconds("") == 40.0
    assert PiperConsole._collect_window_seconds("12") == 12.0
    for bad in ("3", "500"):
        try:
            PiperConsole._collect_window_seconds(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for window {bad!r}")


def test_console_collect_refuses_without_task(capsys):
    piper = FakePiper()
    console = PiperConsole(piper)  # task unset
    console.cmd_collect("10")
    out = capsys.readouterr().out
    assert "No task set" in out
    assert piper.calls == []  # no CAN command sent before the task gate


def test_console_collect_refuses_without_motion_confirm(capsys, monkeypatch):
    piper = FakePiper()
    console = PiperConsole(piper, task="put cube in basket")
    monkeypatch.setattr("builtins.input", lambda *a: "no")
    console.cmd_collect("10")
    assert piper.calls == []  # declined the move-arm confirmation


def test_replay_observation_window_defaults_to_six_seconds_and_accepts_override():
    assert PiperConsole._replay_observation_seconds("") == 6.0
    assert PiperConsole._replay_observation_seconds("10") == 10.0
