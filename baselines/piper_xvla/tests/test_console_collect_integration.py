"""End-to-end test of console `collect`: stub cameras, write a real LeRobot episode."""
import numpy as np
from types import SimpleNamespace

from piper_xvla.console import PiperConsole
from piper_xvla.replay_collector import ReplayObservation


class FakePiper:
    def __init__(self):
        self.calls = []

    def GetArmStatus(self):
        return SimpleNamespace(ctrl_mode=0x02, teach_status=0, motion_status=0, err_code=0)

    def MotionCtrl_1(self, emergency_stop=0, track_ctrl=0, grag_teach_ctrl=0):
        self.calls.append(("motion1", emergency_stop, track_ctrl, grag_teach_ctrl))

    def GetArmJointMsgs(self):
        return SimpleNamespace(joint_state=SimpleNamespace(**{f"joint_{i}": 0 for i in range(1, 7)}))


def _fake_adapter_factory(piper, task, config_path):
    """Return a stub adapter whose snapshot() yields tiny synthetic frames."""
    state = np.zeros(20, dtype=np.float32)

    class Stub:
        def __init__(self):
            self.closed = False

        def snapshot(self):
            img = np.zeros((4, 6, 3), dtype=np.uint8)
            return ReplayObservation(img, img.copy(), state.copy(), task)

        def close(self):
            self.closed = True

    return Stub()


def test_console_collect_stops_early_on_keyboard(capsys, monkeypatch):
    import piper_xvla.snapshot_adapter as sa
    import json

    monkeypatch.setattr(sa.PiperSnapshotAdapter, "from_camera_config", _fake_adapter_factory)
    import tempfile, os
    from pathlib import Path
    d = tempfile.mkdtemp()
    cfg_path = Path(os.path.join(d, "cam.json"))
    cfg_path.write_text(json.dumps({"global_camera": {"device": "/dev/null", "height": 4, "width": 6}}))
    monkeypatch.setattr(sa, "DEFAULT_CAMERA_CONFIG", cfg_path)

    import piper_xvla.console as console_mod
    # Simulate the user pressing Enter/typing 'stop' immediately.
    monkeypatch.setattr(console_mod.select, "select", lambda *a, **k: ([object()], [], []))
    class _Stdin:
        def readline(self):
            return "stop\n"
    monkeypatch.setattr(console_mod.sys, "stdin", _Stdin())
    monkeypatch.setattr("builtins.input", lambda *a: "yes")

    piper = FakePiper()
    console = PiperConsole(piper, task="put cube in basket", data_dir=os.path.join(d, "data"))
    console.cmd_collect("300")  # large cap; keyboard stop must end it quickly

    out = capsys.readouterr().out
    assert "stopped by user" in out, out
    teach_cmds = [c[3] for c in piper.calls if c[0] == "motion1"]
    assert 0x07 in teach_cmds and 0x03 in teach_cmds


def test_collect_session_saves_to_exact_selected_dataset_folder(tmp_path, monkeypatch):
    import json
    import piper_xvla.snapshot_adapter as sa
    from piper_xvla.collect_session import CollectSession

    monkeypatch.setattr(sa.PiperSnapshotAdapter, "from_camera_config", _fake_adapter_factory)
    cfg = tmp_path / "cam.json"
    cfg.write_text(json.dumps({"global_camera": {"height": 4, "width": 6}}))
    selected = tmp_path / "my episodes"
    selected.mkdir()

    session = CollectSession.open(
        FakePiper(), "put object in cup", tmp_path / "default", cfg,
        dataset_root=selected,
    )
    session.write_frame(session.snapshot())
    session.finalize()

    assert (selected / "meta" / "info.json").is_file()
    assert not (tmp_path / "default" / "dataset").exists()

    # A shell profile may point HF_HOME at an unplugged/read-only drive.
    from datasets import config as datasets_config
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("occupied")
    unavailable_cache = blocked / "datasets"
    writable_cache = tmp_path / "writable-cache"
    monkeypatch.setenv("HF_DATASETS_CACHE", str(unavailable_cache))
    monkeypatch.setenv("PIPER_XVLA_DATASETS_CACHE", str(writable_cache))
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", unavailable_cache)
    monkeypatch.setattr(datasets_config, "DOWNLOADED_DATASETS_PATH", unavailable_cache / "downloads")
    monkeypatch.setattr(datasets_config, "EXTRACTED_DATASETS_PATH", unavailable_cache / "downloads" / "extracted")

    # Older LeRobot must append from metadata without opening its full frame
    # cache. That cache can exceed the free space on a recording machine.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if not callable(getattr(LeRobotDataset, "resume", None)):
        def fail_full_load(*args, **kwargs):
            raise AssertionError("full frame cache should not be loaded while resuming")
        monkeypatch.setattr(LeRobotDataset, "load_hf_dataset", fail_full_load)

    resumed = CollectSession.open(
        FakePiper(), "put object in cup", tmp_path / "default", cfg,
        dataset_root=selected,
    )
    assert resumed.episode_index == 1
    resumed.write_frame(resumed.snapshot())
    resumed.finalize()

    assert json.loads((selected / "meta" / "info.json").read_text())["total_episodes"] == 2
    assert datasets_config.HF_DATASETS_CACHE == writable_cache
    from piper_xvla.webui import _episode_state_action, _episode_thumbnails, _load_dataset_cached
    ds = _load_dataset_cached({}, selected)
    assert ds.num_episodes == 2
    assert ds.num_frames == 2
    assert len(_episode_thumbnails(ds, 1)) == 1
    assert _episode_state_action(ds, 1)["indices"] == [1]


def test_console_collect_second_run_resumes_and_appends(tmp_path, monkeypatch, capsys):
    """Re-running collect must append a new episode, not fail with File-exists."""
    import piper_xvla.snapshot_adapter as sa
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    monkeypatch.setattr(sa.PiperSnapshotAdapter, "from_camera_config", _fake_adapter_factory)
    import json
    cfg = tmp_path / "cam.json"
    cfg.write_text(json.dumps({"global_camera": {"device": "/dev/null", "height": 4, "width": 6}}))
    monkeypatch.setattr(sa, "DEFAULT_CAMERA_CONFIG", cfg)
    monkeypatch.setattr("builtins.input", lambda *a: "yes")

    piper = FakePiper()
    console = PiperConsole(piper, task="put cube in basket", data_dir=tmp_path / "data")

    console.cmd_collect("6")
    first_episodes = LeRobotDataset(str(tmp_path / "data" / "dataset")).num_episodes
    assert first_episodes == 1

    # Second run: must resume, not raise File-exists.
    capsys.readouterr()
    console.cmd_collect("6")
    second_out = capsys.readouterr().out
    assert "resuming existing dataset" in second_out, second_out
    assert "File exists" not in second_out

    total_episodes = LeRobotDataset(str(tmp_path / "data" / "dataset")).num_episodes
    assert total_episodes == first_episodes + 1


def test_console_collect_guard_on_corrupt_existing_dataset(tmp_path, monkeypatch, capsys):
    """An existing but unopenable dataset (half-written) must give a clean recovery hint."""
    import piper_xvla.snapshot_adapter as sa
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    monkeypatch.setattr(sa.PiperSnapshotAdapter, "from_camera_config", _fake_adapter_factory)
    import json
    cfg = tmp_path / "cam.json"
    cfg.write_text(json.dumps({"global_camera": {"device": "/dev/null", "height": 4, "width": 6}}))
    monkeypatch.setattr(sa, "DEFAULT_CAMERA_CONFIG", cfg)
    monkeypatch.setattr("builtins.input", lambda *a: "yes")

    # Simulate a leftover dataset that exists on disk but cannot be resumed.
    ds_dir = tmp_path / "data" / "dataset"
    (ds_dir / "meta").mkdir(parents=True)
    (ds_dir / "meta" / "info.json").write_text("{}")
    if callable(getattr(LeRobotDataset, "resume", None)):
        monkeypatch.setattr(LeRobotDataset, "resume",
                            classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(RuntimeError("corrupt"))))
    else:
        import piper_xvla.collect_session as collect_session
        monkeypatch.setattr(collect_session, "_resume_legacy_dataset",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("corrupt")))

    piper = FakePiper()
    console = PiperConsole(piper, task="put cube in basket", data_dir=tmp_path / "data")
    console.cmd_collect("6")
    out = capsys.readouterr().out
    assert "could not be opened" in out, out
    assert "dataset was left untouched" in out and str(ds_dir) in out, out
    assert (ds_dir / "meta" / "info.json").read_text() == "{}"
    # No CAN command was sent: it bailed before arming capture/replay.
    assert piper.calls == []


def test_console_collect_writes_a_readable_lerobot_episode(tmp_path, monkeypatch, capsys):
    import piper_xvla.snapshot_adapter as sa
    from piper_xvla.lerobot_factory import piper_lerobot_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Stub the camera open so no /dev/video* is needed.
    monkeypatch.setattr(sa.PiperSnapshotAdapter, "from_camera_config", _fake_adapter_factory)
    # Point the config read inside cmd_collect at a temp file with tiny dims.
    import json
    cfg = tmp_path / "cam.json"
    cfg.write_text(json.dumps({"global_camera": {"device": "/dev/null", "height": 4, "width": 6}}))
    monkeypatch.setattr(sa, "DEFAULT_CAMERA_CONFIG", cfg)

    piper = FakePiper()
    console = PiperConsole(piper, task="put cube in basket", data_dir=tmp_path / "data")
    # Accept the move-arm confirmation.
    monkeypatch.setattr("builtins.input", lambda *a: "yes")

    console.cmd_collect("6")  # 6s window @20Hz -> ~120 frames

    out = capsys.readouterr().out
    assert "episode written" in out, out

    dataset_dir = tmp_path / "data" / "dataset"
    assert (dataset_dir / "meta" / "info.json").is_file()
    assert (dataset_dir / "data").exists()

    # A combined global|wrist review video must have been written to images/review/.
    review_videos = sorted((dataset_dir / "images" / "review").glob("episode-*.mp4"))
    assert len(review_videos) == 1, review_videos
    assert review_videos[0].stat().st_size > 0
    from piper_xvla.review_video import probe_codec
    assert probe_codec(review_videos[0]) == "h264"
    assert "review video:" in out

    # Read it back: the episode must contain frames with our task text.
    ds = LeRobotDataset(str(dataset_dir))
    assert len(ds) > 0
    sample = ds[0]
    assert sample["task"] == "put cube in basket"
    assert sample["observation.state"].shape[-1] == 10
    # Replay trigger (0x07 move-to-start + 0x03 execute) must have been sent.
    teach_cmds = [c[3] for c in piper.calls if c[0] == "motion1"]
    assert 0x07 in teach_cmds and 0x03 in teach_cmds
