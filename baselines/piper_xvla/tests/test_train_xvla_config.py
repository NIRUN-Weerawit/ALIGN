import json

from piper_xvla.train_xvla_piper import load_config


def test_load_config_reads_every_training_setting_from_json(tmp_path):
    path = tmp_path / "config.json"
    payload = {
        "manifest": "/data/manifest.json", "checkpoint": "/models/xvla", "output": "/output",
        "epochs": 3, "batch_size": 2, "val_episodes": 4, "lr": 1e-5, "device": "cuda",
    }
    path.write_text(json.dumps(payload))

    assert load_config(path) == payload
