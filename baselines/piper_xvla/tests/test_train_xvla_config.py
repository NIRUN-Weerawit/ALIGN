import json

from piper_xvla.train_xvla_piper import apply_xvla_finetuning_config, load_config, status


def test_load_config_reads_every_training_setting_from_json(tmp_path):
    path = tmp_path / "config.json"
    payload = {
        "manifest": "/data/manifest.json", "prepared_cache": "/data/prepared.pt",
        "checkpoint": "/models/xvla", "output": "/output",
        "steps": 20_000, "batch_size": 4, "val_episodes": 4, "device": "cuda",
        "action_mode": "ee6d", "freeze_vision_encoder": False,
        "freeze_language_encoder": False, "train_policy_transformer": True,
        "train_soft_prompts": True, "optimizer_lr": 1e-4,
        "scheduler_warmup_steps": 1_000, "scheduler_decay_steps": 20_000,
        "scheduler_decay_lr": 2.5e-6, "validate_every_steps": 1_000,
        "save_every_steps": 1_000,
    }
    path.write_text(json.dumps(payload))

    assert load_config(path) == payload


def test_apply_xvla_finetuning_config_explicitly_enables_official_new_embodiment_settings():
    class Config:
        pass

    config = Config()
    values = {
        "device": "cuda", "action_mode": "ee6d", "optimizer_lr": 1e-4,
        "scheduler_warmup_steps": 1_000, "scheduler_decay_steps": 20_000,
        "scheduler_decay_lr": 2.5e-6, "freeze_vision_encoder": False,
        "freeze_language_encoder": False, "train_policy_transformer": True,
        "train_soft_prompts": True,
    }

    apply_xvla_finetuning_config(config, values)

    assert config.device == "cuda"
    assert config.dtype == "bfloat16"
    assert config.action_mode == "ee6d"
    assert config.chunk_size == config.n_action_steps == 1
    assert config.tokenizer_max_length == 64
    for key in ("freeze_vision_encoder", "freeze_language_encoder"):
        assert getattr(config, key) is False
    for key in ("train_policy_transformer", "train_soft_prompts"):
        assert getattr(config, key) is True
    assert config.optimizer_lr == 1e-4
    assert config.scheduler_decay_steps == 20_000


def test_status_prints_an_explicit_flushed_stage_marker(capsys):
    status(3, 6, "attaching prepared labels")

    assert capsys.readouterr().out == "[train stage 3/6] attaching prepared labels\n"
