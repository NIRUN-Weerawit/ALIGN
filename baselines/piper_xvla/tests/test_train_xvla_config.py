import json
import torch

from piper_xvla.train_xvla_piper import (
    apply_xvla_finetuning_config,
    build_xvla_optimizer,
    configure_cuda_attention,
    load_config,
    load_xvla_config,
    serialize_policy_config,
    status,
)


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


def test_load_xvla_config_accepts_the_checkpoint_type_discriminator():
    config = load_xvla_config("/media/ucluser/PortableSSD/hf_models/xvla-libero")

    assert config.type == "xvla"


def test_configure_cuda_attention_disables_only_cudnn_sdpa(monkeypatch):
    calls = []

    class Backend:
        @staticmethod
        def enable_cudnn_sdp(enabled):
            calls.append(enabled)

    monkeypatch.setattr("piper_xvla.train_xvla_piper.torch.backends.cuda", Backend())

    configure_cuda_attention("cuda")

    assert calls == [False]


def test_serialize_policy_config_falls_back_to_dataclass_fields_when_to_dict_is_absent():
    from dataclasses import dataclass

    @dataclass
    class LegacyXVLAConfig:
        action_mode: str = "ee6d"
        chunk_size: int = 1

    assert serialize_policy_config(LegacyXVLAConfig()) == {"action_mode": "ee6d", "chunk_size": 1}


def test_fused_xvla_optimizer_keeps_preset_parameter_groups_and_learning_rates():
    from lerobot.optim.optimizers import XVLAAdamWConfig

    parameters = {
        "model.vlm.weight": torch.nn.Parameter(torch.ones(1)),
        "model.soft_prompt.weight": torch.nn.Parameter(torch.ones(1)),
        "model.head.weight": torch.nn.Parameter(torch.ones(1)),
    }
    preset = XVLAAdamWConfig(lr=1e-4, soft_prompt_lr_scale=0.5)

    optimizer = build_xvla_optimizer(preset, parameters, "cuda:0")

    assert isinstance(optimizer, torch.optim.AdamW)
    assert [(group["name"], group["lr"]) for group in optimizer.param_groups] == [
        ("vlm", 1e-5), ("soft_prompts", 5e-5), ("other", 1e-4),
    ]
    assert all(group["fused"] is True for group in optimizer.param_groups)
    assert build_xvla_optimizer(preset, parameters, "cpu").defaults["fused"] is None
