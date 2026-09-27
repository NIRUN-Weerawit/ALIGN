"""Offline held-out evaluation for a trained Piper X-VLA checkpoint."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from piper_xvla.train_xvla_piper import _open_source_dataset, _prepare, load_config, load_xvla_config, split_episode_ids
from piper_xvla.xvla_dataset import CachedPiperXVLALiberoDataset, load_prepared_numeric_cache


def summarize_action_errors(prediction: dict[str, list], target: dict[str, list]) -> dict[str, float]:
    """Return physical-unit-free action errors from already separated action fields."""
    pred_pos = torch.as_tensor(prediction["position"], dtype=torch.float64)
    true_pos = torch.as_tensor(target["position"], dtype=torch.float64)
    pred_rot = torch.as_tensor(prediction["rotation6d"], dtype=torch.float64)
    true_rot = torch.as_tensor(target["rotation6d"], dtype=torch.float64)
    pred_grip = torch.as_tensor(prediction["gripper"], dtype=torch.float64)
    true_grip = torch.as_tensor(target["gripper"], dtype=torch.float64)
    return {
        "position_rmse_m": float(torch.sqrt(torch.sum((pred_pos - true_pos) ** 2, dim=-1)).mean()),
        "rotation6d_rmse": float(torch.sqrt(torch.mean((pred_rot - true_rot) ** 2))),
        "gripper_mae_normalized": float(torch.mean(torch.abs(pred_grip - true_grip))),
    }



def _rotation6d_to_matrix(rotation6d: torch.Tensor) -> torch.Tensor:
    """Convert row-wise 6D rotations into right-handed rotation matrices."""
    first = rotation6d[..., [0, 2, 4]]
    second = rotation6d[..., [1, 3, 5]]
    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    first = first / first_norm.clamp_min(1e-8)
    second = second - first * torch.sum(first * second, dim=-1, keepdim=True)
    second = second / torch.linalg.vector_norm(second, dim=-1, keepdim=True).clamp_min(1e-8)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def _geodesic_degrees(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    relative = first.transpose(-1, -2) @ second
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cosine))


def summarize_action_safety(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    episode_ids: torch.Tensor,
    hz: float,
) -> dict[str, float | int | list[float]]:
    """Check model actions against observed labels without treating data bounds as hardware limits."""
    if predictions.ndim != 2 or predictions.shape != targets.shape or predictions.shape[1] != 20:
        raise ValueError("predictions and targets must have equal [N, 20] EE6D shapes")
    if len(episode_ids) != len(predictions) or hz <= 0:
        raise ValueError("episode IDs must align with predictions and hz must be positive")
    predictions = predictions.detach().float().cpu()
    targets = targets.detach().float().cpu()
    episode_ids = episode_ids.detach().cpu().view(-1)
    finite_rows = torch.isfinite(predictions).all(dim=1)
    xyz = predictions[:, :3]
    target_xyz = targets[:, :3]
    target_min, target_max = target_xyz.min(dim=0).values, target_xyz.max(dim=0).values
    outside_xyz = ((xyz < target_min) | (xyz > target_max)).any(dim=1)
    gripper = predictions[:, 9]
    sixd = predictions[:, 3:9]
    col0 = sixd[:, [0, 2, 4]]
    col1 = sixd[:, [1, 3, 5]]
    invalid_rotation = (torch.linalg.vector_norm(col0, dim=1) < 1e-6) | (torch.linalg.vector_norm(torch.linalg.cross(col0, col1, dim=1), dim=1) < 1e-6)

    xyz_steps: list[torch.Tensor] = []
    rotation_steps: list[torch.Tensor] = []
    gripper_steps: list[torch.Tensor] = []
    for episode_id in torch.unique_consecutive(episode_ids):
        indices = torch.where(episode_ids == episode_id)[0]
        if len(indices) < 2:
            continue
        current_xyz = xyz[indices]
        xyz_steps.append(torch.linalg.vector_norm(current_xyz[1:] - current_xyz[:-1], dim=1))
        current_rot = _rotation6d_to_matrix(sixd[indices])
        rotation_steps.append(_geodesic_degrees(current_rot[:-1], current_rot[1:]))
        gripper_steps.append(torch.abs(gripper[indices][1:] - gripper[indices][:-1]))

    def maximum(values: list[torch.Tensor]) -> float:
        return float(torch.cat(values).max()) if values else 0.0

    return {
        "non_finite_action_count": int((~finite_rows).sum()),
        "gripper_out_of_range_count": int(((gripper < 0.0) | (gripper > 1.0)).sum()),
        "rotation6d_invalid_count": int(invalid_rotation.sum()),
        "outside_target_xyz_envelope_count": int(outside_xyz.sum()),
        "target_xyz_min_m": target_min.tolist(),
        "target_xyz_max_m": target_max.tolist(),
        "predicted_xyz_min_m": xyz.min(dim=0).values.tolist(),
        "predicted_xyz_max_m": xyz.max(dim=0).values.tolist(),
        "max_predicted_xyz_step_m": maximum(xyz_steps),
        "max_predicted_xyz_velocity_mps": maximum(xyz_steps) * hz,
        "max_predicted_rotation_step_deg": maximum(rotation_steps),
        "max_predicted_gripper_step_normalized": maximum(gripper_steps),
    }


def _action_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute active-arm and inactive-arm errors for [N, 20] EE6D actions."""
    predicted = predicted.detach().float().cpu()
    target = target.detach().float().cpu()
    result = summarize_action_errors(
        {
            "position": predicted[:, :3].tolist(),
            "rotation6d": predicted[:, 3:9].tolist(),
            "gripper": predicted[:, 9].tolist(),
        },
        {
            "position": target[:, :3].tolist(),
            "rotation6d": target[:, 3:9].tolist(),
            "gripper": target[:, 9].tolist(),
        },
    )
    result["inactive_arm_rmse"] = float(torch.sqrt(torch.mean(predicted[:, 10:] ** 2)))
    return result


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0]
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def evaluate(config_path: str | Path, checkpoint_path: str | Path, *, device: str | None = None) -> dict[str, Any]:
    """Evaluate a custom ``torch.save`` X-VLA checkpoint on its held-out episodes."""
    values = load_config(config_path)
    device = device or values["device"]
    manifest = json.loads(Path(values["manifest"]).read_text())
    source_root = Path(manifest["source_dataset"])
    if not source_root.exists():
        local_candidate = Path(values["manifest"]).parent / "dataset"
        if local_candidate.exists():
            manifest["source_dataset"] = str(local_candidate.resolve())
    cache = load_prepared_numeric_cache(values["prepared_cache"], manifest)
    _, val_ids = split_episode_ids(manifest["source_episode_ids"], values["val_episodes"])
    source = _open_source_dataset(manifest["source_dataset"], val_ids)
    dataset = CachedPiperXVLALiberoDataset(source, cache)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from transformers import AutoTokenizer

    checkpoint_step = int(checkpoint["step"])
    config_payload = dict(checkpoint["xvla_config"])
    for feature_key in ("input_features", "output_features"):
        serialized = config_payload.get(feature_key, {})
        config_payload[feature_key] = {
            key: PolicyFeature(
                type=FeatureType(value["type"]),
                shape=tuple(value["shape"]),
            )
            for key, value in serialized.items()
        }
    xvla_config = XVLAConfig(**config_payload)
    xvla_config.device = device
    policy = XVLAPolicy(xvla_config)
    policy.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    policy.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(values["checkpoint"] if Path(values["checkpoint"]).exists() else "facebook/bart-large", local_files_only=True)

    loss_rows: list[dict[str, float]] = []
    action_rows: list[dict[str, float]] = []
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_episode_ids: list[torch.Tensor] = []
    episode_rows: dict[int, list[dict[str, float]]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            batch = _prepare(batch, tokenizer, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                _, log = policy(batch)
                predicted = policy.predict_action_chunk(batch)[:, 0]
            target = batch["action"][:, 0] if batch["action"].ndim == 3 else batch["action"]
            metrics = _action_metrics(predicted, target)
            action_rows.append(metrics)
            loss_rows.append({key: float(value) for key, value in log.items()})
            episode_tensor = batch["episode_index"].detach().cpu().view(-1)
            all_predictions.append(predicted.detach().cpu())
            all_targets.append(target.detach().cpu())
            all_episode_ids.append(episode_tensor)
            for episode_id in episode_tensor.tolist():
                episode_rows[int(episode_id)].append(metrics)

    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "step": checkpoint_step,
        "validation_episode_ids": val_ids,
        "frame_count": len(dataset),
        "loss": _mean_dict(loss_rows),
        "action_metrics": _mean_dict(action_rows),
        "action_safety": summarize_action_safety(
            torch.cat(all_predictions), torch.cat(all_targets),
            episode_ids=torch.cat(all_episode_ids), hz=20.0,
        ),
        "per_episode": {str(k): _mean_dict(v) for k, v in sorted(episode_rows.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="piper_xvla/config/piper_xvla_single_task.json")
    parser.add_argument("--checkpoint", default="outputs/piper_xvla_single_task/best.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="outputs/piper_xvla_single_task/offline_eval_best.json")
    args = parser.parse_args()
    result = evaluate(args.config, args.checkpoint, device=args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
