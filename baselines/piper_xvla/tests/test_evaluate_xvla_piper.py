import math

import torch

from piper_xvla.evaluate_xvla_piper import summarize_action_errors, summarize_action_safety


def test_summarize_action_errors_reports_physical_units():
    target = {
        "position": [[0.0, 0.0, 0.0]],
        "rotation6d": [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
        "gripper": [0.25],
    }
    prediction = {
        "position": [[0.01, 0.0, 0.0]],
        "rotation6d": [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
        "gripper": [0.75],
    }

    result = summarize_action_errors(prediction, target)

    assert math.isclose(result["position_rmse_m"], 0.01, rel_tol=1e-6)
    assert math.isclose(result["gripper_mae_normalized"], 0.5, rel_tol=1e-6)
    assert result["rotation6d_rmse"] == 0.0


def test_summarize_action_safety_flags_range_and_step_violations():
    predictions = torch.zeros((3, 20), dtype=torch.float32)
    predictions[:, 3] = 1.0
    predictions[:, 7] = 1.0
    predictions[:, 9] = torch.tensor([0.2, 0.3, 1.2])
    predictions[1, 0] = 0.01
    predictions[2, 0] = 0.10
    targets = torch.zeros((3, 20), dtype=torch.float32)
    targets[:, 3] = 1.0
    targets[:, 7] = 1.0

    report = summarize_action_safety(
        predictions, targets, episode_ids=torch.tensor([7, 7, 7]), hz=20.0,
    )

    assert report["non_finite_action_count"] == 0
    assert report["gripper_out_of_range_count"] == 1
    assert report["outside_target_xyz_envelope_count"] == 2
    assert math.isclose(report["max_predicted_xyz_step_m"], 0.09, rel_tol=1e-6)
    assert math.isclose(report["max_predicted_xyz_velocity_mps"], 1.8, rel_tol=1e-6)
