"""Regression tests for trajectory-evaluator action-head dispatch."""

import torch

from eval.eval_libero_v4_trajectory import _predict_action_chunk


class DummyModel:
    def __init__(self, head_type):
        self.head_type = head_type
        self.sample_calls = 0
        self.predict_calls = 0

    def sample_actions(self, z_v, z_s, intent):
        self.sample_calls += 1
        return torch.zeros(1, 10, 7)

    def predict_actions(self, z_v, z_s, intent):
        self.predict_calls += 1
        return torch.zeros(1, 10, 7)


def test_flow_matching_uses_sampling_path():
    model = DummyModel("flow_matching")
    output = _predict_action_chunk(model, None, None, None)

    assert output.shape == (1, 10, 7)
    assert model.sample_calls == 1
    assert model.predict_calls == 0


def test_direct_head_uses_prediction_path():
    model = DummyModel("transformer")
    output = _predict_action_chunk(model, None, None, None)

    assert output.shape == (1, 10, 7)
    assert model.sample_calls == 0
    assert model.predict_calls == 1
