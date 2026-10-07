"""Action-head dispatch for ALIGN intention checkpoint evaluation."""

import torch

from eval.eval_intention import _predict_action_chunk


class DummyModel:
    def __init__(self, head_type):
        self.head_type = head_type
        self.calls = []

    def sample_actions(self, z_v, z_s, intent):
        self.calls.append(("sample", intent))
        return torch.zeros(1, 10, 7)

    def predict_actions(self, z_v, z_s, intent):
        self.calls.append(("predict", intent))
        return torch.zeros(1, 10, 7)


def test_flow_matching_samples_with_intent_tokens():
    model = DummyModel("flow_matching")
    intent = torch.ones(1, 1, 512)
    out = {"z_v_pooled_seq": None, "z_s_seq": None, "intent_emb": intent}

    assert _predict_action_chunk(model, out).shape == (1, 10, 7)
    assert model.calls == [("sample", intent)]


def test_transformer_predicts_without_intent_tokens():
    model = DummyModel("transformer")
    out = {"z_v_pooled_seq": None, "z_s_seq": None, "intent_emb": None}

    assert _predict_action_chunk(model, out).shape == (1, 10, 7)
    assert model.calls == [("predict", None)]
