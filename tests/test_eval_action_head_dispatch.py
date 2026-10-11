"""Regression tests for trajectory-evaluator action-head dispatch."""

import torch
import numpy as np
import pytest
import queue
import threading

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


def test_text_condition_reaches_generative_sampler():
    class TextModel(DummyModel):
        def sample_actions(self, z_v, z_s, intent, *, text_emb=None):
            assert text_emb is marker
            return super().sample_actions(z_v, z_s, intent)

    marker = torch.ones(1, 8)
    model = TextModel("diffusion")
    assert _predict_action_chunk(model, None, None, None, marker).shape == (1, 10, 7)
    assert model.sample_calls == 1


@pytest.mark.parametrize("head_type", ["diffusion", "flow_matching"])
@pytest.mark.parametrize("use_text", [False, True])
def test_sim_rollout_passes_text_and_conditions_memory_without_intent(monkeypatch, head_type, use_text):
    from types import SimpleNamespace
    import eval.eval_libero_v4_trajectory as evaluator
    monkeypatch.setattr(evaluator, "get_sim_eef_pose", lambda obs, rotation_convention="libero": np.zeros(6, np.float32))
    monkeypatch.setattr(evaluator, "get_sim_frame", lambda *args, **kwargs: np.zeros((4, 4, 3), np.uint8))
    marker = torch.ones(1, 8)

    class Env:
        def reset(self):
            return {}

        def step(self, action):
            return {}, 0.0, False, {}

    class Model(DummyModel):
        use_memory_bank = True
        memory_mode = "legacy"

        def __init__(self):
            super().__init__(head_type)
            self.text_condition = object() if use_text else None
            self.memory_module = SimpleNamespace(reset=lambda **kwargs: None)
            self.condition_calls = 0

        def encode_task_text(self, texts):
            assert texts == ["pick up the cup"]
            return marker

        def __call__(self, frames, states):
            return {"z_v_pooled_seq": states.new_zeros(1, 1, 4),
                    "z_s_seq": states, "h_seq": states.new_zeros(1, 1, 1),
                    "intent_emb": None}

        def condition_actions(self, visual, states, intent):
            assert intent is None
            self.condition_calls += 1
            return visual + 1, states, intent

        def sample_actions(self, visual, states, intent, *, text_emb=None):
            assert torch.all(visual == 1)  # Retrieved features reach the head.
            assert text_emb is (marker if use_text else None)
            return super().sample_actions(visual, states, intent)

    model = Model()
    evaluator.run_model_in_sim(Env(), model, torch.device("cpu"), np.zeros((2, 7), np.float32),
        chunk_size=4, max_steps=2, switch_at=0.0, use_camera=["wrist_image"],
        task_text="pick up the cup")
    assert model.condition_calls == model.sample_calls == 2


def test_direct_head_uses_prediction_path():
    model = DummyModel("transformer")
    output = _predict_action_chunk(model, None, None, None)

    assert output.shape == (1, 10, 7)
    assert model.sample_calls == 0
    assert model.predict_calls == 1


def test_async_recurrence_consumes_every_observation_before_planning():
    from eval.eval_libero_v4_async import InferenceWorker
    stop = threading.Event()

    class StreamingModel(DummyModel):
        intention_encoder = object()
        history_size = 2

        def __init__(self):
            super().__init__("flow_matching")
            self.observations = []

        def encode_step(self, frame, state, cache, produce_intent):
            self.observations.append((int(frame.item()), cache, produce_intent))
            output = (state, state, state, len(self.observations))
            return output + (state.unsqueeze(1),) if produce_intent else output

        def condition_actions(self, visual, state, intent):
            return visual, state, intent

        def sample_actions(self, visual, state, intent):
            stop.set()
            return super().sample_actions(visual, state, intent)

    model = StreamingModel()
    observations, actions = queue.Queue(), queue.Queue(maxsize=1)
    for i in range(4):
        observations.put((torch.tensor([[i]]), torch.ones(1, 4) * i))
    worker = InferenceWorker(model, torch.device("cpu"), 2, observations, actions, stop)
    worker.run()
    assert model.observations == [(0, None, False), (1, 1, False), (2, 2, False), (3, 3, True)]
    assert model.sample_calls == 1
    assert actions.get_nowait()[0].shape == (2, 7)


def test_episode_seed_is_independent_of_episode_order_and_changes_with_trial_seed():
    from eval.eval_libero_v4_trajectory import episode_random_seed
    keys=['ep_000010','ep_000033','ep_000063']
    forward={k:episode_random_seed(42,k) for k in keys}
    backward={k:episode_random_seed(42,k) for k in reversed(keys)}
    assert forward==backward and len(set(forward.values()))==len(keys)
    assert all(episode_random_seed(43,k)!=forward[k] for k in keys)
    assert all(0<=v<2**31 for v in forward.values())
