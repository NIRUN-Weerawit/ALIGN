"""Regression tests for trajectory-evaluator action-head dispatch."""

import torch
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
