"""Native control and chunk contracts, independent of weights and MuJoCo."""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from eval.eval_xvla_libero_native import (NativeChunkPolicy, absolute_sim_action,
    episode_seed, load_cases, reset_case, rollout, summarize)
from eval.xvla_libero_adapter import XVLALiberoAdapter


def target(x=0., rotation=None, close=1.):
    r = np.eye(3) if rotation is None else rotation
    return np.r_[x, 0., 0., r[:, 0], r[:, 1], close, np.zeros(10)]


def test_native_absolute_target_is_never_delta_scaled_or_clipped():
    r = Rotation.from_euler('xyz', [.3, -.2, .5])
    action = absolute_sim_action(target(.15, r.as_matrix(), .51))
    np.testing.assert_allclose(action[:3], [.15, 0., 0.])
    np.testing.assert_allclose(Rotation.from_rotvec(action[3:6]).as_matrix(), r.as_matrix(), atol=1e-7)
    assert action[6] == 1.
    assert absolute_sim_action(target(close=.5))[6] == -1.
    with pytest.raises(ValueError):
        absolute_sim_action(np.zeros(7))


class Adapter:
    chunk_size = 3
    def __init__(self):
        self.inputs = []
    def reset_episode(self):
        self.inputs.clear()
    def predict_sim_action_chunk(self, obs, *, task_text, env, proprio=None):
        assert task_text == 'move bowl'
        self.inputs.append(None if proprio is None else proprio.copy())
        return np.array([[target(.1), target(.2), target(.3)]], dtype=np.float32)


class Env:
    def __init__(self):
        self.controller = SimpleNamespace(use_delta=True, control_dim=6,
            update=lambda: None, ee_pos=np.array([.01, .02, .03]), ee_ori_mat=np.eye(3))
        self.robots = [SimpleNamespace(controller=self.controller)]
        self.calls = []
        self.position = np.zeros(3)
    def seed(self, seed):
        self.calls.append(('seed', seed))
    def obs(self):
        return {'robot0_eef_pos': self.position.copy()}
    def reset(self):
        self.calls.append(('reset', self.controller.use_delta))
        self.position[:] = 0
        return self.obs()
    def set_init_state(self, state):
        self.position[:] = state
        self.calls.append(('restore', state.copy()))
        return self.obs()
    def get_sim_state(self):
        return self.position.copy()
    def step(self, action):
        self.calls.append(('step', self.controller.use_delta, action.copy()))
        if not self.controller.use_delta:
            self.position = action[:3].copy()
        return self.obs(), 0., False, {}
    def check_success(self):
        return False


def test_settle_happens_before_absolute_mode_and_restore_is_repeatable():
    env = Env()
    env.controller.use_delta = False
    obs, state = reset_case(env, 42, initial_state=np.array([.4, .5, .6]), settle_steps=10)
    assert env.calls[1] == ('reset', True)
    steps = [row for row in env.calls if row[0] == 'step']
    assert len(steps) == 10 and all(row[1] is True for row in steps)
    for row in steps:
        np.testing.assert_array_equal(row[2], [0, 0, 0, 0, 0, 0, -1])
    assert env.controller.use_delta is False
    np.testing.assert_array_equal(state, [.4, .5, .6])
    np.testing.assert_array_equal(obs['robot0_eef_pos'], state)


@pytest.mark.parametrize('mode', ['author', 'measured'])
def test_full_chunk_execution_and_proprio_state_reset(mode):
    env, adapter = Env(), Adapter()
    obs, _ = reset_case(env, 42)
    policy = NativeChunkPolicy(adapter, mode)
    outcome, trace = rollout(env, policy, obs, 'move bowl', 4)
    assert outcome['policy_calls'] == 2 and outcome['n_steps'] == 4
    np.testing.assert_allclose(trace['actions'][:, 0], [.1, .2, .3, .1])
    assert trace['pre_action_positions'].shape == (4, 3)
    assert outcome['termination'] == 'step_limit' and not outcome['success']
    if mode == 'author':
        np.testing.assert_allclose(adapter.inputs[0][:3], env.controller.ee_pos)
        np.testing.assert_allclose(adapter.inputs[1], np.r_[target(.3)[:9], 0, np.zeros(10)])
    else:
        assert adapter.inputs == [None, None]
    policy.reset()
    assert policy.proprio is None and not policy.queue and policy.calls == 0


def test_environment_done_does_not_count_as_task_success():
    env, adapter = Env(), Adapter()
    original = env.step
    def step(action):
        obs, reward, _, info = original(action)
        return obs, reward, True, info
    env.step = step
    env.controller.use_delta = False
    outcome, _ = rollout(env, NativeChunkPolicy(adapter), env.obs(), 'move bowl', 10)
    assert outcome['n_steps'] == 1 and not outcome['success']
    assert outcome['termination'] == 'environment_done'


def test_task_success_stops_immediately():
    env, adapter = Env(), Adapter()
    env.controller.use_delta = False
    env.check_success = lambda: env.position[0] > .15
    outcome, _ = rollout(env, NativeChunkPolicy(adapter), env.obs(), 'move bowl', 10)
    assert outcome['success'] and outcome['n_steps'] == 2


def test_episode_selection_checks_reference_seeds_and_exact_task_names(tmp_path):
    data, keys, reference = tmp_path/'data.h5', tmp_path/'episodes.txt', tmp_path/'reference.json'
    with h5py.File(data, 'w') as h5:
        h5.create_group('ep_000033')['texts'] = json.dumps(['turn on the stove'])
        h5.create_group('ep_000010')['texts'] = json.dumps(['move bowl'])
    keys.write_text('ep_000033\nep_000010\n')
    tasks = [SimpleNamespace(language='move bowl'), SimpleNamespace(language='turn on the stove')]
    cases = load_cases(data, keys, tasks)
    assert [c['episode'] for c in cases] == ['ep_000010', 'ep_000033']
    assert cases[0]['episode_seed'] == episode_seed(42, 'ep_000010')
    # A stored seed takes precedence over reconstructing a seed from CLI defaults.
    cases[0]['episode_seed'] = 123
    reference.write_text(json.dumps({'episodes': cases}))
    assert load_cases(data, keys, tasks, reference=reference)[0]['episode_seed'] == 123
    assert len(load_cases(data, keys, tasks, limit=1)) == 1
    with pytest.raises(ValueError, match='exact benchmark task'):
        load_cases(data, keys, tasks[:1])
    keys.write_text('ep_000010\nep_000010\n')
    with pytest.raises(ValueError, match='unique'):
        load_cases(data, keys, tasks)


def test_per_task_counts_keep_all_selected_cases():
    summary = summarize([dict(task_name='a', success=True), dict(task_name='a', success=False),
                         dict(task_name='b', success=True)])
    assert summary['successes'] == 2 and summary['total'] == 3
    assert summary['per_task']['a'] == dict(successes=1, total=2, success_rate=.5)


def test_adapter_uses_explicit_proprio_without_changing_default_contract():
    class Policy(torch.nn.Module):
        def predict_action_chunk(self, batch):
            self.last_state = batch['observation.state'].clone()
            return torch.from_numpy(target().astype(np.float32)[None, None])
    adapter = XVLALiberoAdapter.__new__(XVLALiberoAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.device = torch.device('cpu')
    adapter.chunk_size = 1
    adapter.policy = Policy()
    adapter.domain_id = 3
    adapter._text_cache = {'move bowl': torch.tensor([[1]])}
    env = Env()
    obs = dict(env.obs(), agentview_image=np.zeros((4, 4, 3), dtype=np.uint8),
               robot0_eye_in_hand_image=np.zeros((4, 4, 3), dtype=np.uint8))
    override = target(.6, close=0).astype(np.float32)
    adapter.predict_sim_action_chunk(obs, task_text='move bowl', env=env, proprio=override)
    np.testing.assert_allclose(adapter.policy.last_state[0], override)
    adapter.predict_sim_action_chunk(obs, task_text='move bowl', env=env)
    np.testing.assert_allclose(adapter.policy.last_state[0], target(close=0))
    with pytest.raises(ValueError, match='finite 20-D'):
        adapter.predict_sim_action_chunk(obs, task_text='move bowl', env=env, proprio=np.zeros(8))
