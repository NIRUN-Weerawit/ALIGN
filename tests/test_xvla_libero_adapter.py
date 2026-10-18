"""XVLA-native observation/action contracts and shared-loop integration."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from eval.xvla_libero_adapter import (absolute_to_delta, native_observation,
                                      resolve_snapshot, rotation6d_matrix)


def controller():
    return SimpleNamespace(use_delta=True, control_dim=6, input_min=-np.ones(6),
        input_max=np.ones(6), output_min=-np.array([.05]*3+[.5]*3),
        output_max=np.array([.05]*3+[.5]*3))


def target(position=(0., 0., 0.), rotation=None, close=1.):
    rotation = np.eye(3) if rotation is None else rotation
    return np.r_[position, rotation[:, 0], rotation[:, 1], close, np.zeros(10)]


def test_native_camera_orientation_normalization_and_pose():
    image = np.arange(4*5*3, dtype=np.uint8).reshape(4, 5, 3)
    matrix = Rotation.from_euler('xyz', [.2, -.3, .4]).as_matrix()
    batch = native_observation(dict(robot0_eef_pos=np.array([1., 2., 3.]),
        agentview_image=image, robot0_eye_in_hand_image=image), matrix)
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    main = (batch['observation.images.image'][0] * std + mean) * 255
    wrist = (batch['observation.images.image2'][0] * std + mean) * 255
    np.testing.assert_allclose(main.permute(1, 2, 0), image[::-1, ::-1], atol=1e-4)
    np.testing.assert_allclose(wrist.permute(1, 2, 0), image, atol=1e-4)
    np.testing.assert_allclose(batch['observation.state'][0], target((1, 2, 3), matrix, close=0), atol=1e-6)
    assert 'observation.images.empty_camera_0' not in batch


def test_absolute_action_inverts_controller_scaling_and_gripper():
    current = Rotation.from_euler('xyz', [.3, -.1, .2]).as_matrix()
    world_delta = np.array([.04, -.02, .03])
    desired = Rotation.from_rotvec(world_delta).as_matrix() @ current
    position = np.array([.2, .1, .8])
    action = absolute_to_delta(target(position+[.01, -.02, .005], desired, close=1),
                               position, current, controller())
    np.testing.assert_allclose(action[:3], [.2, -.4, .1], atol=1e-6)
    np.testing.assert_allclose(action[3:6], world_delta/.5, atol=1e-6)
    assert action[6] == 0  # shared loop converts this once to simulator +1 (closed)
    opened = absolute_to_delta(target(position, current, close=0), position, current, controller())
    np.testing.assert_allclose(opened, [0, 0, 0, 0, 0, 0, 1], atol=1e-6)
    clipped = absolute_to_delta(target(position+[10, 0, 0]), position, current, controller())
    assert clipped[0] == 1


@pytest.mark.parametrize('value', [np.zeros(6), np.ones(6), np.full(6, np.nan)])
def test_bad_rotation_is_rejected(value):
    with pytest.raises(ValueError):
        rotation6d_matrix(value)


def test_hf_cache_directory_resolution_and_incomplete_download(tmp_path):
    root = tmp_path/'models--lerobot--xvla-libero'
    snapshot = root/'snapshots'/'revision'
    snapshot.mkdir(parents=True)
    (root/'refs').mkdir()
    (root/'refs/main').write_text('revision\n')
    (snapshot/'config.json').write_text('{}')
    with pytest.raises(FileNotFoundError):
        resolve_snapshot(root)
    (snapshot/'model.safetensors').write_bytes(b'weights')
    assert resolve_snapshot(root) == snapshot.resolve()
    assert resolve_snapshot(snapshot) == snapshot.resolve()
    (root/'refs/main').unlink()
    assert resolve_snapshot(root) == snapshot.resolve()


def test_shared_loop_converts_queued_absolute_goals_at_execution(monkeypatch):
    import eval.eval_libero_v4_trajectory as evaluator
    class Env:
        def __init__(self):
            self.position = np.zeros(3)
            self.actions = []
        def obs(self):
            return dict(robot0_eef_pos=self.position.copy(), robot0_eef_quat=np.array([0, 0, 0, 1]))
        def reset(self):
            self.position[:] = 0
            return self.obs()
        def step(self, action):
            self.actions.append(action.copy())
            self.position += action[:3] * .05
            return self.obs(), 0., False, {}
    class Adapter(torch.nn.Module):
        history_size=1
        use_memory_bank=False
        intention_encoder=None
        text_condition=None
        head_history_mode='mean'
        def __init__(self):
            super().__init__();self.calls=0;self.resets=0
        def reset_episode(self):self.resets+=1
        def predict_sim_action_chunk(self, obs, *, task_text, env):
            assert task_text == 'move bowl'
            self.calls+=1
            goal=target(obs['robot0_eef_pos']+[.01, 0, 0])
            return torch.tensor(np.tile(goal, (1, 3, 1)),dtype=torch.float32)
        def to_libero_action(self, action, obs, env):
            return absolute_to_delta(action,obs['robot0_eef_pos'],np.eye(3),controller())
    env, model = Env(), Adapter()
    monkeypatch.setattr(evaluator, 'get_sim_frame', lambda *a, **k: np.zeros((4, 4, 3), np.uint8))
    result=evaluator.run_model_in_sim(env,model,torch.device('cpu'),np.zeros((3,7),np.float32),
        chunk_size=3,max_steps=3,switch_at=0,action_horizon=2,
        use_camera=['image','wrist_image'],task_text='move bowl')
    assert model.calls == 2 and model.resets == 1
    np.testing.assert_allclose(np.array(env.actions)[:,0],[.2,0,.2],atol=1e-6)
    np.testing.assert_allclose(np.array(env.actions)[:,6],1.)
    assert result['stored_actions'].shape == (3,7)
    assert result['n_steps'] == 3


def test_headless_opencv_cleanup_is_supported(monkeypatch):
    import eval.eval_libero_v4_trajectory as evaluator
    class HeadlessCV:
        error = RuntimeError
        @staticmethod
        def destroyAllWindows():
            raise RuntimeError('No GUI backend')
    monkeypatch.setattr(evaluator, 'CV2_AVAILABLE', True)
    monkeypatch.setattr(evaluator, 'cv2', HeadlessCV())
    evaluator._live_view_close()
