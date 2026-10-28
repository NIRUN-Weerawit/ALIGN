"""Probe frozen task tokens after replaying development expert prefixes in LIBERO.

Uses captured simulator states and expert actions from a completed evaluation.
No autonomous actions, policy updates, or reserved episodes are used. Diagnostic
classifiers fit only the training features produced by probe_task_context_tokens.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torch.nn.attention import SDPBackend, sdpa_kernel

from data.gripper_state import executed_binary_gripper
from data.task_context_dataset import file_hash
from eval.eval_libero_v4_trajectory import (OffScreenRenderEnv, get_bddl_path,
    get_sim_eef_pose, get_sim_frame, seed_episode)
from eval.libero_protocol import reset_simulation
from models.align_model import VisionEncoder
from models.task_context_policy import TaskContextPolicyBranch, rotation_columns
from tools.probe_task_context_tokens import feature_views, geometric_separation, linear_probe


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True, help='Completed development result.json with expert-replay NPZs')
    parser.add_argument('--probe', type=Path, required=True, help='Completed cached-token probe directory')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-memory-fraction', type=float, default=.24)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    reference_hash = file_hash(args.reference)
    reference = json.loads(args.reference.read_text())
    cached_result = json.loads((args.probe / 'results.json').read_text())
    policy_hash = file_hash(args.policy)
    if cached_result['protocol']['policy_sha256'] != policy_hash or reference['evaluation_protocol']['checkpoint_sha256'] != policy_hash:
        raise ValueError('Policy identity differs from the input evaluations')
    train = dict(np.load(args.probe / 'train_features.npz', allow_pickle=False))
    dev = dict(np.load(args.probe / 'development_features.npz', allow_pickle=False))
    rows = {r['episode']:r for r in reference['episodes']}
    if set(rows) != set(dev['keys']) or len(reference['episodes']) != len(dev['keys']):
        raise ValueError('Reference must contain exactly the protected development episodes')
    if set(train['keys']) & set(rows):
        raise ValueError('Training/development overlap')
    if not reference['evaluation_complete']:
        raise ValueError('Incomplete reference evaluation')
    checkpoint = torch.load(args.policy, map_location='cpu', weights_only=True, mmap=True)
    cfg = checkpoint['config']
    if set(cfg['task_context_split']['development_episodes']) != set(rows):
        raise ValueError('Policy development partition differs')
    branch = TaskContextPolicyBranch(config=cfg['task_context_config']).requires_grad_(False).eval()
    prefix = 'task_context.'
    branch.load_state_dict({k[len(prefix):]:v for k,v in checkpoint['model_state_dict'].items() if k.startswith(prefix)}, strict=True)
    branch = branch.cuda()
    encoder = branch.observer.encoder
    args.output.mkdir(parents=True)
    protocol = dict(schema='align-task-token-simulator-prefix-v1', policy_sha256=policy_hash,
        reference=str(args.reference), reference_sha256=reference_hash,
        cached_probe=str(args.probe), cached_probe_sha256=file_hash(args.probe / 'results.json'),
        train_features_sha256=file_hash(args.probe / 'train_features.npz'),
        development_features_sha256=file_hash(args.probe / 'development_features.npz'),
        source_sha256=file_hash(__file__), prefix_fraction=.4,
        query='Observe frames 0 through floor(demonstration_length * 0.4), inclusive',
        inputs='Live simulator cameras and EEF pose; previous binary command; elapsed seconds',
        scene='Captured seeded simulator reset from reference, not claimed to be the original demonstration scene',
        replay='Saved expert actions, converted from dataset to simulator gripper polarity; no added noise',
        pose_reproduction_tolerance_m=1e-4,
        reset_sequence='Seed, setup reset, then expert replay reset and saved flat-state restore; matches reference expert run',
        classifiers='Train-only scaler and balanced logistic regression C=0.1; same probe as cached analysis',
        confirmation_opened=False, replay_filters_episodes=False, encoder_updated=False,
        runtime={name:importlib.metadata.version(name) for name in ['torch','mujoco','robosuite','numpy']})
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    vision = VisionEncoder(num_cameras=1, use_patch_tokens=True, fusion_type='linear').requires_grad_(False).eval().cuda()
    values = {key:[] for key in ['keys','groups','tokens','execution','summary','states','times','mode','anchors','lengths']}
    audits = []
    start = time.monotonic()
    with h5py.File(args.data, 'r') as data:
        for index, key in enumerate(dev['keys']):
            row = rows[str(key)]
            n = int(dev['lengths'][index])
            boundary = int(n*.4)
            if n != row['reference_steps'] or int(dev['anchors'][index,3]) != boundary:
                raise ValueError(f'{key}: inconsistent demonstration length/anchor')
            trace = args.reference.parent / f'{key}_replay.npz'
            with np.load(trace, allow_pickle=False) as f:
                actions, expected, initial = (f[name].copy() for name in ['executed_actions','sim_positions','initial_sim_state'])
            bddl = get_bddl_path('libero_goal', row['task_name'])
            seed_episode(row['episode_seed'])
            env = OffScreenRenderEnv(bddl_file_name=bddl, use_camera_obs=True,
                camera_names=['agentview','robot0_eye_in_hand'], camera_widths=256,
                camera_heights=256, reward_shaping=False, control_freq=20, initialization_noise=None)
            try:
                env.seed(row['episode_seed'])
                # The reference evaluator first resets to capture its initial
                # state, then resets again for expert replay. LIBERO samples
                # fixture model.body_pos outside the flat qpos/qvel snapshot,
                # so matching this reset sequence is necessary at contact.
                env.reset()
                obs = reset_simulation(env, initial)
                previous_gripper = float(data[str(key)].attrs.get('initial_gripper',0.))
                stream = encoder.new_stream(1)
                max_error = 0.
                summaries = []
                for t in range(boundary+1):
                    pose = get_sim_eef_pose(obs, 'libero')
                    if t:
                        error = float(np.linalg.norm(pose[:3]-expected[t-1,:3]))
                        max_error = max(max_error,error)
                        if error > protocol['pose_reproduction_tolerance_m']:
                            raise ValueError(f'{key}, frame {t}: saved expert replay differs by {error}m')
                    frames = np.stack([get_sim_frame(env, camera, flip_vertical=True, flip_horizontal=True)
                                       for camera in ['image','wrist_image']])
                    state = torch.as_tensor(np.r_[pose,previous_gripper].astype(np.float32), device='cuda')[None]
                    with sdpa_kernel(backends=[SDPBackend.MATH]):
                        raw = vision(torch.from_numpy(frames).cuda())
                        patches = raw[:,:-1].reshape(1,512,768)
                        cls = raw[:,-1].reshape(1,2,768)
                        summary = branch.summaries(patches[:,None], cls[:,None], state[:,None],
                            torch.tensor([[t/branch.fps]],device='cuda'))[:,0]
                        out = encoder.step(summary,stream)
                    if index == 0:
                        summaries.append(summary)
                    if t < boundary:
                        action = actions[t].copy()
                        previous_gripper,action[6] = executed_binary_gripper(action[6])
                        obs,_,_,_ = env.step(action)
                tokens = ((out['tokens']-encoder.latent_mean)/encoder.latent_std)[0]
                geometry = torch.cat([state[:,:3],rotation_columns(state[:,3:6]),state[:,6:7]],-1)
                geometry = (geometry-branch.observer.state_mean)/branch.observer.state_std
                for field,value in dict(keys=str(key),groups=dev['groups'][index],tokens=tokens.cpu().numpy(),
                        execution=out['execution_token'][0,0].cpu().numpy(),summary=summary[0].cpu().numpy(),
                        states=geometry[0].cpu().numpy(),times=t/branch.fps,mode=int(out['mode'][0]),
                        anchors=boundary,lengths=n).items():
                    values[field].append(value)
                audit = dict(episode=str(key),frames_observed=boundary+1,
                    replay_trace_sha256=file_hash(trace),bddl_sha256=file_hash(bddl),
                    max_pose_reproduction_error_m=max_error,reference_replay_valid=row['replay_check']['valid'])
                if index == 0:
                    # Verify CUDA streaming and causal sequence readouts agree.
                    sequence = encoder(torch.stack(summaries,1),torch.tensor([boundary+1],device='cuda'),
                        torch.tensor([[boundary]],device='cuda'))['tokens'][0,0]
                    difference = float((sequence-out['tokens'][0]).abs().max())
                    audit['stream_vs_sequence_max_absolute_error'] = difference
                    if difference > 1e-3:
                        raise ValueError('Streaming/sequence token disagreement')
                audits.append(audit)
            finally:
                env.close()
            print(f'SIMULATOR {index+1}/{len(dev["keys"])} {key} max_pose_error={max_error:.2g}m '
                  f'elapsed={time.monotonic()-start:.1f}s',flush=True)
    if file_hash(args.reference) != reference_hash:
        raise ValueError('Reference result changed during the diagnostic')
    arrays = {key:np.asarray(value) for key,value in values.items()}
    np.savez_compressed(args.output / 'handover_features.npz',**arrays)
    # Reuse train-side coordinates/classifier without adapting to simulator data.
    simulator = {key:(value[:,None] if key not in ['keys','groups','lengths'] else value)
                 for key,value in arrays.items()}
    simulator['cls'] = np.zeros((len(arrays['keys']),1,1536),dtype=np.float32)
    train_views,sim_views = feature_views(train,3),feature_views(simulator,0)
    result = dict(protocol=protocol,episodes=len(arrays['keys']),task_names=cached_result['task_names'],
        replay_audit=audits,linear_probes={},encoder_updated=False)
    with threadpool_limits(limits=2):
        for view,x in train_views.items():
            if view == 'current_cls':
                continue  # This diagnostic does not reconstruct cache-specific CLS scaling.
            result['linear_probes'][view],prediction = linear_probe(x,train['groups'],sim_views[view],arrays['groups'])
            result['linear_probes'][view]['predicted_group'] = prediction.tolist()
        aligned_train = {key:(value[:,3:4] if key not in ['keys','groups','lengths'] else value)
                         for key,value in train.items()}
        result['geometry'],_,_,_ = geometric_separation(aligned_train,simulator,0)
    result['elapsed_seconds'] = time.monotonic()-start
    result['limitations'] = ['Known development tasks and captured simulator resets; no new-task or scene-disjoint claim.',
        'Expert-controlled prefix only; autonomous token drift and behavioral use are not measured.',
        'No action policy is rerun; no claim of reproducing the noisy 40%-handover policy evaluation.']
    (args.output / 'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PROBES',{k:v['development']['balanced_accuracy'] for k,v in result['linear_probes'].items()},flush=True)
    print('DONE',args.output,flush=True)


if __name__ == '__main__':
    main()
