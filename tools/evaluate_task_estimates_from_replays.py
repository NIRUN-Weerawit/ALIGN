"""Reconstruct task estimates from saved expert and policy evaluation traces.

Only frozen perception/context and a fixed label probe run. Saved actions drive
the simulator, so this does not generate new policy-success measurements.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import h5py
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torch.nn.attention import SDPBackend, sdpa_kernel

from data.gripper_state import executed_binary_gripper
from data.task_context_dataset import file_hash
from eval.eval_libero_v4_trajectory import (OffScreenRenderEnv,get_bddl_path,
    get_sim_eef_pose,get_sim_frame,seed_episode)
from eval.libero_protocol import reset_simulation
from eval.task_estimation import TaskLabelEstimator, TaskEstimationTrace, PeakTaskSnapshot, summarize_task_estimation
from models.align_model import VisionEncoder
from models.task_context_policy import TaskContextPolicyBranch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--task-estimator',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episode-subset',choices=['all','failed-expert-replay'],default='failed-expert-replay')
    parser.add_argument('--sources',nargs='+',choices=['expert','policy'],default=['expert','policy'])
    parser.add_argument('--gpu-memory-fraction',type=float,default=.24)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    reference_hash = file_hash(args.reference)
    reference = json.loads(args.reference.read_text())
    checkpoint = torch.load(args.policy,map_location='cpu',weights_only=True,mmap=True)
    cfg = checkpoint['config']
    if file_hash(args.policy) != reference['evaluation_protocol']['checkpoint_sha256']:
        raise ValueError('Policy and reference evaluation differ')
    dev = set(cfg['task_context_split']['development_episodes'])
    if any(row['episode'] not in dev for row in reference['episodes']):
        raise ValueError('Use a development-only reference; training/confirmation excluded')
    selected = [row for row in reference['episodes'] if args.episode_subset == 'all' or not row['replay_check']['valid']]
    if not selected:
        raise ValueError('No episodes selected')
    branch = TaskContextPolicyBranch(config=cfg['task_context_config']).requires_grad_(False).eval()
    prefix = 'task_context.'
    branch.load_state_dict({k[len(prefix):]:v for k,v in checkpoint['model_state_dict'].items() if k.startswith(prefix)},strict=True)
    estimator = TaskLabelEstimator(args.task_estimator)
    estimator.validate_branch(branch)
    branch = branch.cuda()
    encoder = branch.observer.encoder
    args.output.mkdir(parents=True)
    protocol = dict(schema='align-reconstructed-task-estimates-v1',reference=str(args.reference),
        reference_sha256=reference_hash,policy_sha256=reference['evaluation_protocol']['checkpoint_sha256'],
        estimator_sha256=file_hash(args.task_estimator),source_sha256=file_hash(__file__),
        episode_subset=args.episode_subset,episodes=[r['episode'] for r in selected],sources=args.sources,
        confirmation_opened=False,neural_weights_updated=False,action_policy_rerun=False,
        pose_reproduction_tolerance_m=1e-4,probabilities_calibrated=False,
        reset_sequence=dict(expert='Seed, setup reset, replay reset, flat-state restore',
                            policy='Seed, policy reset, flat-state restore; matching post-replay reseed'),
        note='Reconstructed frozen task estimates under saved actions; not a new policy performance evaluation')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    vision = VisionEncoder(num_cameras=1,use_patch_tokens=True,fusion_type='linear').requires_grad_(False).eval().cuda()
    results = []
    start = time.monotonic()
    with threadpool_limits(limits=2),h5py.File(args.data,'r') as data:
        for row in selected:
            key = row['episode']
            n = len(data[key]['actions'])
            actual_task = json.loads(data[key]['texts'][()])[0]
            if actual_task != row['task_name']:
                raise ValueError(f'{key}: dataset/reference task names differ')
            for source in args.sources:
                trace_path = args.reference.parent/f'{key}_{"replay" if source == "expert" else "rollout"}.npz'
                with np.load(trace_path,allow_pickle=False) as saved:
                    actions,expected,initial = [saved[field].copy() for field in ['executed_actions','sim_positions','initial_sim_state']]
                if len(actions) != len(expected):
                    raise ValueError('Saved action/pose coverage differs')
                seed_episode(row['episode_seed'])
                env = OffScreenRenderEnv(bddl_file_name=get_bddl_path('libero_goal',actual_task),
                    use_camera_obs=True,camera_names=['agentview','robot0_eye_in_hand'],camera_widths=256,
                    camera_heights=256,reward_shaping=False,control_freq=20,initialization_noise=None)
                try:
                    env.seed(row['episode_seed'])
                    if source == 'expert':
                        env.reset()
                    obs = reset_simulation(env,initial)
                    stream = encoder.new_stream(1)
                    switch = len(actions) if source == 'expert' else row['switch_step']
                    trace = TaskEstimationTrace(estimator,actual_task,switch,n)
                    previous_gripper = float(data[key].attrs.get('initial_gripper',0.))
                    snapshot = None
                    peak = (PeakTaskSnapshot(estimator) if source == 'policy' and
                            (row.get('task_context_selection') or {}).get('strategy')=='highest_decoder_score' else None)
                    max_error = 0.
                    for t in range(len(actions)):
                        pose = get_sim_eef_pose(obs,'libero')
                        frames = np.stack([get_sim_frame(env,c,flip_vertical=True,flip_horizontal=True)
                                           for c in ['image','wrist_image']])
                        state = torch.as_tensor(np.r_[pose,previous_gripper].astype(np.float32),device='cuda')[None]
                        with sdpa_kernel(backends=[SDPBackend.MATH]):
                            raw = vision(torch.from_numpy(frames).cuda())
                            summary = branch.summaries(raw[:,:-1].reshape(1,1,512,768),
                                raw[:,-1].reshape(1,1,2,768),state[:,None],
                                torch.tensor([[t/branch.fps]],device='cuda'))[:,0]
                            out = encoder.step(summary,stream)
                        if source == 'policy' and peak is None and row['task_context_mode'] in ('freeze','handover') and t == switch:
                            snapshot = out['tokens'].clone()
                        selection = None
                        if peak is not None:
                            snapshot,selection = peak.observe(t,out['tokens'])
                        trace.observe(t,out['tokens'],out['tokens'] if snapshot is None else snapshot,
                            execution_token=out['execution_token'],snapshot_selection=selection)
                        action = actions[t].copy()
                        if source == 'expert':
                            previous_gripper,action[6] = executed_binary_gripper(action[6])
                        else:
                            if action[6] not in (-1.,1.):
                                raise ValueError('Saved policy gripper is not in simulator polarity')
                            previous_gripper = (1.-action[6])/2.
                        obs,_,_,_ = env.step(action)
                        error = float(np.linalg.norm(get_sim_eef_pose(obs,'libero')[:3]-expected[t,:3]))
                        max_error = max(max_error,error)
                        if error > protocol['pose_reproduction_tolerance_m']:
                            raise ValueError(f'{key}/{source}, step {t}: trajectory mismatch {error}m')
                    trace.save(args.output/f'{key}_{source}_task_estimates')
                    result = dict(episode=key,source=source,task_name=actual_task,
                        replay_check=row['replay_check'],success=row['success'] if source == 'policy' else None,
                        saved_trace_sha256=file_hash(trace_path),max_pose_reproduction_error_m=max_error,
                        task_estimation=trace.summary())
                    results.append(result)
                finally:
                    env.close()
                checkpoints = {k:(v['inferred']['predicted_task'],v['inferred']['correct']) if v else None
                               for k,v in result['task_estimation']['checkpoints'].items()}
                print('EPISODE',key,source,'max_pose_error',max_error,'elapsed',round(time.monotonic()-start,1),
                      'predictions',checkpoints,flush=True)
                (args.output/'progress.json').write_text(json.dumps(dict(protocol=protocol,episodes=results),indent=2)+'\n')
    if file_hash(args.reference) != reference_hash:
        raise ValueError('Reference report changed during reconstruction')
    aggregates = {source:summarize_task_estimation([r for r in results if r['source'] == source])
                  for source in args.sources}
    (args.output/'result.json').write_text(json.dumps(dict(protocol=protocol,episodes=results,
        task_estimation=aggregates,elapsed_seconds=time.monotonic()-start),indent=2)+'\n')
    print('DONE',args.output,flush=True)


if __name__ == '__main__':
    main()
