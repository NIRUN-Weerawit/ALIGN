"""Matched-scene closed-loop pilot of a frozen context policy, with no training.

Select development episodes before outcomes. Verify scene/render identity and
identical human prefixes. Historical results are provenance, not paired baselines.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import h5py
import numpy as np
import torch
from data.task_context_dataset import file_hash
from eval.context_interventions import ARMS, StreamingIntervention, select_donors
from eval.eval_intention import load_intention_model
from eval.eval_libero_v4_trajectory import OffScreenRenderEnv,get_bddl_path,seed_episode,run_model_in_sim


class AuditedEnvironment:
    def __init__(self,env):self._wrapped=env;self.initial_identity=None;self.step_audit=[]
    def __getattr__(self,name):return getattr(self._wrapped,name)
    def synchronized_observations(self):
        # Explicitly refresh both cameras at the current state for this diagnostic.
        # Repeat controls still need to verify image/trajectory reproducibility.
        self._wrapped.sim.forward()
        self._wrapped._update_observables(force=True)
        return self._wrapped.env._get_observations()
    def step(self,action):
        result=self._wrapped.step(action)
        result=(self.synchronized_observations(),*result[1:])
        self.step_audit.append(dict(state=hashlib.sha256(np.asarray(self._wrapped.sim.get_state().flatten()).tobytes()).hexdigest(),
            cameras=hashlib.sha256(b''.join(np.asarray(result[0][k]).tobytes() for k in sorted(result[0]) if k.endswith('_image'))).hexdigest()))
        return result
    def set_init_state(self,state):
        obs=self._wrapped.set_init_state(state)
        obs=self.synchronized_observations()
        digest=hashlib.sha256()
        # Flat state alone omits randomized fixture geometry. Include the actual
        # model geometry and both initial rendered camera observations.
        for name in ('body_pos','body_quat','geom_pos','geom_quat','geom_size'):
            digest.update(np.asarray(getattr(self._wrapped.sim.model,name)).tobytes())
        digest.update(np.asarray(self._wrapped.sim.get_state().flatten()).tobytes())
        cameras=sorted(k for k in obs if k.endswith('_image'))
        if len(cameras)<2:raise ValueError('Need both camera observations for initial-scene audit')
        for camera in cameras:digest.update(np.asarray(obs[camera]).tobytes())
        self.initial_identity=digest.hexdigest()
        return obs


def summarize(rows,arms):
    baseline={r['episode']:r for r in rows if r['arm']=='baseline'}
    result={}
    for arm in arms:
        selected=[r for r in rows if r['arm']==arm]
        eligible=[r for r in selected if r['policy_eligible']]
        paired=[r for r in eligible if baseline.get(r['episode'],{}).get('policy_eligible')]
        result[arm]=dict(episodes=len(selected),eligible=len(eligible),successes=sum(r['success'] for r in eligible),
            success_rate=sum(r['success'] for r in eligible)/len(eligible) if eligible else None,
            paired_improvements=[r['episode'] for r in paired if r['success'] and not baseline[r['episode']]['success']],
            paired_regressions=[r['episode'] for r in paired if not r['success'] and baseline[r['episode']]['success']])
    return result


def render(report):
    if report.get('status')=='invalid_repeat_control':
        return ('# Task-context rollout pilot: invalid repeat control\n\n'
                'The identical-baseline repeat diverged. Intervention completion comparisons were aborted.\n\n'
                'No closed-loop benefit or regression can be attributed to task tokens from this run.\n\n'
                'Repeat diagnostics:\n\n```json\n'+json.dumps(report['repeat_audit'],indent=2)+'\n```\n')
    lines=['# Matched task-context rollout pilot','',
        ('Status: complete.' if report.get('complete') else 'Status: incomplete; results are provisional.'), '',
        f"{len(report['protocol']['episodes'])} development episodes; selection: {report['protocol']['selection']}. "
        'One simulation/noise seed per episode. All selected cases included regardless of historical expert replay success.',
        'Each arm uses the same verified initial scene and expert prefix. This is a new matched experiment, not a replay of historical success totals.','',
        '| Intervention | Policy completion | Improved vs baseline | Regressed vs baseline |',
        '|---|---:|---:|---:|']
    for arm,r in report['aggregate'].items():
        score=f"{r['successes']}/{r['eligible']} ({r['success_rate']:.1%})" if r['eligible'] else 'N/A'
        lines.append(f"| {arm} | {score} | {len(r['paired_improvements'])} | {len(r['paired_regressions'])} |")
    lines+=['','| Episode | Task | '+ ' | '.join(report['protocol']['arms'])+' |',
            '|---|---|'+ '---:|'*len(report['protocol']['arms'])]
    for ep in report['protocol']['episodes']:
        matches=[r for r in report['episodes'] if r['episode']==ep]
        if not matches:continue
        byarm={r['arm']:r for r in matches}
        lines.append('| '+ep+' | '+matches[0]['task_name']+' | '+' | '.join(
            ('Success' if byarm[a]['success'] else 'Failure') if a in byarm else 'Pending' for a in report['protocol']['arms'])+' |')
    lines+=['','Interpret as a small mechanistic pilot, not a reliable ranking of architectures. '
            'Different-task donor tokens come from training episodes and are deliberately incorrect context. '
            'Task-mean tokens are also training-only. No weights changed; no confirmation episodes evaluated.']
    return '\n'.join(lines)+'\n'


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--token-probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes-per-task',type=int,default=1,help='0 selects all saved development episodes')
    p.add_argument('--arms',nargs='+',choices=ARMS,default=['baseline','freeze_task','task_mean','task_swap','no_task','no_execution'])
    p.add_argument('--seed',type=int,default=20261021)
    p.add_argument('--gpu-memory-fraction',type=float,default=.24)
    args=p.parse_args()
    if OffScreenRenderEnv is None:
        raise ImportError('LIBERO is unavailable; add its checkout to PYTHONPATH before running this pilot')
    if args.output.exists():raise FileExistsError('Use a new output directory')
    if args.episodes_per_task<0 or args.arms[0]!='baseline' or len(set(args.arms))!=len(args.arms):
        raise ValueError('Require nonnegative subset size and unique arms with baseline first')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.use_deterministic_algorithms(True)
    model,cfg=load_intention_model(str(args.checkpoint),torch.device('cuda'))
    model.requires_grad_(False).eval()
    if model.task_context is None or model.use_memory_bank or model.text_condition is not None or model.history_size!=1:
        raise ValueError('Requires context_no_memory without text, history_size=1')
    reference=json.loads(args.reference.read_text());prior=json.loads((args.token_probe/'protocol.json').read_text())
    checkpoint_hash=file_hash(args.checkpoint)
    if reference['evaluation_protocol']['checkpoint_sha256']!=checkpoint_hash or prior['policy_sha256']!=checkpoint_hash:
        raise ValueError('Policy/reference/probe identity mismatch')
    dev=set(cfg['task_context_split']['development_episodes'])
    refs={r['episode']:r for r in reference['episodes']}
    if set(refs)!=dev or len(refs)!=len(reference['episodes']):raise ValueError('Reference must contain exactly saved development')
    train=dict(np.load(args.token_probe/'train_features.npz',allow_pickle=False))
    if set(train['keys'])!=set(cfg['task_context_split']['train_episodes']) or len(set(train['keys']))!=len(train['keys']):
        raise ValueError('Donors must be training-only')
    names=json.loads((args.token_probe/'results.json').read_text())['task_names']
    grouped={name:sorted(k for k in dev if refs[k]['task_name']==name) for name in sorted({r['task_name'] for r in refs.values()})}
    selected=sorted(k for keys in grouped.values() for k in (keys[:args.episodes_per_task] if args.episodes_per_task else keys))
    encoder=model.task_context.observer.encoder
    native=torch.from_numpy(train['tokens'][:,3]).cuda()*encoder.latent_std+encoder.latent_mean
    mean=native.mean(0,keepdim=True)
    protocol=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=checkpoint_hash,
        reference=str(args.reference),reference_sha256=file_hash(args.reference),
        train_features_sha256=file_hash(args.token_probe/'train_features.npz'),
        episodes=selected,arms=args.arms,selection='first keys per task, lexicographic; independent of outcomes',
        episodes_per_task=args.episodes_per_task,seed=args.seed,expert_prefix_fraction=.4,max_steps=300,
        action_horizon=1,noise_std=1e-5,weights_updated=False,confirmation_opened=False,
        repeat_control='Full duplicate baseline on first episode; identical commands and poses required',
        pairing='Same environment/render context per episode, soft resets; initial geometry/state/camera hash; exact expert prefix; per-step diffusion seed',
        source_sha256={name:file_hash(ROOT/name) for name in ['tools/evaluate_context_interventions.py','eval/context_interventions.py','eval/eval_libero_v4_trajectory.py']})
    args.output.mkdir(parents=True)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[];started=time.monotonic();repeat_audit=None
    with h5py.File(args.data,'r') as data:
        for ei,key in enumerate(selected):
            ref=refs[key];task=ref['task_name'];ep=data[key]
            if json.loads(ep['texts'][()])[0]!=task:raise ValueError('Dataset task mismatch')
            actions=ep['actions'][:];poses=ep['poses'][:]
            switch=int(len(actions)*.4)
            if not 0<switch<300:raise ValueError('Invalid handover step')
            with np.load(args.reference.parent/f'{key}_rollout.npz',allow_pickle=False) as saved:
                initial=saved['initial_sim_state'].copy()
            same,other=select_donors(train['keys'],train['groups'],names.index(task),args.seed+ei)
            replacements=dict(task_mean=mean,task_swap=native[other:other+1],same_task_swap=native[same:same+1])
            baseline=None;identity=None
            arms=list(args.arms)
            if ei==0 and 'repeat_control' not in arms:arms.insert(1,'repeat_control')
            seed_episode(ref['episode_seed'])
            env=AuditedEnvironment(OffScreenRenderEnv(bddl_file_name=get_bddl_path('libero_goal',task),
                use_camera_obs=True,camera_names=['agentview','robot0_eye_in_hand'],camera_widths=256,camera_heights=256,
                reward_shaping=False,control_freq=20,initialization_noise=None,hard_reset=False))
            try:
                for arm in arms:
                    seed_episode(ref['episode_seed'])
                    env.seed(ref['episode_seed'])
                    env.step_audit=[]
                    with StreamingIntervention(model.intention_head,arm,switch,args.seed+ei*10000,replacements.get(arm)) as intervention:
                        out=run_model_in_sim(env,model,torch.device('cuda'),actions,poses,
                            chunk_size=model.chunk_size,max_steps=300,switch_at=.4,use_camera=['image','wrist_image'],
                            flip_vertical=True,flip_horizontal=True,noise_std=1e-5,action_horizon=1,
                            initial_gripper=float(ep.attrs.get('initial_gripper',0.)),task_context_mode='dynamic',
                            initial_sim_state=initial,record_frames=False)
                    if intervention.step!=out['n_steps']:raise ValueError('Expected one head call per observation')
                    if arm=='baseline':
                        baseline=out;identity=env.initial_identity
                        baseline_head_audit=intervention.audit
                        baseline_sim_audit=env.step_audit
                    if identity!=env.initial_identity:raise ValueError('Initial scene differs between arms')
                    if not np.array_equal(out['executed_actions'][:switch],baseline['executed_actions'][:switch]):
                        raise ValueError('Expert prefixes differ')
                    if not np.array_equal(out['sim_positions'][:switch],baseline['sim_positions'][:switch]):
                        raise ValueError('Expert-prefix trajectories differ')
                    if arm=='repeat_control':
                        repeat_audit=dict(episode=key,max_action_delta=float(np.abs(out['executed_actions']-baseline['executed_actions']).max()),
                            max_pose_delta=float(np.abs(out['sim_positions']-baseline['sim_positions']).max()))
                        repeat_audit['first_action_difference'] = next((i for i,(a,b) in enumerate(zip(out['executed_actions'],baseline['executed_actions'])) if not np.array_equal(a,b)),None)
                        repeat_audit['first_prediction_difference'] = next((i for i,(a,b) in enumerate(zip(out['stored_actions'],baseline['stored_actions'])) if not np.array_equal(a,b)),None)
                        repeat_audit['first_head_differences'] = {field:next((i for i,(a,b) in enumerate(zip(intervention.audit,baseline_head_audit)) if a[field]!=b[field]),None)
                            for field in ('visual','state','payload','condition')}
                        repeat_audit['first_sim_differences'] = {field:next((i for i,(a,b) in enumerate(zip(env.step_audit,baseline_sim_audit)) if a[field]!=b[field]),None)
                            for field in ('state','cameras')}
                        if repeat_audit['max_action_delta'] or repeat_audit['max_pose_delta']:
                            np.savez_compressed(args.output/'failed_repeat.npz',executed_actions=out['executed_actions'],predicted_actions=out['stored_actions'],sim_positions=out['sim_positions'])
                            (args.output/'repeat_failure.json').write_text(json.dumps(repeat_audit,indent=2)+'\n')
                            failed_report=dict(protocol=protocol,complete=False,status='invalid_repeat_control',
                                episodes=rows,aggregate={},repeat_audit=repeat_audit)
                            (args.output/'result.json').write_text(json.dumps(failed_report,indent=2)+'\n')
                            (args.output/'summary.md').write_text(render(failed_report))
                            raise ValueError(f'Repeat control failed: {repeat_audit}')
                    if env.step_audit[:switch]!=baseline_sim_audit[:switch]:
                        raise ValueError('Expert-prefix simulator states or camera observations differ')
                    np.savez_compressed(args.output/f'{key}_{arm}.npz',executed_actions=out['executed_actions'],
                        predicted_actions=out['stored_actions'],sim_positions=out['sim_positions'],initial_sim_state=initial)
                    row=dict(episode=key,task_name=task,arm=arm,initial_scene_sha256=identity,
                        historical_expert_replay=ref['replay_check'],
                        same_task_donor=str(train['keys'][same]),different_task_donor=str(train['keys'][other]),
                        **{k:out[k] for k in ['success','episode_success','policy_eligible','success_before_handover','first_success_step','switch_step','n_steps']})
                    rows.append(row)
                    print(f'ROLLOUT {ei+1}/{len(selected)} {key} {arm} success={out["success"]} elapsed={time.monotonic()-started:.1f}s',flush=True)
                    report=dict(protocol=protocol,complete=False,episodes=rows,aggregate=summarize(rows,args.arms),repeat_audit=repeat_audit)
                    (args.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
                    (args.output/'summary.md').write_text(render(report))
            finally:
                env.close()
    report.update(complete=True,elapsed_seconds=time.monotonic()-started)
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'summary.md').write_text(render(report))
    print(render(report),flush=True)


if __name__=='__main__':main()
