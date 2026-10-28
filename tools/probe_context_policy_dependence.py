"""Matched-noise action interventions on all protected development episodes.

Uses cached expert observations, never autonomous rollout success. Training-only
native task tokens supply mean/swap controls. No weights or labels enter inference
except explicitly labeled diagnostic donor selection.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from data.align_dataset import ALIGNDataset
from data.task_context_dataset import file_hash, json_hash
from eval.eval_intention import load_intention_model
from eval.context_interventions import ARMS, intervene, select_donors

FRACTIONS = (.4, .5, .7, .9)
QUERY_INDICES = (3, 4, 5, 6)
GROUPS = {'position': slice(0,3), 'rotation': slice(3,6), 'gripper': slice(6,7)}


def aggregate(rows):
    result = {}
    for arm in ARMS:
        subset = [r for r in rows if r['arm'] == arm]
        if not subset:
            continue
        fields = [k for k,v in subset[0].items() if isinstance(v,float)]
        result[arm] = {k:float(np.mean([r[k] for r in subset])) for k in fields}
        result[arm]['observations'] = len(subset)
        for name in GROUPS:
            result[arm][name+'_delta_rms'] = result[arm][name+'_delta_mse']**.5
        paired = {}
        for ep in sorted({r['episode'] for r in subset}):
            values = [r for r in subset if r['episode']==ep]
            paired[ep] = np.mean([r['action_mse']-r['baseline_action_mse'] for r in values])
        differences = np.asarray(list(paired.values()))
        rng = np.random.default_rng(20261021)
        bootstrap = rng.choice(differences,(2000,len(differences)),replace=True).mean(1)
        result[arm]['paired_action_mse_change'] = float(differences.mean())
        result[arm]['episode_bootstrap_95pct_interval'] = np.quantile(bootstrap,[.025,.975]).tolist()
    return result


def render(report):
    lines = ['# Task-context action dependence', '',
        f"{len(report['protocol']['episodes'])} development episodes; 40%, 50%, 70%, 90% expert prefixes; "
        f"{report['protocol']['noise_draws']} matched diffusion draws per anchor.",
        'No policy weights changed; confirmation excluded. Expert observations are held fixed across interventions.',
        'Positive paired MSE change means worse imitation than the normal task/execution condition.', '',
        '| Intervention | Position action change RMS | Rotation action change RMS | Gripper flips | Action MSE | First-action MSE | Paired MSE change [95% episode bootstrap] |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm,r in report['aggregate'].items():
        lo,hi = r['episode_bootstrap_95pct_interval']
        lines.append(f"| {arm} | {r['position_delta_rms']:.6f} | {r['rotation_delta_rms']:.6f} | "
                     f"{r['gripper_flips']:.1%} | {r['action_mse']:.6f} | {r['first_action_mse']:.6f} | {r['paired_action_mse_change']:+.6f} [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['', '## Imitation error changes by task', '',
        '| Task | Remove task context | Different-task swap | Remove execution context |',
        '|---|---:|---:|---:|']
    for task,values in report.get('per_task',{}).items():
        lines.append('| '+task+' | '+' | '.join(f"{values[arm]['paired_action_mse_change']:+.6f}"
            for arm in ('no_task','task_swap','no_execution'))+' |')
    lines += ['', 'Task swaps use a different training task; same-task swaps control for episode-specific differences. '
              'Donors are fixed per development episode and matched by observation fraction. The mean is computed from training episodes only.',
              'Freeze uses the current episode’s 40% tokens; the execution token continues updating. '
              'No-task/no-execution remove the corresponding projected residual, including its bias; they do not merely zero raw tokens.',
              'Action differences establish sensitivity, not correct semantic use. Cached expert-state imitation error does not establish closed-loop benefit. '
              'The bootstrap resamples episodes and is descriptive for this selected checkpoint. '
              'The visual-zero arm is an out-of-distribution positive control.']
    return '\n'.join(lines)+'\n'


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--prefix-cache',type=Path,required=True)
    p.add_argument('--token-probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--noise-draws',type=int,default=3)
    p.add_argument('--seed',type=int,default=20261021)
    p.add_argument('--gpu-memory-fraction',type=float,default=.24)
    args = p.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output directory')
    if args.noise_draws < 1:raise ValueError('noise-draws must be positive')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.manual_seed(args.seed)
    model,cfg = load_intention_model(str(args.checkpoint),torch.device('cuda'))
    model.requires_grad_(False).eval()
    if model.task_context is None or model.use_memory_bank or model.text_condition is not None or model.history_size != 1:
        raise ValueError('This controlled probe requires context_no_memory, no text, history_size=1')
    head = model.intention_head
    if not hasattr(head,'context_task'):raise ValueError('Expected context diffusion head')
    split = cfg['task_context_split']
    manifest = json.loads((args.prefix_cache/'manifest.json').read_text())
    prior = json.loads((args.token_probe/'protocol.json').read_text())
    policy_hash = file_hash(args.checkpoint)
    if json_hash({k:v for k,v in manifest.items() if k!='manifest_sha256'}) != manifest['manifest_sha256']:
        raise ValueError('Invalid prefix manifest hash')
    if prior['policy_sha256'] != policy_hash or prior['prefix_manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('Token probe/cache is not bound to this policy')
    for part in ('train','development','confirmation'):
        if set(manifest['split'][part]) != set(split[part+'_episodes']):raise ValueError('Split mismatch')
    train = dict(np.load(args.token_probe/'train_features.npz',allow_pickle=False))
    if len(set(train['keys'])) != len(train['keys']) or set(train['keys']) != set(split['train_episodes']):
        raise ValueError('Donor features must contain exactly the training partition')
    encoder = model.task_context.observer.encoder
    native = torch.from_numpy(train['tokens']).cuda()*encoder.latent_std + encoder.latent_mean
    means = native.mean(0)
    ds = ALIGNDataset(str(args.data),mode='head',cameras=split['cameras'],traj_window=model.chunk_size,dinov2_path=str(args.cache))
    episodes = sorted(split['development_episodes'])
    args.output.mkdir(parents=True)
    protocol = dict(checkpoint=str(args.checkpoint),checkpoint_sha256=policy_hash,
        episodes=episodes,confirmation_opened=False,weights_updated=False,seed=args.seed,noise_draws=args.noise_draws,
        fractions=list(FRACTIONS),arms=list(ARMS),data=str(args.data),cache=str(args.cache),
        prefix_manifest_sha256=file_hash(args.prefix_cache/'manifest.json'),
        train_features_sha256=file_hash(args.token_probe/'train_features.npz'),
        source_sha256={name:file_hash(ROOT/name) for name in ['tools/probe_context_policy_dependence.py','eval/context_interventions.py','models/task_context_policy.py']},
        task_scale=float(head.context_task_scale),execution_scale=float(head.context_execution_scale),
        precision='BF16 action conditioning; FP32 context and DDIM; saved matmul setting',
        metric='Unweighted MSE in dataset action units; first-action MSE also recorded',
        contract='Matched cached expert observations and diffusion noise. No closed-loop success claim.')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[];audits=[];started=time.monotonic()
    try:
        for ei,key in enumerate(episodes):
            ep = ds._episode_keys.index(key)
            path = args.prefix_cache/f'{key}.npz'
            if file_hash(path) != manifest['files'][key]:raise ValueError('Prefix cache hash mismatch')
            with np.load(path,allow_pickle=False) as record:
                summaries = torch.from_numpy(record['summary'].copy()).cuda()[None]
                group = int(record['group'])
            n = ds._get_episode_length(ep)
            if summaries.shape[1]!=n:raise ValueError('Episode length mismatch')
            anchors = [int(n*f) for f in FRACTIONS]
            if any(t+model.chunk_size>n for t in anchors):raise ValueError('Anchor does not have a full target action chunk')
            out = encoder(summaries,torch.tensor([n],device='cuda'),torch.tensor([anchors],device='cuda'))
            payloads = torch.cat([out['tokens'][0],out['execution_token'][0]],dim=-2)
            same,other = select_donors(train['keys'],train['groups'],group,args.seed+ei)
            task_name = ds._read_text(ep)
            for ai,(t,qi) in enumerate(zip(anchors,QUERY_INDICES)):
                frame = torch.from_numpy(np.array(ds._read_frames_dinov2_view(ep,t,1),copy=True)).cuda()
                pose = ds._read_poses(ep,t,1)[:,:6]
                grip = ds._read_poses_gripper(ep,t,1)
                robot = torch.from_numpy(np.concatenate([pose,grip[:,None]],1).astype(np.float32)).cuda()
                raw = frame.reshape(1,model.num_cameras,257,768)
                patches,cls = raw[:,:,:-1].reshape(1,-1,768),raw[:,:,-1]
                # Verify the cached observer summary agrees with the actual policy input pipeline.
                fresh = model.task_context.summaries(patches[:,None],cls[:,None],robot[:,None],
                    torch.tensor([[t/model.task_context.fps]],device='cuda'))[:,0]
                err = float((fresh-summaries[:,t]).abs().max())
                if err>2e-4:raise ValueError(f'{key}/{t}: cached summary mismatch {err}')
                with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(backends=[SDPBackend.MATH]):
                    state = model.state_encoder(robot)[:,None]
                    visual = model.encode_visual_features(patches,state[:,0]).flatten(1)[:,None]
                    visual,state,payload = model.condition_actions(visual,state,payloads[ai:ai+1])
                    replacement = dict(freeze_task=payloads[0:1,:-1],task_mean=means[qi:qi+1],
                        task_swap=native[other:other+1,qi],same_task_swap=native[same:same+1,qi])
                    conds = {arm:intervene(head,head._build_per_step_cond,visual,state,payload,arm,replacement.get(arm)) for arm in ARMS}
                    contexts = {arm:model.action_context(torch.zeros_like(visual) if arm=='visual_zero_control' else visual) for arm in ARMS}
                    task_residual=head.context_task_scale*head.context_task(payload[:,:-1].flatten(1))
                    exec_residual=head.context_execution_scale*head.context_execution(payload[:,-1])
                    audits.append(dict(episode=key,anchor=t,summary_max_error=err,
                        task_residual_rms=float(task_residual.float().square().mean().sqrt()),
                        execution_residual_rms=float(exec_residual.float().square().mean().sqrt()),
                        condition_rms=float(conds['baseline'].float().square().mean().sqrt()),
                        same_task_donor=str(train['keys'][same]),different_task_donor=str(train['keys'][other])))
                target=torch.from_numpy(ds._read_actions(ep,t,model.chunk_size).astype(np.float32)).cuda()[None]
                for draw in range(args.noise_draws):
                    outputs={}
                    for arm in ARMS:
                        torch.manual_seed(args.seed+ei*1000+ai*10+draw)
                        with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(backends=[SDPBackend.MATH]):
                            outputs[arm]=head.sample(conds[arm],**contexts[arm]).float()
                    baseline=outputs['baseline']
                    if not torch.equal(baseline,outputs['repeat_control']):raise ValueError('Matched-noise repeat control failed')
                    for arm,action in outputs.items():
                        row=dict(episode=key,task=task_name,anchor=t,fraction=FRACTIONS[ai],draw=draw,arm=arm,
                            action_mse=float((action-target).square().mean()),baseline_action_mse=float((baseline-target).square().mean()),
                            first_action_mse=float((action[:,0]-target[:,0]).square().mean()),
                            gripper_flips=float(((action[...,6]>.5)!=(baseline[...,6]>.5)).float().mean()))
                        for name,sl in GROUPS.items():
                            row[name+'_delta_mse']=float((action[...,sl]-baseline[...,sl]).square().mean())
                            row[name+'_action_mse']=float((action[...,sl]-target[...,sl]).square().mean())
                        rows.append(row)
            print(f'PROBED {ei+1}/{len(episodes)} {key} elapsed={time.monotonic()-started:.1f}s',flush=True)
            (args.output/'progress.json').write_text(json.dumps(dict(episodes_completed=ei+1,rows=rows,audits=audits))+'\n')
            if hasattr(ds,'_dinov2_memmaps'):ds._dinov2_memmaps.pop(ep,None)
    finally:
        ds.close()
    result=dict(protocol=protocol,aggregate=aggregate(rows),rows=rows,audits=audits,
        per_task={task:aggregate([r for r in rows if r['task']==task]) for task in sorted({r['task'] for r in rows})},
        per_fraction={str(f):aggregate([r for r in rows if r['fraction']==f]) for f in FRACTIONS},
        elapsed_seconds=time.monotonic()-started)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'summary.md').write_text(render(result))
    print(render(result),flush=True)


if __name__=='__main__':main()
