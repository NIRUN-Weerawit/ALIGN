#!/usr/bin/env python3
"""Paired interventions on saved heads using held-out cached observations.

No model weights are changed. Noise draws, crops, and memory histories are
matched across interventions. This measures dependence, not simulator success.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_intention_ablation import (ALIGNDataset, ALIGNIntentionModel,
    CachedEpisodes, collate_segments, seed_everything, atomic_json)

BANK_FIELDS = ('perceptual_bank', 'state_bank', 'cognitive_bank', '_count')
GROUPS = {'position': slice(0,3), 'rotation': slice(3,6), 'gripper': slice(6,7)}


def snapshot(bank):
    return {key: getattr(bank, key).clone() for key in BANK_FIELDS}


def restore(bank, saved, permutation=None):
    for key, value in saved.items():
        setattr(bank, key, value.clone() if permutation is None else value[permutation].clone())


def load_model(path, cameras):
    with torch.serialization.safe_globals([type(torch.__version__)]):
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
    c = ckpt['config']
    kwargs = {key: c[key] for key in ('state_dim','mamba_output_dim','chunk_size',
        'history_size','compressed_dim','head_type','head_d_model','use_intent_tokens',
        'intent_dim','use_memory_bank','memory_bank_len')}
    for key, default in [('num_intent_tokens',1),('mamba_d_state',16),('mamba_d_conv',4),('mamba_expand',2)]:
        kwargs[key] = c.get(key, default)
    model = ALIGNIntentionModel(action_dim=7, num_cameras=len(cameras), **kwargs)
    model._build_head_and_bank(c.get('pool_out_dim',256*len(cameras)*c['compressed_dim']))
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    vision = model.vision_encoder
    model.vision_encoder = torch.nn.Identity()
    model.cuda().eval()
    model.vision_encoder = vision
    return model, ckpt['epoch']


@torch.no_grad()
def encode(model, batch):
    frames = batch['frames_segment'].cuda(non_blocking=True)
    states = batch['states_segment'].cuda().float()
    B,S,T,D = frames.shape
    V = T//257
    camera = frames.reshape(B*S,V,257,D)
    cls = camera[:,:,-1].reshape(B,S,V,D)
    patches = camera[:,:,:-1].reshape(B*S,V*256,D)
    state = model.state_encoder(states.reshape(B*S,7)).reshape(B,S,-1)
    visual = model.vision_patch_encoder(patches, state.reshape(B*S,-1)).reshape(B,S,-1)
    n = S-model.chunk_size+1
    intent = (model.intention_encoder.forward_sequence(cls[:,:n], state[:,:n],
               readout_start=model.history_size-1)
              if model.intention_encoder is not None else None)
    return visual, state, intent


@torch.no_grad()
def evaluate_variant(model, loader, anchors, seed):
    totals = defaultdict(lambda: defaultdict(float))
    rows = []
    for batch_index, batch in enumerate(loader):
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            visual, state, intents = encode(model,batch)
            B,S,_ = visual.shape
            if B < 2:
                raise ValueError('Shuffling requires at least two distinct episodes per batch')
            permutation = torch.arange(B,device='cuda').roll(1)
            if model.use_memory_bank:
                model.memory_module.reset(B, torch.device('cuda'))
            for t in range(S-model.chunk_size+1):
                if t < model.history_size-1:
                    continue
                p,s = visual[:,t-model.history_size+1:t+1],state[:,t-model.history_size+1:t+1]
                i = None if intents is None else intents[:,t]
                before = snapshot(model.memory_module) if model.use_memory_bank and t in anchors else None
                fused = model.condition_actions(p,s,i)
                if t not in anchors:
                    continue
                after = snapshot(model.memory_module) if before is not None else None
                conds = {'baseline': model.intention_head(*fused)}
                conds['repeat_control'] = conds['baseline'].clone()
                if i is not None:
                    conds['intent_zero'] = model.intention_head(fused[0],fused[1],torch.zeros_like(fused[2]))
                    conds['intent_shuffle'] = model.intention_head(fused[0],fused[1],fused[2][permutation])
                if before is not None:
                    conds['memory_bypass'] = model.intention_head(p,s,i)
                    restore(model.memory_module,before,permutation)
                    conds['memory_shuffle'] = model.intention_head(*model.condition_actions(p,s,i))
                    restore(model.memory_module,after)
                conds['visual_zero_control'] = model.intention_head(torch.zeros_like(fused[0]),fused[1],fused[2])
                conds['state_zero_control'] = model.intention_head(fused[0],torch.zeros_like(fused[1]),fused[2])
                target = batch['actions_segment'][:,t:t+model.chunk_size].cuda().float()
                draw_seed = seed + batch_index*100 + t
                seed_everything(draw_seed)
                noise = torch.randn_like(target)
                probe_indices = [1,5,9]
                inputs = [(model.intention_head.alpha_bar[k].sqrt()*target +
                          model.intention_head.sigma[k]*noise) for k in probe_indices]
                outputs = {}
                for name,cond in conds.items():
                    seed_everything(draw_seed)
                    action = model.intention_head.sample(cond).float()
                    eps = torch.stack([model.intention_head.predict_noise(x,
                        torch.full((B,),k,device='cuda',dtype=torch.long),cond).float()
                        for k,x in zip(probe_indices,inputs)])
                    outputs[name] = (action,eps)
                baseline, baseline_eps = outputs['baseline']
                for name,(action,eps) in outputs.items():
                    acc = totals[name]
                    acc['count'] += B*model.chunk_size
                    acc['gripper_flips'] += ((action[:,:,6]>.5)!=(baseline[:,:,6]>.5)).sum().item()
                    acc['gripper_correct'] += ((action[:,:,6]>.5)==(target[:,:,6]>.5)).sum().item()
                    acc['max_action_delta'] = max(acc['max_action_delta'],(action-baseline).abs().max().item())
                    acc['max_noise_delta'] = max(acc['max_noise_delta'],(eps-baseline_eps).abs().max().item())
                    row = dict(batch=batch_index,t=t,intervention=name,
                               bank_count=0 if before is None else before['_count'].tolist())
                    for group,sl in GROUPS.items():
                        delta = (action[:,:,sl]-baseline[:,:,sl]).square().mean().item()
                        mse = (action[:,:,sl]-target[:,:,sl]).square().mean().item()
                        acc[group+'_delta_sq'] += delta*B*model.chunk_size
                        acc[group+'_mse'] += mse*B*model.chunk_size
                        acc[group+'_baseline_sq'] += baseline[:,:,sl].square().mean().item()*B*model.chunk_size
                        acc[group+'_noise_delta_sq'] += (eps[:,:,:,sl]-baseline_eps[:,:,:,sl]).square().mean().item()*B*model.chunk_size
                        acc[group+'_noise_mse'] += (eps[:,:,:,sl]-noise[None,:,:,sl]).square().mean().item()*B*model.chunk_size
                        row[group+'_delta_rms'] = delta**.5
                        row[group+'_mse'] = mse
                    rows.append(row)
        print(f'  batch {batch_index+1}/{len(loader)}',flush=True)
    control = totals['repeat_control']
    if control['max_action_delta'] != 0 or control['max_noise_delta'] != 0:
        raise RuntimeError('Identical-condition control failed; intervention comparison is invalid')
    result = {}
    for name,acc in totals.items():
        n = acc['count']
        r = dict(action_predictions=int(n),gripper_flip_fraction=acc['gripper_flips']/n,
                 gripper_accuracy=acc['gripper_correct']/n,max_action_delta=acc['max_action_delta'],
                 max_noise_delta=acc['max_noise_delta'])
        for group in GROUPS:
            r[group+'_action_delta_rms'] = (acc[group+'_delta_sq']/n)**.5
            r[group+'_action_mse'] = acc[group+'_mse']/n
            r[group+'_baseline_action_rms'] = (acc[group+'_baseline_sq']/n)**.5
            r[group+'_noise_delta_rms'] = (acc[group+'_noise_delta_sq']/n)**.5
            r[group+'_noise_mse'] = acc[group+'_noise_mse']/n
        result[name] = r
    return result,rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--anchors',type=int,nargs='+',default=[0,6,12])
    args = parser.parse_args()
    manifest = json.loads((args.run/'manifest.json').read_text())
    if manifest['head_type'] != 'diffusion':
        raise ValueError('This diagnostic currently probes diffusion epsilon predictors')
    if any(t<manifest['history_size']-1 or t>manifest['segment_length']-manifest['chunk_size'] for t in args.anchors):
        raise ValueError('anchors must be valid conditioning timesteps')
    args.output.mkdir(parents=True,exist_ok=True)
    cameras = manifest.get('cameras',['image','wrist_image'])
    dataset = ALIGNDataset(manifest['data'],mode='head',cameras=cameras,
        traj_window=manifest['segment_length'],dinov2_path=manifest['cache'])
    task_groups = defaultdict(list)
    for ep in manifest['val_episodes']:
        task = dataset._h5[f'{ep}/texts'][()]
        task_groups[str(task)].append(dataset._episode_keys.index(ep))
    # Interleave tasks so a cyclic batch permutation exchanges distinct tasks,
    # rather than mostly neighboring demonstrations of the same task.
    groups = list(task_groups.values())
    episodes = [group[k] for k in range(max(map(len,groups))) for group in groups if k<len(group)]
    if len(episodes)%args.batch_size == 1:
        raise ValueError('Choose a batch size that avoids a final singleton batch')
    for start in range(0,len(episodes),args.batch_size):
        batch_tasks = [str(dataset._h5[f'{dataset._episode_keys[e]}/texts'][()])
                       for e in episodes[start:start+args.batch_size]]
        if any(task==batch_tasks[(k-1)%len(batch_tasks)] for k,task in enumerate(batch_tasks)):
            raise ValueError('Batch shuffling would pair identical tasks; choose a different batch size')
    cached = CachedEpisodes(dataset,episodes,manifest['segment_length'],manifest['seed']+1000,False)
    loader = DataLoader(cached,batch_size=args.batch_size,shuffle=False,num_workers=0,
                        pin_memory=True,collate_fn=collate_segments)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.3)
    report = dict(protocol=dict(run=str(args.run.resolve()),held_out_episodes=len(episodes),
        episode_keys=[dataset._episode_keys[ep] for ep in episodes],batch_size=args.batch_size,anchors=args.anchors,
        crops='original fixed validation crops',seed=manifest['seed']+20000,
        intent_intervention='final head intent; perceptual/state conditioning held fixed; shuffle across distinct tasks',
        memory_shuffle='all bank streams swapped between distinct tasks/episodes, queries held fixed; baseline history restored',
        precision='BF16 conditioning/epsilon probes; FP32 DDIM denoiser and state',noise_probe_timesteps=[1,5,9],
        caveat='Dependence and held-out action errors do not establish closed-loop benefit. Old endpoint buffers retained.'),variants={})
    for name in manifest['variants']:
        model,epoch = load_model(args.run/name/'intention_best.pt',cameras)
        print(f'{name} best epoch {epoch}',flush=True)
        results,rows = evaluate_variant(model,loader,set(args.anchors),manifest['seed']+20000)
        report['variants'][name] = dict(epoch=epoch,sampling_timesteps=model.intention_head.sampling_timesteps().tolist(),results=results)
        atomic_json(args.output/(name+'.json'),dict(results=results,rows=rows))
        atomic_json(args.output/'summary.json',report)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    lines = ['# Trained head condition dependence','',
        f"{len(episodes)} held-out episodes; {len(args.anchors)} matched-noise anchors per episode; original best checkpoints.",
        'Shuffles exchange distinct tasks. Values are dataset action units. Intent interventions change final head tokens; memory interventions change retrieval.',
        '', '| Model | Intervention | Position delta RMS | Rotation delta RMS | Gripper delta RMS | Gripper flips | Position MSE | Gripper accuracy |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,v in report['variants'].items():
        for intervention,r in v['results'].items():
            lines.append(f"| {name} | {intervention} | {r['position_action_delta_rms']:.6f} | {r['rotation_action_delta_rms']:.6f} | {r['gripper_action_delta_rms']:.6f} | {r['gripper_flip_fraction']:.1%} | {r['position_action_mse']:.6f} | {r['gripper_accuracy']:.1%} |")
    lines += ['', 'Repeated-condition controls must have exactly zero action/noise changes.',
        'Zeroing is out of distribution. Shuffling tests information sensitivity but does not prove semantic understanding.',
        'Memory is empty at t=0, partially populated at t=6, and at capacity before t=12; histories remain independent.',
        'These weights predate the schedule and gripper-loss fixes. New training is required to measure those improvements.']
    (args.output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(f"Saved {args.output/'comparison.md'}",flush=True)

if __name__=='__main__':
    main()
