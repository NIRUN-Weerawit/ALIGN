#!/usr/bin/env python3
"""Frozen intent-only policy interventions and causal history diagnostics.

Uses whole-episode splits from an existing strict task probe. Task labels enter
linear probes and donor selection only; instruction-conditioned checkpoints
retain their normal instruction at the action head. No policy is trained.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import inspect
import json
import mmap
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.align_dataset import ALIGNDataset
from models.align_intention import ALIGNIntentionModel
from tools.probe_intent_clustering import fit_decoder, classification_metrics, save_json


def load_policy(checkpoint, device):
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    config = saved['config']
    if config.get('use_memory_bank') or not config.get('use_intent_tokens'):
        raise ValueError('This assessment requires intention tokens without memory')
    if config.get('history_size', 1) != 1 or config.get('head_type') != 'diffusion':
        raise ValueError('This assessment requires history-size-1 diffusion conditioning')
    accepted = inspect.signature(ALIGNIntentionModel.__init__).parameters
    kwargs = {k:v for k,v in config.items() if k in accepted}
    kwargs['action_dim'] = 7
    # Cached visual features make the frozen backbone unnecessary. All other
    # weights, including the complete trained action head, must load strictly.
    with patch('models.align_intention.VisionEncoder', lambda **kw: torch.nn.Linear(1,1,bias=False)):
        model = ALIGNIntentionModel(**kwargs)
    model._build_head_and_bank(config.get('pool_out_dim',
        config['num_cameras']*256*config['compressed_dim']))
    model.vision_encoder = torch.nn.Identity()
    weights = {k.removeprefix('module.'):v for k,v in saved['model_state_dict'].items()}
    model.load_state_dict({k:v for k,v in weights.items() if not k.startswith('vision_encoder.')}, strict=True)
    digest = hashlib.sha256()
    for prefix in ['state_encoder.', 'intention_encoder.']:
        for key, value in sorted(weights.items()):
            if key.startswith(prefix):
                digest.update(key.encode())
                digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return model.to(device).eval(), config, saved['epoch'], digest.hexdigest()


def prefix_inputs(cls, states, t, window=None, mode='normal'):
    start = 0 if window is None else max(0, t+1-window)
    c, s = cls[start:t+1], states[start:t+1]
    if mode == 'reverse_past':
        c = torch.cat([c[:-1].flip(0), c[-1:]])
        s = torch.cat([s[:-1].flip(0), s[-1:]])
    elif mode == 'repeat_current':
        c = c[-1:].expand_as(c)
        s = s[-1:].expand_as(s)
    elif mode != 'normal':
        raise ValueError(mode)
    return c, s


@torch.inference_mode()
def readouts(encoder, cls, states, indices, window=None, mode='normal', capture=False):
    # Different-length prefixes never receive padding. Equal lengths can batch.
    groups = defaultdict(list)
    for i,t in enumerate(indices):
        groups[min(int(t)+1, window or int(t)+1)].append(i)
    out, pre, obs = {}, {}, {}
    captured = []
    hook = encoder.intent_proj.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0])) if capture else None
    try:
        for ids in groups.values():
            values = [prefix_inputs(cls, states, int(indices[i]), window, mode) for i in ids]
            c = torch.stack([v[0] for v in values]);s = torch.stack([v[1] for v in values])
            h, intent = encoder(c, s)
            for j,i in enumerate(ids):
                out[i] = intent[j].flatten().float().cpu().numpy()
                if capture:
                    pre[i] = captured[-1][j].flatten().float().cpu().numpy()
                    obs[i] = h[j,-1].float().cpu().numpy()
            captured.clear()
    finally:
        if hook is not None:hook.remove()
    result = np.stack([out[i] for i in range(len(indices))])
    return (result, np.stack([pre[i] for i in range(len(indices))]),
            np.stack([obs[i] for i in range(len(indices))])) if capture else result


@torch.inference_mode()
def extract(model, config, base, args):
    device = next(model.parameters()).device
    dataset = ALIGNDataset(str(args.data), mode='head', cameras=config['cameras'],
        traj_window=config['chunk_size'], dinov2_path=str(args.cache))
    lookup = {key:i for i,key in enumerate(dataset._episode_keys)}
    selected = np.array([0,5,10,15,20])
    if len(base['progress']) != 21:
        raise ValueError('Expected the recorded 21-position reference probe')
    expected = config['train_episodes']+config['val_episodes']
    if list(base['episodes']) != expected:
        raise ValueError('Reference probe and checkpoint episode order differ')
    features = defaultdict(list)
    head = defaultdict(list)
    before = time.perf_counter()
    for index,key in enumerate(base['episodes']):
        ep = lookup[str(key)];length = dataset._get_episode_length(ep)
        frames = dataset._read_frames_dinov2_view(ep,0,length)
        backing = dataset._dinov2_memmaps[ep]._mmap
        if hasattr(backing,'madvise'):backing.madvise(mmap.MADV_RANDOM)
        camera = frames.reshape(length,config['num_cameras'],257,768)
        cls = torch.from_numpy(np.array(camera[:,:,-1],copy=True)).to(device).float()
        poses = dataset._read_poses(ep,0,length)[:,:6]
        gripper = dataset._read_poses_gripper(ep,0,length)
        if config.get('gripper_state_source') != 'causal':
            raise ValueError('Explicit causal gripper checkpoint required')
        raw = np.concatenate([poses,gripper[:,None]],-1).astype(np.float32)
        state = model.state_encoder(torch.from_numpy(raw).to(device))
        times = base['timesteps'][index,selected]
        full,pre,obs = readouts(model.intention_encoder,cls,state,times,capture=True)
        np.testing.assert_allclose(full,base['intent'][index,selected],atol=3e-4,rtol=2e-3,
            err_msg='Native prefix readouts disagree with the frozen causal reference')
        features['full'].append(full);features['readout_hidden'].append(pre)
        features['observation_hidden'].append(obs)
        features['reset'].append(readouts(model.intention_encoder,cls,state,times,window=1))
        for window in args.windows:
            features[f'recent_{window}'].append(readouts(model.intention_encoder,cls,state,times,window=window))
        for mode in ['reverse_past','repeat_current']:
            features[mode].append(readouts(model.intention_encoder,cls,state,times,mode=mode))
        if not base['train_mask'][index]:
            last = length-config['chunk_size']
            if last < 0:raise ValueError('Episode shorter than action chunk')
            ht = np.rint(base['progress'][selected]*last).astype(int)
            patches = np.array(camera[ht,:,:-1],copy=True).reshape(len(ht),-1,768)
            visual = model.vision_patch_encoder(torch.from_numpy(patches).to(device).float(),state[ht])
            head['visual'].append(visual.flatten(1).cpu().numpy())
            head['state'].append(state[ht].cpu().numpy())
            head['intent'].append(readouts(model.intention_encoder,cls,state,ht))
            head['reset'].append(readouts(model.intention_encoder,cls,state,ht,window=1))
            for window in args.windows:
                head[f'recent_{window}'].append(readouts(model.intention_encoder,cls,state,ht,window=window))
            for mode in ['reverse_past','repeat_current']:
                head[mode].append(readouts(model.intention_encoder,cls,state,ht,mode=mode))
            head['target'].append(np.stack([dataset._read_actions(ep,int(t),config['chunk_size']) for t in ht]))
            head['times'].append(ht)
        del frames,camera
        dataset._dinov2_memmaps.pop(ep,None)
        if (index+1)%50 == 0 or index+1 == len(base['episodes']):
            print(f'EXTRACT {index+1}/{len(base["episodes"])} {time.perf_counter()-before:.1f}s',flush=True)
    dataset.close()
    result = {k:np.stack(v) for k,v in features.items()}
    result.update(labels=base['labels'],episodes=base['episodes'],train_mask=base['train_mask'],progress=base['progress'][selected])
    heads = {k:np.stack(v) for k,v in head.items()}
    heads.update(labels=base['labels'][~base['train_mask']],episodes=base['episodes'][~base['train_mask']],progress=result['progress'])
    np.savez_compressed(args.output/'history_features.npz',**result)
    np.savez_compressed(args.output/'head_features.npz',**heads)
    return result,heads


def analyze_history(features,args):
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits
    train = features['train_mask'].astype(bool)
    y = np.searchsorted(np.unique(features['labels']),features['labels'])
    results = {}
    with threadpool_limits(limits=2):
        for key,x in features.items():
            if x.ndim != 3:continue
            rows = []
            for t,p in enumerate(features['progress']):
                _,pred = fit_decoder(x[train,t],y[train],x[~train,t],args.seed)
                rows.append(dict(progress=float(p),**classification_metrics(y[~train],pred)))
            variance = PCA().fit(x[train,-1]).explained_variance_ratio_
            results[key] = dict(dimensions=x.shape[-1],decoding=rows,
                final_pc1=float(variance[0]),final_pc1_pc2=float(variance[:2].sum()),
                final_participation_dimension=float(1/(variance**2).sum()))
            print('HISTORY',key,[round(r['balanced_accuracy'],3) for r in rows],flush=True)
    save_json(args.output/'history_results.json',results)
    return results


def projection_diagnostics(features,output,seed=42,weight=None):
    """Dimension-matched controls isolate learned projection from input width."""
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits
    train=features['train_mask'].astype(bool)
    y=np.searchsorted(np.unique(features['labels']),features['labels'])
    result={'protocol':'Final-position frozen features; FP32 128-D label-free projections; scaling/classifier fit training episodes only',
            'features':{}}
    with threadpool_limits(limits=2):
        for key in ['readout_hidden','observation_hidden']:
            x=features[key][:,-1]
            rows=[]
            for offset in range(3):
                rng=np.random.default_rng(seed+offset)
                matrix=rng.normal(size=(x.shape[-1],128)).astype(np.float32)*np.float32(1/np.sqrt(128.))
                projected=x@matrix
                _,pred=fit_decoder(projected[train],y[train],projected[~train],seed)
                rows.append(dict(seed=seed+offset,**classification_metrics(y[~train],pred)))
            pca=PCA(n_components=128,svd_solver='randomized',random_state=seed).fit(x[train])
            projected=pca.transform(x)
            _,pred=fit_decoder(projected[train],y[train],projected[~train],seed)
            result['features'][key]=dict(random_128=rows,training_pca_128=classification_metrics(y[~train],pred))
        if weight is not None:
            singular=np.linalg.svd(weight,compute_uv=False);energy=singular**2/(singular**2).sum()
            result['trained_projection_weight']=dict(first_two_energy=float(energy[:2].sum()),
                participation_dimension=float(1/(energy**2).sum()))
    save_json(output/'projection_controls.json',result)
    return result


def donor_indices(labels,seed,same_task):
    rng = np.random.default_rng(seed);labels=np.asarray(labels)
    donors=[]
    for i,label in enumerate(labels):
        valid=np.flatnonzero((labels==label) if same_task else (labels!=label))
        valid=valid[valid!=i]
        if not len(valid):raise ValueError('Insufficient donor episodes')
        donors.append(int(rng.choice(valid)))
    return np.asarray(donors)


def condition_cases(model,head,seed,device,return_context=False):
    e,p = head['intent'].shape[:2]
    tensor=lambda key:torch.from_numpy(head[key]).to(device).float().flatten(0,1)
    visual,state=tensor('visual')[:,None],tensor('state')[:,None]
    intent=tensor('intent').reshape(e*p,model.num_intent_tokens,model.intent_dim)
    labels=np.repeat(head['labels'],p).tolist()
    text=model.encode_task_text(labels) if model.use_text else None
    same=donor_indices(head['labels'],seed,True)
    other=donor_indices(head['labels'],seed,False)
    same_flat=(same[:,None]*p+np.arange(p)[None]).reshape(-1)
    other_flat=(other[:,None]*p+np.arange(p)[None]).reshape(-1)
    def build(i=intent,v=visual,s=state,t=text):
        cond=model.action_condition(v,s,i,t)
        return (cond,model.action_context(v)) if return_context else cond
    baseline=build()
    repeat=((baseline[0].clone(),{k:v.clone() for k,v in baseline[1].items()})
            if return_context else baseline.clone())
    cases={'baseline':baseline,'repeat_control':repeat,
        'intent_zero':build(torch.zeros_like(intent)),
        'intent_same_task':build(intent[same_flat]),'intent_other_task':build(intent[other_flat]),
        'intent_early':build(tensor('intent').reshape(e,p,-1)[:,0:1].expand(-1,p,-1).reshape_as(intent)),
        'visual_zero':build(v=torch.zeros_like(visual)),'state_zero':build(s=torch.zeros_like(state))}
    for key in ['reset','reverse_past','repeat_current']+[f'recent_{w}' for w in model._assessment_windows]:
        cases['intent_'+key]=build(tensor(key).reshape_as(intent))
    if text is not None:cases['text_other_task']=build(t=text[other_flat])
    return cases,dict(same_task=same.tolist(),other_task=other.tolist())


def paired_interval(delta,seed,repeats=3000):
    # Episodes are the resampling unit; each already averages phases/noise draws.
    rng=np.random.default_rng(seed)
    estimates=np.mean(delta[rng.integers(len(delta),size=(repeats,len(delta)))],axis=1)
    return np.quantile(estimates,[.025,.975]).tolist()


@torch.inference_mode()
def assess_head(model,heads,args):
    device=next(model.parameters()).device
    e,p=heads['intent'].shape[:2]
    target=torch.from_numpy(heads['target']).to(device).float().flatten(0,1)
    head=model.intention_head
    totals=defaultdict(lambda:defaultdict(list));donors=[]
    valid=torch.where(head.alpha_bar>1e-8)[0]
    last=int(valid[-1]);steps=sorted(set([1,max(1,last//2),max(1,round(.9*last))]))
    for draw in range(args.noise_draws):
        cases,chosen=condition_cases(model,heads,args.seed+draw,device,return_context=True);donors.append(chosen)
        outputs={k:[] for k in cases};noise_errors={k:[] for k in cases}
        for start in range(0,len(target),args.head_batch_size):
            stop=min(len(target),start+args.head_batch_size);gt=target[start:stop]
            seed=args.seed+draw*10000+start
            torch.manual_seed(seed+500000);noise=torch.randn_like(gt)
            noisy=[head.alpha_bar[t].sqrt()*gt+head.sigma[t]*noise for t in steps]
            for name,(cond,visual_kwargs) in cases.items():
                context_slice={key:value[start:stop] for key,value in visual_kwargs.items()}
                torch.manual_seed(seed)
                outputs[name].append(head.sample(cond[start:stop],**context_slice).cpu())
                errs=[]
                for t,x in zip(steps,noisy):
                    eps=head.predict_noise(x,torch.full((len(gt),),t,device=device,dtype=torch.long),cond[start:stop],**context_slice)
                    errs.append((eps-noise).square().mean(1).cpu())
                noise_errors[name].append(torch.stack(errs).mean(0))
        actions={k:torch.cat(v).reshape(e,p,model.chunk_size,7).numpy() for k,v in outputs.items()}
        denoise={k:torch.cat(v).reshape(e,p,7).numpy() for k,v in noise_errors.items()}
        np.testing.assert_array_equal(actions['baseline'],actions['repeat_control'])
        np.testing.assert_array_equal(denoise['baseline'],denoise['repeat_control'])
        truth=heads['target'];base=actions['baseline']
        for name,action in actions.items():
            stats=totals[name]
            stats['action_mse'].append(((action-truth)**2).mean((-1,-2)))
            stats['action_delta_sq'].append(((action-base)**2).mean((-1,-2)))
            stats['noise_mse'].append(denoise[name].mean(-1))
            for group,sl in [('position',slice(0,3)),('rotation',slice(3,6)),('gripper',slice(6,7))]:
                stats[group+'_mse'].append(((action[...,sl]-truth[...,sl])**2).mean((-1,-2)))
                stats[group+'_delta_sq'].append(((action[...,sl]-base[...,sl])**2).mean((-1,-2)))
            pred=action[...,6]>.5;true=truth[...,6]>.5
            stats['gripper_flip_fraction'].append((pred!=(base[...,6]>.5)).mean(-1))
            for k,bits in [('tp',pred&true),('tn',~pred&~true),('fp',pred&~true),('fn',~pred&true)]:
                stats[k].append(bits.sum(-1))
        print(f'HEAD noise draw {draw+1}/{args.noise_draws}',flush=True)
    averaged={name:{k:np.stack(v).mean(0) for k,v in metrics.items()} for name,metrics in totals.items()}
    rows=[];summary={}
    for name,metrics in averaged.items():
        result={k:float(v.mean()) for k,v in metrics.items() if k not in ['tp','tn','fp','fn']}
        result['action_delta_rms']=float(np.sqrt(result['action_delta_sq']))
        for group in ['position','rotation','gripper']:
            result[group+'_delta_rms']=float(np.sqrt(result[group+'_delta_sq']))
        for metric in ['action_mse','noise_mse']:
            delta=(metrics[metric]-averaged['baseline'][metric]).mean(1)
            result[metric+'_change']=float(delta.mean())
            result[metric+'_change_95pct_episode_bootstrap']=paired_interval(delta,args.seed)
        tp,tn,fp,fn=[float(metrics[k].sum()) for k in ['tp','tn','fp','fn']]
        result['gripper_balanced_accuracy']=.5*(tp/(tp+fn)+tn/(tn+fp)) if (tp+fn)*(tn+fp) else None
        summary[name]=result
        for i,ep in enumerate(heads['episodes']):
            for t,fraction in enumerate(heads['progress']):
                rows.append(dict(episode=str(ep),task=str(heads['labels'][i]),progress=float(fraction),
                    intervention=name,**{k:float(v[i,t]) for k,v in metrics.items()}))
    for name,reference in [('intent_other_task','intent_same_task'),('intent_reset','baseline')]:
        delta=(averaged[name]['action_mse']-averaged[reference]['action_mse']).mean(1)
        summary[name]['action_mse_change_vs_'+reference]=dict(mean=float(delta.mean()),
            paired_95pct=paired_interval(delta,args.seed))
    result=dict(summary=summary,rows=rows,donor_episode_indices=donors,
        noise_probe_timesteps=steps,sampling_timesteps=head.sampling_timesteps().tolist())
    save_json(args.output/'head_results.json',result)
    phase_results={}
    for t,fraction in enumerate(heads['progress']):
        delta=averaged['intent_other_task']['action_mse'][:,t]-averaged['intent_same_task']['action_mse'][:,t]
        phase_results[str(float(fraction))]=dict(other_minus_same_mean=float(delta.mean()),
            paired95=paired_interval(delta,args.seed),mean_mse={name:float(metrics['action_mse'][:,t].mean())
                for name,metrics in averaged.items()})
    save_json(args.output/'phase_head_results.json',phase_results)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=ROOT/'data/libero_goal.h5')
    parser.add_argument('--cache',type=Path,default=ROOT/'data/libero_goal.dinov2.external')
    parser.add_argument('--windows',nargs='+',type=int,default=[8,32,128])
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--noise-draws',type=int,default=3)
    parser.add_argument('--head-batch-size',type=int,default=32)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--reuse-features',action='store_true')
    args=parser.parse_args()
    if min(args.windows)<2 or args.noise_draws<1 or args.head_batch_size<1:parser.error('Invalid positive sizes')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((args.probe/'extraction.json').read_text())
    model,config,epoch,digest=load_policy(metadata['checkpoint'],torch.device(args.device))
    if digest!=metadata['encoder_sha256']:raise ValueError('Checkpoint encoder changed since the reference probe')
    model._assessment_windows=args.windows
    if args.reuse_features:
        prior=json.loads((args.output/'protocol.json').read_text())
        if prior['encoder_sha256']!=digest or prior['windows']!=args.windows:
            raise ValueError('Cached feature checkpoint/windows do not match this assessment')
        with np.load(args.output/'history_features.npz') as z:features={k:z[k] for k in z.files}
        with np.load(args.output/'head_features.npz') as z:heads={k:z[k] for k in z.files}
    else:
        with np.load(args.probe/'intent_embeddings.npz') as z:base={k:z[k] for k in z.files}
        features,heads=extract(model,config,base,args)
    history=analyze_history(features,args)
    projection_diagnostics(features,args.output,args.seed,
        model.intention_encoder.intent_proj.weight.detach().float().cpu().numpy())
    head=assess_head(model,heads,args)
    save_json(args.output/'protocol.json',dict(checkpoint=metadata['checkpoint'],epoch=epoch,
        encoder_sha256=digest,config=config,precision='FP32, TF32 disabled',train_episodes=int(features['train_mask'].sum()),
        held_out_episodes=int((~features['train_mask']).sum()),windows=args.windows,noise_draws=args.noise_draws,
        progress=features['progress'].tolist(),head_anchors='Five positions from first through last complete action chunk; no padded targets',
        history_anchors='Five positions from first through last observed frame, matching the reference task probe',
        controls='Matched diffusion noise, fixed current visual/state and instruction, same/different-task donors at matched normalized progress',
        caveats=['Offline expert-episode action error and dependence, not simulator success.',
                 'Zeroing and counterfactual histories may be out of distribution.',
                 'Same/different-task donors also differ in episode-specific state and trajectory.',
                 'No policy weights updated; task decoding does not establish future-goal inference.']))
    print('COMPLETE',args.output,flush=True)


if __name__=='__main__':main()
