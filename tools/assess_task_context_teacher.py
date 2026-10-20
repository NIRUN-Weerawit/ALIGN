#!/usr/bin/env python3
"""Paired held-out task-information gate for a completed-demonstration teacher.

Development is the default. Confirmation is opened only explicitly after the
development gate, and never participates in checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.task_context_dataset import TaskContextEpisodes, file_hash
from training.train_task_context_teacher import (load_checkpoint,make_loader,to_device,episode_metrics,seed_all)


@torch.no_grad()
def extract(model,dataset,device,mode,batch_size=8):
    arrays={k:[] for k in ['tokens','current','actions','terminal','procedure','procedure_states','group']}
    keys=[]
    for batch in make_loader(dataset,batch_size):
        keys.extend(batch['keys']);batch=to_device(batch,device)
        tokens,current=model.encode(batch['features'],batch['states'],batch['times'],batch['lengths'],batch['anchors'],mode)
        arrays['tokens'].append(tokens.float().cpu().numpy())
        arrays['current'].append(current.float().cpu().numpy())
        for name in ['actions','terminal','procedure','procedure_states','group']:arrays[name].append(batch[name].cpu().numpy())
    return dict(keys=keys,**{k:np.concatenate(v) for k,v in arrays.items()})


@torch.no_grad()
def score(model,arrays,tokens,device,batch_size=8,sample_actions=False,seed=20261020):
    values={}
    for start in range(0,len(tokens),batch_size):
        end=start+batch_size
        batch={k:torch.from_numpy(arrays[k][start:end]).to(device) for k in ['actions','terminal','procedure','procedure_states']}
        z=torch.from_numpy(tokens[start:end]).to(device)
        current=torch.from_numpy(arrays['current'][start:end]).to(device)
        out=model.decode(z,current)
        metrics=episode_metrics(out,batch,model.action_std)
        if sample_actions:
            seed_all(seed+start)
            sampled=model.diffusion.sample(out['condition']).reshape_as(batch['actions'])
            err=(sampled-batch['actions']).square()
            metrics.update(ddim_position=err[...,:3].mean((1,2,3)),
                ddim_rotation=err[...,3:6].mean((1,2,3)),ddim_gripper=err[...,6].mean((1,2)),
                ddim_gripper_accuracy=(sampled[...,6]>.5).eq(batch['actions'][...,6]>.5).float().mean((1,2)))
        for k,v in metrics.items():values.setdefault(k,[]).append(v.cpu().numpy())
    return {k:np.concatenate(v) for k,v in values.items()}


def donor_indices(groups,seed=20261020):
    rng=np.random.default_rng(seed);same=[];wrong=[]
    for i,group in enumerate(groups):
        candidates=np.flatnonzero((groups==group)&(np.arange(len(groups))!=i))
        if not len(candidates):raise ValueError('Same-task donor intervention needs distinct episodes per group')
        same.append(int(rng.choice(candidates)))
        wrong.append(int(rng.choice(np.flatnonzero(groups!=group))))
    return np.asarray(same),np.asarray(wrong)


def paired_difference(a,b,seed=20261020,repeats=10000):
    """Mean a-b; resample whole episodes, retaining every anchor in its cluster."""
    delta=np.asarray(a)-np.asarray(b)
    if delta.ndim!=1 or not np.isfinite(delta).all():raise ValueError('Expected finite paired episode metrics')
    rng=np.random.default_rng(seed)
    boot=delta[rng.integers(0,len(delta),(repeats,len(delta)))].mean(1)
    return dict(mean=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),episodes=len(delta))


def validate_confirmation(report, paths, manifest_hash):
    if report.get('split')!='development' or not report.get('gate',{}).get('passed'):
        raise ValueError('Confirmation requires a passing development assessment')
    if report.get('manifest_sha256')!=manifest_hash:
        raise ValueError('Development/confirmation target manifest differs')
    for name,path in paths.items():
        if report['checkpoints'][name]['sha256']!=file_hash(path):
            raise ValueError('A selected checkpoint changed after the development gate')


def task_probe(train,test):
    result={}
    # Use a single 40%-anchor current feature; full episode tokens are static.
    index=train['tokens'].shape[1]//2
    for name,x,y in [('task_tokens',train['tokens'][:,index].reshape(len(train['keys']),-1),test['tokens'][:,index].reshape(len(test['keys']),-1)),
                     ('current_summary',train['current'][:,index],test['current'][:,index])]:
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000,class_weight='balanced'))
        clf.fit(x,train['group'])
        result[name]=float(balanced_accuracy_score(test['group'],clf.predict(y)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--targets',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--current-checkpoint',required=True)
    p.add_argument('--constant-checkpoint',required=True)
    p.add_argument('--split',choices=['development','confirmation'],default='development')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--sample-actions',action='store_true')
    p.add_argument('--confirm-development-report',help='Required passing development report to unlock confirmation')
    args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(f'Assessment already exists: {output}')
    if args.split=='confirmation':
        if not args.confirm_development_report or not json.loads(Path(args.confirm_development_report).read_text())['gate']['passed']:
            p.error('Confirmation requires a passing development report')
    torch.set_num_threads(2);device=torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(.24)
        torch.backends.cuda.enable_cudnn_sdp(False)
    dataset=TaskContextEpisodes(args.targets,args.split,anchors=3,verify=True)
    if args.split=='confirmation':
        validate_confirmation(json.loads(Path(args.confirm_development_report).read_text()),
            {'full':args.checkpoint,'current':args.current_checkpoint,'constant':args.constant_checkpoint},
            dataset.manifest['manifest_sha256'])
    train_data=TaskContextEpisodes(args.targets,'train',anchors=3,seed=42)
    model,checkpoint=load_checkpoint(args.checkpoint,device)
    if checkpoint['config']['mode']!='full':raise ValueError('Expected full-episode teacher')
    if checkpoint['config']['manifest_sha256']!=dataset.manifest['manifest_sha256']:
        raise ValueError('Checkpoint/data manifest mismatch')
    arrays=extract(model,dataset,device,'full')
    train_arrays=extract(model,train_data,device,'full')
    # One latent mean per episode, then mean over training episodes (not padding/frames).
    mean=train_arrays['tokens'].mean((0,1),keepdims=True)
    same,wrong=donor_indices(arrays['group'])
    interventions=dict(correct=arrays['tokens'],same_task=arrays['tokens'][same],
        wrong_task=arrays['tokens'][wrong],train_mean=np.broadcast_to(mean,arrays['tokens'].shape).copy())
    scores={name:score(model,arrays,z,device,sample_actions=args.sample_actions) for name,z in interventions.items()}
    summaries={name:{k:float(v.mean()) for k,v in metrics.items()} for name,metrics in scores.items()}
    references={}
    for name,path in [('current',args.current_checkpoint),('constant',args.constant_checkpoint)]:
        ref,meta=load_checkpoint(path,device)
        if meta['config']['manifest_sha256']!=dataset.manifest['manifest_sha256'] or meta['config']['mode']!=name:
            raise ValueError('Reference mode or data mismatch')
        # Pooling is used exclusively by the full-episode branch. Current and
        # constant controls never execute that branch, so they can be reused
        # across its readout ablation without retraining an identical control.
        active_config=lambda cfg:{k:v for k,v in cfg.items() if k!='pooling'}
        if active_config(ref.config)!=active_config(model.config) or any(meta['config'][k]!=checkpoint['config'][k]
                for k in ['epochs','batch_size','anchors','seed','lr','diffusion_weight']):
            raise ValueError('Reference architecture or training budget differs')
        other=extract(ref,dataset,device,name)
        if other['keys']!=arrays['keys']:raise ValueError('Reference episode ordering differs')
        references[name]=score(ref,other,other['tokens'],device,sample_actions=args.sample_actions)
        summaries[name]={k:float(v.mean()) for k,v in references[name].items()}
        del ref
    contrasts={}
    for metric in scores['correct']:
        contrasts[metric]={
            'wrong_minus_correct':paired_difference(scores['wrong_task'][metric],scores['correct'][metric]),
            'same_task_minus_correct':paired_difference(scores['same_task'][metric],scores['correct'][metric]),
            'wrong_minus_same_task':paired_difference(scores['wrong_task'][metric],scores['same_task'][metric]),
            'current_minus_full':paired_difference(references['current'][metric],scores['correct'][metric]),
            'constant_minus_full':paired_difference(references['constant'][metric],scores['correct'][metric])}
    # Predeclared milestone gate: outcome/trace compression plus task-specific
    # continuation dependence. It intentionally does not claim online intent.
    requirements={}
    for metric in ['terminal','procedure']:
        for reference in ['current','constant']:
            requirements[f'{metric}_10pct_better_than_{reference}']=(summaries['correct'][metric] < .9*summaries[reference][metric])
            requirements[f'{metric}_{reference}_paired_positive']=(contrasts[metric][f'{reference}_minus_full']['ci95'][0]>0)
    requirements['wrong_task_harms_continuation']=contrasts['action_normalized']['wrong_minus_correct']['ci95'][0]>0
    requirements['task_specific_donor_effect']=contrasts['action_normalized']['wrong_minus_same_task']['ci95'][0]>0
    requirements['task_decodable_above_chance']=False
    probe=task_probe(train_arrays,arrays)
    requirements['task_decodable_above_chance']=probe['task_tokens']>1/len(np.unique(arrays['group']))
    result=dict(split=args.split,manifest_sha256=dataset.manifest['manifest_sha256'],
        checkpoints={name:dict(path=str(Path(path).resolve()),sha256=file_hash(path)) for name,path in
            [('full',args.checkpoint),('current',args.current_checkpoint),('constant',args.constant_checkpoint)]},
        teacher_epoch=checkpoint['epoch'],teacher_pooling=model.config.get('pooling','learned'),
        assessment_sha256=file_hash(__file__),summary=summaries,paired_episode_contrasts=contrasts,
        task_probe=probe,gate=dict(passed=bool(all(requirements.values())),requirements={k:bool(v) for k,v in requirements.items()},
            scope='Useful information in completed demonstrated outcomes/procedure traces and oracle continuation. No causal intent or robot-success claim.'),
        protocol=dict(anchors=[.1,.4,.7],donor_seed=20261020,bootstrap='10000 whole-episode paired resamples; pointwise intervals',
            action_noise='Matched initial DDIM noise per episode batch across conditions and references',
            same_task_donors=[arrays['keys'][i] for i in same],wrong_task_donors=[arrays['keys'][i] for i in wrong]),
        limitations=dataset.manifest['limitations'])
    # Actual current-image persistence and train-mean endpoint residual controls
    # are separate from the independently learned current/constant encoders.
    persistence=[];population=[];class_terminal=[];class_procedure=[]
    group_terminal={int(g):train_arrays['terminal'][train_arrays['group']==g].mean(0) for g in np.unique(train_arrays['group'])}
    group_procedure={int(g):train_arrays['procedure'][train_arrays['group']==g].mean(0) for g in np.unique(train_arrays['group'])}
    for i in range(len(dataset)):
        item=dataset[i]
        initial=item['features'][0]
        current_delta=(item['features'][item['anchors']]-initial)/dataset.norm['visual_scale']
        persistence.append(float(((current_delta-item['terminal'])**2).mean()))
        population.append(float(((dataset.norm['terminal_mean']/dataset.norm['visual_scale']-item['terminal'])**2).mean()))
        class_terminal.append(float(((group_terminal[item['group']]-item['terminal'])**2).mean()))
        class_procedure.append(float(((group_procedure[item['group']]-item['procedure'])**2).mean()))
    result['endpoint_baselines']=dict(current_image_persistence=float(np.mean(persistence)),train_population_delta=float(np.mean(population)),
        privileged_task_id_terminal_template=float(np.mean(class_terminal)),privileged_task_id_procedure_template=float(np.mean(class_procedure)))
    result['task_id_template_contrasts']=dict(
        terminal=paired_difference(np.asarray(class_terminal),scores['correct']['terminal']),
        procedure=paired_difference(np.asarray(class_procedure),scores['correct']['procedure']))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    rows={}
    for condition,metrics in dict(scores,**references).items():
        rows[condition]=[dict(key=key,**{k:float(v[i]) for k,v in metrics.items()}) for i,key in enumerate(arrays['keys'])]
    output.with_suffix('.episodes.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    latent_path=output.with_suffix('.latents.npz')
    np.savez(latent_path,keys=np.asarray(arrays['keys']),tokens=arrays['tokens'],groups=arrays['group'])
    print(json.dumps(dict(gate=result['gate'],summary=summaries,task_probe=probe),indent=2),flush=True)


if __name__=='__main__':main()
