#!/usr/bin/env python3
"""Train completed-demonstration teachers with matched current/constant controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.task_context_dataset import TaskContextEpisodes, collate_task_context, file_hash
from models.task_context_teacher import TaskContextTeacher, teacher_loss


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def to_device(batch,device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def episode_metrics(output,batch,action_std):
    error=(output["regression"].float()-batch["actions"].float())**2
    return dict(
        action_normalized=(error/action_std.square()).mean((1,2,3)),
        position=error[:,:,:,:3].mean((1,2,3)),
        rotation=error[:,:,:,3:6].mean((1,2,3)),
        gripper=error[:,:,:,6].mean((1,2)),
        gripper_accuracy=(output["regression"][...,6]>.5).eq(batch["actions"][...,6]>.5).float().mean((1,2)),
        terminal=(output["terminal"].float()-batch["terminal"][:,None]).square().flatten(1).mean(1),
        procedure=(output["procedure"].float()-batch["procedure"][:,None]).square().flatten(1).mean(1),
        procedure_state=(output["procedure_states"].float()-batch["procedure_states"][:,None]).square().flatten(1).mean(1))


@torch.no_grad()
def evaluate(model,loader,device,mode):
    model.eval()
    rows=[]
    for batch in loader:
        batch=to_device(batch,device)
        # FP32 validation is deterministic and does not consume training RNG.
        out=model(batch,mode)
        metrics=episode_metrics(out,batch,model.action_std)
        for i,key in enumerate(batch['keys']):
            rows.append(dict(key=key,**{k:float(v[i]) for k,v in metrics.items()}))
    means={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='key'}
    means['selection']=float(np.mean([means[k] for k in ['action_normalized','terminal','procedure','procedure_state']]))
    return means,rows


def make_loader(dataset,batch_size,shuffle=False,seed=0):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=0,
        collate_fn=collate_task_context,generator=torch.Generator().manual_seed(seed))


def load_checkpoint(path,device='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=TaskContextTeacher(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'],strict=True)
    return model.to(device).eval(),checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    from data.training_split import PIPER_TARGETS, validate_goal_training_manifest, validation_partition
    p.add_argument('--targets',default=str(PIPER_TARGETS.parent))
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['full','current','constant'],default='full')
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--anchors',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--width',type=int,default=256)
    p.add_argument('--diffusion-width',type=int,default=64)
    p.add_argument('--diffusion-weight',type=float,default=1.)
    p.add_argument('--pooling',choices=['learned','quarters'],default='learned')
    p.add_argument('--decoder-style',choices=['flat','temporal'],default='flat')
    p.add_argument('--device',default='cuda')
    p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--evaluate-every',type=int,default=2)
    p.add_argument('--memory-fraction',type=float,default=.24)
    p.add_argument('--final-refit',action=argparse.BooleanOptionalAction,default=True,
        help='Fixed-budget training with no validation or checkpoint selection; requires empty development split')
    p.add_argument('--allow-legacy-split',action='store_true',help='Explicitly reproduce a non-Piper historical Goal cache')
    args=p.parse_args()
    if min(args.epochs,args.batch_size,args.anchors,args.width,args.evaluate_every)<=0 or args.diffusion_weight<0:
        p.error('Invalid training configuration')
    output=Path(args.output)
    if output.exists():raise FileExistsError(f'Use a new run directory: {output}')
    output.mkdir(parents=True)
    device=torch.device(args.device)
    torch.set_num_threads(2)
    if device.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cuda.enable_cudnn_sdp(False)
    seed_all(args.seed)
    train=TaskContextEpisodes(args.targets,'train',anchors=args.anchors,seed=args.seed,verify=True)
    if not args.allow_legacy_split:validate_goal_training_manifest(train.manifest)
    if args.final_refit and train.manifest['split']['development']:
        raise ValueError('Final refit requires an empty development split')
    development=None if args.final_refit else TaskContextEpisodes(args.targets,validation_partition(train.manifest['split']),anchors=3,seed=args.seed,verify=True)
    train_loader=make_loader(train,args.batch_size,True,args.seed)
    val_loader=None if args.final_refit else make_loader(development,args.batch_size)
    model=TaskContextTeacher(cameras=len(train.manifest['cameras']),raw_dim=train.manifest['feature_shape'][-1],width=args.width,
        horizon=train.manifest['action_horizon'],procedure_points=train.manifest['procedure_points'],
        diffusion_width=args.diffusion_width,pooling=args.pooling,decoder_style=args.decoder_style).to(device)
    actions=np.concatenate([np.load(train.root/f'{key}.npz')['actions'] for key in train.keys])
    with torch.no_grad():
        model.action_mean.copy_(torch.from_numpy(actions.mean(0)))
        model.action_std.copy_(torch.from_numpy(np.maximum(actions.std(0),.05)))
    del actions
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    config=dict(vars(args),model_config=model.config,manifest_sha256=train.manifest['manifest_sha256'],
        checkpoint_selection='fixed_budget_final' if args.final_refit else 'development',
        source_hashes={name:file_hash(ROOT/name) for name in ['data/task_context_dataset.py','models/task_context_teacher.py','training/train_task_context_teacher.py']},
        parameters=sum(v.numel() for v in model.parameters()),
        contract='Completed-demonstration oracle; observation-only; no causal takeover claim')
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(config),flush=True)
    best=float('inf');step=0;start=time.perf_counter()
    for epoch in range(args.epochs):
        train.epoch=epoch
        model.train();sums={};count=0
        for batch in train_loader:
            batch=to_device(batch,device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                out=model(batch,args.mode)
                loss,parts=teacher_loss(model,out,batch,diffusion_weight=args.diffusion_weight)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite training loss at {epoch}/{step}')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradients')
            optimizer.step();step+=1
            n=len(batch['keys']);count+=n
            for k,v in dict(loss=loss,**parts).items():sums[k]=sums.get(k,0.)+float(v.detach())*n
            if args.max_steps and step>=args.max_steps:break
        row=dict(epoch=epoch+1,steps=step,elapsed_s=time.perf_counter()-start,
            train={k:v/count for k,v in sums.items()})
        if not args.final_refit and ((epoch+1)%args.evaluate_every==0 or epoch==0 or epoch+1==args.epochs or (args.max_steps and step>=args.max_steps)):
            metrics,rows=evaluate(model,val_loader,device,args.mode)
            row['development']=metrics
            if metrics['selection']<best:
                best=metrics['selection']
                checkpoint=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    model_config=model.config,config=config,epoch=epoch+1,steps=step,development=metrics)
                temporary=output/'best.tmp'
                torch.save(checkpoint,temporary);temporary.replace(output/'teacher_best.pt')
                (output/'best_development.json').write_text(json.dumps(rows,indent=2)+'\n')
        if device.type=='cuda':row['peak_allocated_mb']=torch.cuda.max_memory_allocated()/1024**2
        print(json.dumps(row),flush=True)
        with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if args.max_steps and step>=args.max_steps:break
    if args.final_refit:
        checkpoint=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            model_config=model.config,config=config,epoch=epoch+1,steps=step,
            checkpoint_selection='fixed_budget_final')
        temporary=output/'final.tmp'
        torch.save(checkpoint,temporary);temporary.replace(output/'teacher_final.pt')
    print(json.dumps(dict(completed=True,best_selection=None if args.final_refit else best,steps=step,elapsed_s=time.perf_counter()-start)),flush=True)


if __name__=='__main__':main()
