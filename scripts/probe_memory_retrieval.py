#!/usr/bin/env python3
"""Measure historical content suppression before the action head, without training.

One held-out episode per distinct task is used. Queries are zero to represent a
current observation outage. Wrong-task values replace all historical values;
keys retain the same recorded ages so this isolates content rather than timing.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.probe_condition_dependence import (ROOT, ALIGNDataset, CachedEpisodes,
    collate_segments, load_model, encode, atomic_json)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--patch-temporal-override',action=argparse.BooleanOptionalAction,default=None,help='Diagnostic-only architecture intervention; weights stay fixed, not a policy-quality comparison')
    parser.add_argument('--frames',type=int,default=40)
    parser.add_argument('--batch-size',type=int,default=4)
    args=parser.parse_args()
    if args.frames<1 or args.batch_size<2:parser.error('Need past observations and at least two tasks')
    manifest=json.loads((args.run/'manifest.json').read_text())
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    dataset=ALIGNDataset(manifest['data'],mode='head',cameras=manifest['cameras'],
                         traj_window=manifest['segment_length'],dinov2_path=manifest['cache'])
    tasks={}
    for ep in manifest['val_episodes']:
        tasks.setdefault(str(dataset._h5[f'{ep}/texts'][()]),dataset._episode_keys.index(ep))
    indices=list(tasks.values())[:args.batch_size]
    if len(indices)!=args.batch_size:parser.error('Not enough distinct held-out tasks')
    cached=CachedEpisodes(dataset,indices,manifest['segment_length'],manifest['seed']+1000,
                          False,'episode',16,manifest['chunk_size'])
    batch=collate_segments([cached[k] for k in range(len(indices))])
    if int(min(batch['segment_len']))<=args.frames:parser.error('Frame exceeds a selected episode')
    model,epoch=load_model(args.checkpoint,manifest['cameras'])
    if not model.use_memory_bank or model.memory_mode!='episodic':
        parser.error('Requires an episodic-memory checkpoint')
    bank=model.memory_module
    if args.patch_temporal_override is not None:
        if bank.patch_dim is None:parser.error("Temporal layout intervention requires a patch bank")
        bank.patch_temporal=args.patch_temporal_override
    report=dict(checkpoint=str(args.checkpoint.resolve()),epoch=epoch,
                frames=args.frames,episode_keys=[dataset._episode_keys[k] for k in indices],
                context_only=model.memory_context_only,value_preserving=model.memory_value_preserving,patch_temporal=bank.patch_temporal,saved_patch_temporal=model.memory_patch_temporal,
                architecture_override=args.patch_temporal_override,
                protocol='Zero current queries; cross-task values swapped; key ages held fixed; BF16 conditioning',
                caveat='Feature sensitivity diagnostic, not action accuracy or policy success.',streams={})
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        p,s,c=encode(model,batch);bank.reset(len(indices),torch.device('cuda'))
        for t in range(args.frames):
            bank.observe_only(p[:,t],s[:,t],None if c is None else c[:,t],
                              timestamp=torch.full((len(indices),),float(t),device='cuda'))
        mask=torch.arange(bank.bank_len,device='cuda')[None]<bank._count[:,None]
        age=args.frames-bank.timestamps
        streams=[('perceptual',torch.zeros_like(p[:,args.frames]),bank.perceptual_bank,
                  bank.perceptual_retrieval,bank.perceptual_gate),
                 ('state',torch.zeros_like(s[:,args.frames]),bank.state_bank,
                  bank.state_retrieval,bank.state_gate)]
        for name,query,values,retrieval,gate in streams:
            if name=='perceptual' and bank.patch_dim is not None:
                query=query.reshape(len(indices),-1,bank.patch_dim)
            outputs=[]
            hook=retrieval.retrieval_attn.register_forward_hook(lambda m,a,out:outputs.append(out[0].detach()))
            correct=bank._retrieve(retrieval,query,values,mask,age)
            wrong=bank._retrieve(retrieval,query,values.roll(1,0),mask,age)
            hook.remove()
            def rms(x):return x.float().square().mean().sqrt().item()
            report['streams'][name]=dict(bank_rms=rms(values),bank_donor_delta_rms=rms(values-values.roll(1,0)),
                attention_donor_delta_rms=None if not outputs else rms(outputs[0]-outputs[1]),retrieved_rms=rms(correct),
                retrieved_donor_delta_rms=rms(correct-wrong),
                gated_donor_delta_rms=rms(gate(query,correct)-gate(query,wrong)))
    atomic_json(args.output,report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
