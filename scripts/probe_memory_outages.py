#!/usr/bin/env python3
"""Paired fixed-weight memory-field intervention during sustained camera outages.

Current state remains available. Past expert observations prime the bank;
wrong histories come from distinct tasks. This is an offline diagnostic.
"""
import argparse, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data.align_dataset import ALIGNDataset
from scripts.run_intention_ablation import CachedEpisodes, atomic_json, seed_everything, collate_segments
from scripts.probe_condition_dependence import encode, load_model, gripper_class_metrics
from scripts.summarize_memory_benefit import summarize, summarize_gripper_controls


def field_mask_effects(cases,durations):
    effects={}
    for duration in durations:
        paired=[]
        for masked,arm in [(True,'baseline'),(False,'memory_bypass')]:
            paired.extend(dict(row,intervention=arm) for row in cases[f'field_masks_{masked}_outage_{duration}']['rows']
                          if row['intervention']=='baseline')
        error_rows=[dict(row,intervention='memory_shuffle' if row['intervention']=='memory_bypass' else row['intervention']) for row in paired]
        error=summarize(error_rows)['full_observation']['metrics']['position_mse']
        effects[duration]=dict(masked_position_mse=error['correct'],unmasked_position_mse=error['shuffled'],
            relative_error_reduction=error['relative_benefit'],absolute_error_reduction_interval=error['bootstrap_95_percent_interval'],
            gripper_recall=summarize_gripper_controls(paired)['normal_vs_bypass']['metrics'])
    return effects


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--outage-frames',type=int,nargs='+',default=[8,16,32])
    args=parser.parse_args()
    if args.batch_size<2 or min(args.outage_frames)<1:parser.error('Need paired tasks and positive outage durations')
    manifest=json.loads((args.run/'manifest.json').read_text())
    if manifest["head_type"]!="diffusion":parser.error("This probe requires a diffusion checkpoint")
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    model,epoch=load_model(args.checkpoint,manifest['cameras'])
    if not model.use_memory_bank or model.memory_mode!='episodic' or model.memory_write_fused or model.use_intent_tokens or model.history_size!=1:
        parser.error('Requires no-intent, history-1 episodic raw-write memory')
    dataset=ALIGNDataset(manifest['data'],mode='head',cameras=manifest['cameras'],
        traj_window=manifest['segment_length'],dinov2_path=manifest['cache'])
    groups=defaultdict(list)
    for ep in manifest['val_episodes']:
        groups[str(dataset._h5[f'{ep}/texts'][()])].append(dataset._episode_keys.index(ep))
    groups=list(groups.values())
    episodes=[group[k] for k in range(max(map(len,groups))) for group in groups if k<len(group)]
    for start in range(0,len(episodes),args.batch_size):
        tasks=[str(dataset._h5[f'{dataset._episode_keys[e]}/texts'][()]) for e in episodes[start:start+args.batch_size]]
        if len(tasks)<2 or any(v==tasks[(k-1)%len(tasks)] for k,v in enumerate(tasks)):
            parser.error('Batch pairs must have distinct tasks')
    cached=CachedEpisodes(dataset,episodes,manifest['segment_length'],manifest['seed']+1000,
        False,'episode',manifest.get('supervision_points',16),manifest['chunk_size'])
    loader=DataLoader(cached,batch_size=args.batch_size,collate_fn=collate_segments,num_workers=0)
    rows=defaultdict(list)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for batch_index,batch in enumerate(loader):
            visual,state,_=encode(model,batch);B=visual.shape[0]
            limit=int(min(batch['segment_len']))-model.chunk_size+1
            start=limit//2
            if start+max(args.outage_frames)>limit:parser.error('Episode common prefix is too short for the outage')
            donor=torch.arange(B,device='cuda').roll(1)
            for masked in [False,True]:
                model.memory_module.mask_missing_fields=masked
                for arm in ['baseline','memory_shuffle']:
                    model.memory_module.reset(B,torch.device('cuda'))
                    order=donor if arm=='memory_shuffle' else torch.arange(B,device='cuda')
                    for t in range(start):
                        model.memory_module.observe_only(visual[order,t],state[order,t],timestamp=torch.full((B,),float(t),device='cuda'))
                    for duration in range(1,max(args.outage_frames)+1):
                        t=start+duration-1;p=torch.zeros_like(visual[:,t:t+1]);s=state[:,t:t+1]
                        cond=model.condition_actions(p,s,timestamp=torch.full((B,),float(t),device='cuda'))
                        if duration not in args.outage_frames:continue
                        targets=batch['actions_segment'][:,t:t+model.chunk_size].cuda().float()
                        conditions={arm:cond}
                        if arm=='baseline':conditions['memory_bypass']=model.prepare_head_inputs(p,s)
                        for name,value in conditions.items():
                            seed_everything(manifest['seed']+20000+batch_index*100+t)
                            action=model.intention_head.sample(model.intention_head(*value)).float()
                            pred=action[:,:,6]>.5;true=targets[:,:,6]>.5
                            counts={key:int(bits.sum()) for key,bits in [('tp',pred & true),('tn',~pred & ~true),('fp',pred & ~true),('fn',~pred & true)]}
                            row=dict(batch=batch_index,t=t,intervention=name,bank_count=model.memory_module._count.tolist(),
                                gripper_counts=counts,**{group+'_mse':float((action[:,:,sl]-targets[:,:,sl]).square().mean())
                                for group,sl in [('position',slice(0,3)),('rotation',slice(3,6)),('gripper',slice(6,7))]})
                            rows[masked,duration].append(row)
            print(f'Outage probe batch {batch_index+1}/{len(loader)}',flush=True)
    cases={}
    for (masked,duration),records in rows.items():
        rates={}
        for arm in ['baseline','memory_shuffle','memory_bypass']:
            counts={k:sum(r['gripper_counts'][k] for r in records if r['intervention']==arm) for k in ['tp','tn','fp','fn']}
            rates[arm]=gripper_class_metrics(counts)
        cases[f'field_masks_{masked}_outage_{duration}']=dict(history_benefit=summarize(records)["full_observation"],
            gripper_controls=summarize_gripper_controls(records),gripper_classes=rates,rows=records)
    report=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),epoch=epoch,
        protocol='Fixed trained weights, field-mask override off/on; middle-of-episode camera outages, current expert state available; distinct-task donor histories; matched sampling noise; batch-cluster intervals.',
        held_out_episodes=len(episodes),saved_field_masks=model.memory_field_masks,
        field_mask_effects=field_mask_effects(cases,args.outage_frames),
        field_mask_effect_protocol='Same correct histories; masks ON minus OFF; gripper normal means masked and comparison means unmasked.',
        caveat='Offline architecture intervention, not training or simulator success. Class 0 is closed and class 1 open for LIBERO.',cases=cases)
    args.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.output,report)
    print(f'Saved {args.output}',flush=True)


if __name__=='__main__':main()
