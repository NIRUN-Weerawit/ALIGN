"""Matched-noise held-out switch diagnostics for the compression pilots."""
import sys, inspect, json, gc
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/whinnoy/ALIGN')
sys.path.insert(0,str(ROOT))
import models.align_intention as align
from data.align_dataset import ALIGNDataset
from scripts.run_intention_ablation import CachedEpisodes, seed_everything
from training.plain_history_training import SparseHistoryEpisodes, collate_sparse_histories, history_conditions
class NoVision(torch.nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.unused=torch.nn.Parameter(torch.empty(0),requires_grad=False)
align.VisionEncoder=NoVision
torch.set_num_threads(2)
torch.backends.cuda.enable_cudnn_sdp(False)
torch.cuda.set_per_process_memory_fraction(.3)
class ProbeEpisodes(CachedEpisodes):
    def __getitem__(self,index):
        item=super().__getitem__(index)
        a=item['actions_segment'][:,6]>.5
        switches=np.flatnonzero(a[1:]!=a[:-1])+1
        switches=switches[switches<=len(a)-self.chunk_size]
        item['loss_anchor_mask'][switches]=True
        return item
rows=[]
for name in ['d8_uniform','d8_transition','d16_transition','d32_transition']:
    run=ROOT/'checkpoints/compression_gripper_2026_10_14'/name/'no_intent_no_memory'
    cfg=json.loads((run/'config.json').read_text())
    checkpoint=torch.load(run/'training_last.pt',map_location='cpu',weights_only=False,mmap=True)
    accepted=inspect.signature(align.ALIGNIntentionModel.__init__).parameters
    model=align.ALIGNIntentionModel(**{k:v for k,v in cfg.items() if k in accepted})
    model._build_head_and_bank(cfg['pool_out_dim'])
    incompatible=model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    assert not incompatible.unexpected_keys and incompatible.missing_keys == ['vision_encoder.unused']
    model=model.cuda().eval()
    dataset=ALIGNDataset(str(ROOT/'data/libero_goal.h5'),mode='head',cameras=cfg['cameras'],traj_window=cfg['segment_length'],dinov2_path=str(ROOT/'data/libero_goal.dinov2'))
    eps=[dataset._episode_keys.index(e) for e in cfg['val_episodes']]
    parent=ProbeEpisodes(dataset,eps,cfg['segment_length'],1042,False,'episode',16,8)
    sparse=SparseHistoryEpisodes(parent,1)
    counts={k:dict(n=0,correct=0,pos_sum=0.,rot_sum=0.,false_switch=0) for k in ['switch','stable']}
    with torch.no_grad():
        for i in range(len(eps)):
            batch=collate_sparse_histories([sparse[i]])
            cond=history_conditions(model,batch,torch.device('cuda'))
            seed_everything(20042+i)
            with torch.autocast('cuda',dtype=torch.bfloat16): pred=model.intention_head.sample(cond).float().cpu()
            target=batch['targets']
            previous=batch['states'][batch['history_indices'][:,-1],6]>.5
            desired=target[:,0,6]>.5
            valid=batch["anchor_timesteps"] > 0
            switch=(desired!=previous) & valid
            stable=(desired==previous) & valid
            for group,mask in [('switch',switch),('stable',stable)]:
                c=counts[group]; c['n']+=int(mask.sum())
                c['correct']+=int(((pred[:,0,6]>.5)==desired)[mask].sum())
                c['false_switch']+=int(((pred[:,0,6]>.5)!=previous)[mask].sum())
                c['pos_sum']+=float((pred[mask,:,:3]-target[mask,:,:3]).square().mean((1,2)).sum())
                c['rot_sum']+=float((pred[mask,:,3:6]-target[mask,:,3:6]).square().mean((1,2)).sum())
    for c in counts.values():
        c['accuracy']=c['correct']/c['n']; c['pos_mse']=c['pos_sum']/c['n']; c['rot_mse']=c['rot_sum']/c['n']
    row=dict(name=name,epoch=checkpoint['epoch'],counts=counts)
    rows.append(row);print(json.dumps(row),flush=True)
    (ROOT/'reports/compression_gripper_2026_10_14/transition_probe.json').write_text(json.dumps(rows,indent=2))
    dataset.close();del model,checkpoint;gc.collect();torch.cuda.empty_cache()
