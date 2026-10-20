"""Offline development assessment for integrated, split-bound action policies.

This never selects a new checkpoint or opens confirmation. Use after the bounded
integration pilot or after full training; results are not closed-loop success.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from data.task_context_dataset import file_hash
from eval.eval_intention import load_intention_model
from training.task_context_policy_evaluation import evaluate_fixed_split
from scripts.run_intention_ablation import seed_everything, source_provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--max-batches',type=int,default=0)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(2)
    device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.24)
    rows={};reference=None
    for path in a.checkpoints:
        model,cfg=load_intention_model(path,device)
        split=cfg.get('task_context_split')
        if not split:raise ValueError('Policy is missing its pretraining-safe episode split')
        protocol=(split,cfg['seed'],cfg['chunk_size'],cfg.get('supervision_points',16),cfg['head_type'])
        if reference is None:reference=protocol
        elif reference!=protocol:raise ValueError('Policy arms do not share split/sampling protocol')
        seed_everything(cfg['seed']+10000)
        start=time.perf_counter()
        result=evaluate_fixed_split(model,cfg,[a.data],a.batch_size,a.max_batches,device)
        name=Path(path).parent.name
        if name in rows:raise ValueError('Duplicate policy arm name')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        rows[name]=dict(checkpoint=str(path),sha256=file_hash(path),selected_epoch=checkpoint.get('epoch'),
            source_training_limits=dict(epochs=cfg.get('epochs'),max_steps=cfg.get('max_steps'),
                selection_max_val_batches=cfg.get('max_val_batches')),
            elapsed_s=time.perf_counter()-start,**result)
        print(json.dumps(dict(arm=name,**result)),flush=True)
        del checkpoint,model
        if device.type=='cuda':torch.cuda.empty_cache()
    report=dict(split='development',confirmation_opened=False,seed=reference[1]+10000,
        episodes_in_split=len(reference[0]['development_episodes']),max_batches=a.max_batches,
        arms=rows,source_provenance=source_provenance(),
        contract='Fixed selected policies; matched noise seed and saved episode split. Offline sampled-action errors; not robot success or validated handover benefit.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
