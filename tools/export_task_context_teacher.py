#!/usr/bin/env python3
"""Export immutable privileged task-token targets after the confirmation gate."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from data.task_context_dataset import TaskContextEpisodes,file_hash
from training.train_task_context_teacher import load_checkpoint,make_loader,to_device


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--targets',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--gate-report',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError('Use a new export directory')
    report=json.loads(Path(args.gate_report).read_text())
    if report.get('split')!='confirmation' or not report.get('gate',{}).get('passed'):
        raise ValueError('A passing confirmation report is required')
    digest=file_hash(args.checkpoint)
    if digest!=report['checkpoints']['full']['sha256']:
        raise ValueError('Selected teacher changed after confirmation')
    torch.set_num_threads(2);device=torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(.24)
        torch.backends.cuda.enable_cudnn_sdp(False)
    model,checkpoint=load_checkpoint(args.checkpoint,device)
    model.requires_grad_(False)
    output.mkdir(parents=True)
    exported={}
    for split in ['train','development','confirmation']:
        data=TaskContextEpisodes(args.targets,split,anchors=1,verify=True)
        if data.manifest['manifest_sha256']!=report['manifest_sha256']:
            raise ValueError('Teacher target manifest mismatch')
        keys=[];values=[]
        for batch in make_loader(data,8):
            keys.extend(batch['keys']);batch=to_device(batch,device)
            tokens,_=model.encode(batch['features'],batch['states'],batch['times'],batch['lengths'],batch['anchors'],'full')
            values.append(tokens[:,0].cpu().numpy())
        path=output/f'{split}.npz'
        np.savez(path,keys=np.asarray(keys),task_tokens=np.concatenate(values))
        exported[split]=dict(episodes=len(keys),sha256=file_hash(path))
    manifest=dict(schema='align-privileged-task-tokens-v1',teacher_sha256=digest,
        target_manifest_sha256=report['manifest_sha256'],gate_report_sha256=file_hash(args.gate_report),
        teacher_epoch=checkpoint['epoch'],shape=[4,128],splits=exported,
        contract='Whole-episode privileged supervision only. Not a causal deployment input; preserve split boundaries.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
