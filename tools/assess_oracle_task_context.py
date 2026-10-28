"""Matched-noise, fixed-observation oracle diagnostic; no policy training."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from data.task_context_dataset import file_hash
from eval.task_context_oracle import TaskContextOracleBank
from eval.task_estimation import TaskLabelEstimator,canonical_task_name
from tools.experiment_context_fusion import make_head,evaluate
from tools.assess_context_fusion_experiment import mean_rows,paired


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--oracle-bank',type=Path,required=True)
    p.add_argument('--task-estimator',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--noise-seeds',type=int,nargs='+',default=[20261022,20261023,20261024])
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2)
    device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.15)
    torch.backends.cudnn.enabled=False;torch.backends.cuda.matmul.allow_tf32=False
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    features=torch.load(a.features,map_location='cpu',weights_only=True)
    checkpoint_hash=file_hash(a.checkpoint)
    if features['metadata']['checkpoint_sha256']!=checkpoint_hash:raise ValueError('Feature/checkpoint binding differs')
    data=features['development']
    estimator=TaskLabelEstimator(a.task_estimator)
    bank=TaskContextOracleBank(a.oracle_bank,estimator,saved['config'],checkpoint_hash)
    if set(data['episodes'])!=set(bank.entries):raise ValueError('Expected all saved development episodes')
    current=data['frozen_task40']
    actual=np.asarray([estimator.name_to_group[canonical_task_name(t)] for t in data['tasks']])
    correct=estimator.probabilities(current).argmax(1)==actual
    oracle=torch.stack([torch.from_numpy(bank.entries[k]['tokens']) for k in data['episodes']])
    repaired=current.clone();repaired[~correct]=oracle[~correct]
    head=make_head(saved,saved['config'].get('task_context_fusion','additive'),128,device).eval().requires_grad_(False)
    runs={}
    for name,tokens in [('baseline',current),('repair',repaired),('episode',oracle),('no_task',current)]:
        changed=dict(data,frozen_task40=tokens)
        draws=[evaluate(head,changed,'freeze40',device,seed,32,'no_task' if name=='no_task' else 'normal') for seed in a.noise_seeds]
        rows=mean_rows(draws)
        runs[name]=dict(rows=rows,mean={f:float(np.mean([r[f] for r in rows])) for f in
            ['action_mse','position_mse','first_position_mse','rotation_mse','gripper_error']},
            decoded_correct_anchors=None if name=='no_task' else int((estimator.probabilities(tokens).argmax(1)==actual).sum()),
            draws=draws)
    subsets={'all':np.ones(len(correct),dtype=bool),'baseline_task_correct':correct,'baseline_task_wrong':~correct}
    comparisons={}
    for name in ['repair','episode','no_task']:
        comparisons[name]={}
        for subset,mask in subsets.items():
            comparisons[name][subset]=paired([r for r,keep in zip(runs[name]['rows'],mask) if keep],
                                            [r for r,keep in zip(runs['baseline']['rows'],mask) if keep])
    protocol=dict(privileged=True,deployable=False,ground_truth_used=True,weights_updated=False,
        confirmation_opened=False,episodes=sorted(set(data['episodes'])),noise_seeds=a.noise_seeds,
        context_mode='task fixed at40%, execution live',checkpoint_sha256=checkpoint_hash,
        features_sha256=file_hash(a.features),oracle_bank_sha256=file_hash(a.oracle_bank),
        estimator_sha256=file_hash(a.task_estimator),source_sha256=file_hash(__file__),
        definition=bank.metadata['definition'],metric='Expert-state sampled-action error, not simulator completion')
    result=dict(protocol=protocol,runs=runs,comparisons=comparisons,complete=True)
    a.output.mkdir(parents=True)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Oracle task context: fixed-observation action diagnostic','',
        '**Privileged diagnostic. GT labels and potentially future demonstration frames select task snapshots. No weights changed.**','',
        'All 44 development episodes, four expert anchors per episode, three matched noise draws; live execution context. '
        'Correct decoder identity does not guarantee complete semantic task correctness.','',
        '| Context | Correct task identity (anchors) | First-action position MSE | Chunk position MSE | Action MSE | Gripper error |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in runs.items():
        m=r['mean'];identity='N/A' if r['decoded_correct_anchors'] is None else f"{r['decoded_correct_anchors']}/{len(correct)}"
        lines.append(f"| {name} | {identity} | {m['first_position_mse']:.6f} | {m['position_mse']:.6f} | {m['action_mse']:.6f} | {m['gripper_error']:.1%} |")
    lines+=['','Positive paired deltas mean worse action imitation than inferred baseline.','',
        '| Arm | Subset | First-action position ΔMSE [95% episode bootstrap] |', '|---|---|---:|']
    for name,subsets in comparisons.items():
        for subset,values in subsets.items():
            r=values['first_position_mse'];lo,hi=r['episode_bootstrap_95pct']
            lines.append(f"| {name} | {subset} | {r['mean']:+.6f} [{lo:+.6f}, {hi:+.6f}] |")
    (a.output/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:12]),flush=True)


if __name__=='__main__':main()
