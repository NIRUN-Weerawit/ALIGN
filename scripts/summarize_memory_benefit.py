#!/usr/bin/env python3
"""Paired history-content error benefit, excluding anchors with empty banks.

Probe rows aggregate episodes within each batch, so resample batch clusters
rather than pretending that individual predictions are independent trials.
Positive benefit means the correct history has lower error than shuffled history.
"""
import argparse,json
from pathlib import Path
import numpy as np


def summarize(rows,draws=20000,seed=42):
    indexed={}
    for row in rows:
        counts=row.get('bank_count',0)
        if not isinstance(counts,list) or not counts or min(counts)<=0:continue
        key=(row['batch'],row['t'],row['intervention'])
        if key in indexed:raise ValueError('Duplicate probe row')
        indexed[key]=row
    pairs=[('full_observation','baseline','memory_shuffle'),
           ('last_camera','last_camera_correct','last_camera_shuffle'),
           ('all_visual','all_visual_correct','all_visual_shuffle'),
           ('all_observation','all_observation_correct','all_observation_shuffle')]
    result={}
    for case,correct,wrong in pairs:
        anchors=sorted({(b,t) for b,t,arm in indexed if arm in [correct,wrong]})
        if not anchors:continue
        clusters={}
        for b,t in anchors:
            if (b,t,correct) not in indexed or (b,t,wrong) not in indexed:
                raise ValueError(f'Unpaired {case} anchor: batch {b}, t={t}')
            clusters.setdefault(b,[]).append((indexed[b,t,correct],indexed[b,t,wrong]))
        r=dict(batch_clusters=len(clusters),nonempty_anchors=len(anchors),metrics={})
        rng=np.random.default_rng(seed)
        samples=rng.integers(0,len(clusters),size=(draws,len(clusters)))
        for metric in ['position_mse','rotation_mse','gripper_mse']:
            correct_error=np.array([np.mean([a[metric] for a,z in pairs]) for pairs in clusters.values()])
            wrong_error=np.array([np.mean([z[metric] for a,z in pairs]) for pairs in clusters.values()])
            delta=wrong_error-correct_error
            ci=np.quantile(delta[samples].mean(1),[.025,.975]).tolist()
            r['metrics'][metric]=dict(correct=float(correct_error.mean()),shuffled=float(wrong_error.mean()),
                benefit=float(delta.mean()),relative_benefit=float(delta.mean()/wrong_error.mean()) if wrong_error.mean()!=0 else None,
                bootstrap_95_percent_interval=ci)
        result[case]=r
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True,help='Per-memory-variant JSON with results and rows')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report=dict(source=str(args.probe.resolve()),
                protocol='Correct vs cross-task shuffled histories, matched noise; empty-bank anchors excluded; 20,000 paired batch-cluster bootstrap draws, seed 42.',
                caveat='Rows aggregate episodes within batches. Intervals describe prediction errors, not simulator success. Statistical detectability alone does not establish practical benefit.',
                cases=summarize(json.loads(args.probe.read_text())['rows']))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
