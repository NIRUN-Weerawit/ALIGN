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
        key=(row['batch'],row['t'],row['intervention'])
        if key in indexed:raise ValueError('Duplicate probe row')
        indexed[key]=row
    pairs=[('previous_observation_control','previous_observation_correct','previous_observation_shuffle'),
           ('full_observation','baseline','memory_shuffle'),
           ('last_camera','last_camera_correct','last_camera_shuffle'),
           ('all_visual','all_visual_correct','all_visual_shuffle'),
           ('all_observation','all_observation_correct','all_observation_shuffle')]
    result={}
    for case,correct,wrong in pairs:
        def available(row):
            if case=='previous_observation_control':return row.get('history_available',row['t']>0)
            counts=row.get('bank_count',0)
            return isinstance(counts,list) and bool(counts) and min(counts)>0
        anchors=sorted({(b,t) for (b,t,arm),row in indexed.items()
                        if arm in [correct,wrong] and available(row)})
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


def summarize_gripper_controls(rows,draws=20000,seed=42):
    """Resample paired batches, recomputing class recall from summed counts."""
    indexed={(r['batch'],r['t'],r['intervention']):r for r in rows}
    if len(indexed)!=len(rows):raise ValueError('Duplicate probe row')
    result={}
    for case,left,right in [('normal_vs_bypass','baseline','memory_bypass'),
                            ('normal_vs_wrong_history','baseline','memory_shuffle')]:
        clusters={}
        for (b,t,arm),row in indexed.items():
            if arm!=left or 'gripper_counts' not in row:continue
            other=indexed.get((b,t,right))
            if other is None or 'gripper_counts' not in other:continue
            clusters.setdefault(b,[]).append((row,other))
        if not clusters:continue
        counts=np.array([[[sum(a['gripper_counts'][k] for a,z in pairs) for k in ['tp','tn','fp','fn']],
                          [sum(z['gripper_counts'][k] for a,z in pairs) for k in ['tp','tn','fp','fn']]]
                         for pairs in clusters.values()],dtype=float)
        samples=np.random.default_rng(seed).integers(0,len(clusters),size=(draws,len(clusters)))
        total=counts.sum(0)
        sampled=counts[samples].sum(1)
        metrics={}
        for label,num,den in [('label_1_recall',0,[0,3]),('label_0_recall',1,[1,2])]:
            totals=total[:,den].sum(-1)
            if np.any(totals==0):continue
            point=total[:,num]/totals
            denominators=sampled[:,:,den].sum(-1)
            valid=(denominators>0).all(1)
            rates=sampled[valid,:,num]/denominators[valid]
            metrics[label]=dict(normal=float(point[0]),comparison=float(point[1]),
                recall_difference=float(point[0]-point[1]),
                bootstrap_95_percent_interval=np.quantile(rates[:,0]-rates[:,1],[.025,.975]).tolist(),
                valid_bootstrap_draws=int(valid.sum()))
        result[case]=dict(batch_clusters=len(clusters),anchors=sum(map(len,clusters.values())),metrics=metrics)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True,help='Per-memory-variant JSON with results and rows')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=json.loads(args.probe.read_text())['rows']
    report=dict(source=str(args.probe.resolve()),
                protocol='Correct vs cross-task shuffled histories, matched noise; empty-bank anchors excluded (direct prior-frame controls require an actual past frame); 20,000 paired batch-cluster bootstrap draws, seed 42.',
                caveat='Rows aggregate episodes within batches. Intervals describe prediction errors, not simulator success. Statistical detectability alone does not establish practical benefit.',
                cases=summarize(rows),gripper_controls=summarize_gripper_controls(rows))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
