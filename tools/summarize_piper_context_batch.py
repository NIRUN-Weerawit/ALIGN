"""Compare a complete immutable Piper evaluation batch, with assistance separated."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def paired(a,b):
    a={e['episode']:e for e in a['episodes']}
    b={e['episode']:e for e in b['episodes']}
    assert a.keys()==b.keys()
    wins=[k for k in a if b[k]['success'] and not a[k]['success']]
    losses=[k for k in a if a[k]['success'] and not b[k]['success']]
    n=len(wins)+len(losses)
    p=min(1.,2*sum(math.comb(n,i) for i in range(min(len(wins),len(losses))+1))/2**n)
    return dict(improved=wins,regressed=losses,exact_mcnemar_p=p)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    args=parser.parse_args();root=args.batch.resolve()
    manifest=json.loads((root/'batch_manifest.json').read_text())
    models={(m['model'],m['variant']):m for m in manifest['models']}
    runs={};source=[]
    expected=sorted(manifest['episodes'])
    comparison_protocol=None
    for job in manifest['jobs']:
        path=Path(job['out_dir'])/'result.json'
        if not path.exists():raise ValueError('Incomplete batch: '+job['id'])
        d=json.loads(path.read_text());pr=d['evaluation_protocol'];a=d['evaluation_args']
        assert d['evaluation_complete'] and len(d['episodes'])==44
        assert sorted(e['episode'] for e in d['episodes'])==expected
        assert pr['checkpoint_sha256']==job['checkpoint_sha256']
        assert a['switch_at']==job['switch_at']
        if not job['use_text']:assert a['task_context_mode']==job['mode']
        assert a['seed']==42 and a['max_steps']==300 and a['action_horizon']==1
        assert pr['stop_on_task_success'] and not pr['replay_filters_policy_episodes']
        protocol=(pr['evaluator_sha256'],pr['protocol_sha256'])
        if comparison_protocol is None:comparison_protocol=protocol
        assert protocol==comparison_protocol
        runs[job['id']]=d
        eligible=[e for e in d['episodes'] if e['policy_eligible']]
        count=sum(bool(e['success_after_handover']) for e in eligible)
        source.append(dict(id=job['id'],successes=count,episodes=len(eligible),
            expert_completed_before_handover=d['expert_completed_episodes'],
            result=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            per_task=d['per_task_success']))
    metrics={r['id']:r for r in source}
    def cell(key):
        r=metrics[key];n=r['episodes'];s=r['successes']
        return f'{s}/{n} ({100*s/n:.1f}%)' if n else 'N/A'
    contexts=[(key,m) for key,m in models.items() if not m['use_text']]
    texts=[(key,m) for key,m in models.items() if m['use_text']]
    labels={key:f'C{i+1}' for i,(key,m) in enumerate(contexts)}
    labels.update({key:f'T{i+1}' for i,(key,m) in enumerate(texts)})
    best_context=max((r for r in source if not models[tuple(r['id'].split('/')[:2])]['use_text']),key=lambda r:r['successes']/max(r['episodes'],1))
    best_text=max((r for r in source if models[tuple(r['id'].split('/')[:2])]['use_text']),key=lambda r:r['successes']/max(r['episodes'],1))
    reference=next(iter(runs.values()))
    expert_successes=sum(e['replay_check']['task_success'] for e in reference['episodes'])
    contrasts={}
    state_audit=[]
    for (family,variant),m in contexts:
        prefix=f'{family}/{variant}/'
        contrasts[prefix+'handover_vs_dynamic']=paired(runs[prefix+'dynamic'],runs[prefix+'handover'])
        contrasts[prefix+'freeze_vs_handover']=paired(runs[prefix+'handover'],runs[prefix+'freeze'])
        for key in expected:
            states=[]
            for mode in ['dynamic','freeze','handover']:
                job=next(j for j in manifest['jobs'] if j['id']==prefix+mode)
                with np.load(Path(job['out_dir'])/(key+'_rollout.npz'),allow_pickle=False) as trace:
                    states.append(trace['initial_sim_state'].copy())
            state_audit.append(dict(model=family,variant=variant,episode=key,
                initial_states_identical=all(np.array_equal(states[0],s) for s in states[1:])))
    lines=['# Piper evaluation matrix — 2026-10-26','',
        f"**Best context condition: {best_context['id']} — {cell(best_context['id'])}, with a 40% expert prefix.**",
        f"**Best autonomous text condition: {best_text['id']} — {cell(best_text['id'])}, from 0%.**",'',
        f"All {len(source)} runs completed, with immutable checkpoint and source snapshots. Each requests the same 44 saved policy-validation episodes, "
        'seed 42, action horizon 1, a 300-step cap, and stopping at first success. Expert replay failures are included.','',
        f"Expert replay task completion: **{expert_successes}/44 ({100*expert_successes/44:.1f}%)**. "
        f"Pose-replay qualification passes {reference['replay_pass_count']}/44; these are separate criteria. Expert replay is not a success ceiling.",'',
        '## Context policies: 40% expert prefix','',
        '| ID | Model | Variant | Dynamic | Highest-score freeze | Fixed handover |',
        '|---|---|---|---:|---:|---:|']
    for (family,variant),m in contexts:
        prefix=f'{family}/{variant}/'
        lines.append(f'| {labels[(family,variant)]} | {family} | {variant} | '+ ' | '.join(cell(prefix+mode) for mode in ['dynamic','freeze','handover'])+' |')
    lines+=['','Freeze selects the highest decoder score observed so far, starting at the first observation. '
        'Handover fixes the task snapshot at authority transfer. Execution context remains live. '
        'The frozen task decoder is fitted on 384 training episodes and 3,456 predeclared prefix queries; '
        'no evaluation observations/labels are used for fitting or model selection. Scores are uncalibrated.','',
        '## Text policies: autonomous from 0%','',
        '| ID | Model | Variant | Completion |','|---|---|---|---:|']
    for (family,variant),m in texts:
        lines.append(f'| {labels[(family,variant)]} | {family} | {variant} | {cell(f"{family}/{variant}/text")} |')
    lines+=['','Text models receive task instructions and no expert action prefix. Their scores are a different assistance condition '
        'from the 40%-prefix context policies, so a cross-table difference is not a controlled architecture effect.','',
        '## Checkpoint configuration','',
        '| Model | Variant | Saved epoch | cp dim | Head width | History | Chunk | LR | Configured epochs |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for (family,variant),m in models.items():
        c=m['config'];lines.append(f"| {family} | {variant} | {m['epoch']} | {c['compressed_dim']} | {c['head_d_model']} | "
            f"{c['history_size']} | {c['chunk_size']} | {c['lr']} | {c['epochs']} |")
    lines+=['','Saved epoch comes from the evaluated checkpoint, and may use zero-based indexing. '
        'Configured training duration is not the number of epochs completed. All source configs are saved beside the snapshotted weights.','',
        '## Task recognition at handover','',
        '| Context model | Mode | Live inferred identity | Actor-used identity |',
        '|---|---|---:|---:|']
    for (family,variant),m in contexts:
        for mode in ['dynamic','freeze','handover']:
            d=runs[f'{family}/{variant}/{mode}']
            rows=[e for e in d['episodes'] if e.get('task_estimation',{}).get('checkpoints',{}).get('handover')]
            counts=[sum(e['task_estimation']['checkpoints']['handover'][role]['correct'] for e in rows) for role in ['inferred','policy']]
            lines.append(f'| {labels[(family,variant)]} | {mode} | {counts[0]}/{len(rows)} | {counts[1]}/{len(rows)} |')
    lines+=['','## Paired context-mode outcomes','',
        '| Model and comparison (second vs first) | Gained episodes | Lost episodes | Exact paired p |',
        '|---|---:|---:|---:|']
    for key,c in contrasts.items():lines.append(f"| {key} | {len(c['improved'])} | {len(c['regressed'])} | {c['exact_mcnemar_p']:.3f} |")
    lines+=['','These are single-seed exploratory comparisons across multiple models. '
        'Simulator/inference variation can change outcomes even with matched resets; small differences need repeated seeds. '
        f"Initial states match across all three modes in {sum(r['initial_states_identical'] for r in state_audit)}/{len(state_audit)} model/episode comparisons.",'',
        '## Results by task','']
    tasks=sorted(next(iter(runs.values()))['per_task_success'])
    for mode,group in [('handover',contexts),('dynamic',contexts),('freeze',contexts),('text',texts)]:
        lines += [f'### {mode}','', '| Task | Expert replay | '+' | '.join(labels[k] for k,m in group)+' |',
            '|---|---:|'+'---:|'*len(group)]
        for task in tasks:
            first=runs[f'{group[0][0][0]}/{group[0][0][1]}/{mode}']
            rr=[e for e in first['episodes'] if e['task_name']==task]
            n=len(rr);s=sum(e['replay_check']['task_success'] for e in rr)
            cells=[f'{s}/{n} ({100*s/n:.0f}%)']
            for (family,variant),m in group:
                rr=[e for e in runs[f'{family}/{variant}/{mode}']['episodes'] if e['task_name']==task and e['policy_eligible']]
                n=len(rr);s=sum(e['success_after_handover'] for e in rr)
                cells.append(f'{s}/{n} ({100*s/n:.0f}%)' if n else 'N/A')
            lines.append('| '+task+' | '+' | '.join(cells)+' |')
        lines.append('')
    lines+=['## Provenance and scope','',
        'The 44 episodes are held out from the 384 policy-training episodes, but are used for policy validation and checkpoint selection. '
        'The context-pretraining manifest calls them confirmation; this batch evaluates them explicitly. '
        'They are known-task episodes, not an unseen-task generalization benchmark. '
        'No policy weights are updated by evaluation or decoder fitting.','',
        '[All checkpoint configs and original paths](checkpoints.json). '
        '[Exact commands and batch manifest](batch_manifest.json). '
        '[Result hashes, all mode/task counts, and paired episode changes](comparison.json).']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    (root/'comparison.json').write_text(json.dumps(dict(runs=source,paired_mode_comparisons=contrasts,
        all_protocols_match=True,all_checkpoints_match=True,initial_state_audit=state_audit,episode_ids=expected,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    (root/'comparison_source.py').write_bytes(Path(__file__).read_bytes())
    print(root/'comparison.md')


if __name__=='__main__':main()
