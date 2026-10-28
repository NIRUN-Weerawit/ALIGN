"""Summarize matched normal/GT-donor results for all non-text Piper checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.summarize_task_token_counterfactuals import clustered_interval


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(root):
    protocol=json.loads((root/'protocol.json').read_text())
    seeds,keys=protocol['seeds'],protocol['episodes']
    reference=Path(protocol['reference_study'])
    donor_plan=json.loads((root/'donor_plan.json').read_text())
    rows=[];sources=[]
    for model in protocol['models']:
        base=(Path(model['evaluation_runs']) if model.get('evaluation_runs') else
              reference/'runs' if model['reused'] else root/'runs'/model['model']/model['variant'])
        outcomes={};eligible={};correct=[];per_seed={};pairs=[];tasks={};per_task={}
        expert_success=0
        for arm in protocol['arms']:
            outcomes[arm]=np.zeros((len(seeds),len(keys)),dtype=bool)
            eligible[arm]=np.zeros_like(outcomes[arm])
        exact=np.zeros_like(outcomes['normal'])
        for si,seed in enumerate(seeds):
            reports={};directories={}
            for arm in protocol['arms']:
                directory=base/f'seed_{seed}'/arm;directories[arm]=directory
                path=directory/'result.json';data=json.loads(path.read_text())
                assert data['evaluation_complete']
                assert data['evaluation_protocol']['checkpoint_sha256']==model['checkpoint_sha256']
                assert sorted(r['episode'] for r in data['episodes'])==keys
                reports[arm]={r['episode']:r for r in data['episodes']}
                sources.append(dict(model=model['model'],variant=model['variant'],seed=seed,arm=arm,
                                    path=str(path),sha256=digest(path),reused=model['reused']))
            cohort=[]
            for ki,key in enumerate(keys):
                for arm in protocol['arms']:
                    r=reports[arm][key]
                    outcomes[arm][si,ki]=r['success_after_handover']
                    eligible[arm][si,ki]=r['policy_eligible']
                task=reports['normal'][key]['task_name'];tasks[key]=task
                audit={};heads={};traces={}
                for arm in protocol['arms']:
                    directory=directories[arm]/'counterfactual'
                    audit[arm]=json.loads((directory/(key+'_audit.json')).read_text())
                    heads[arm]=json.loads((directory/(key+'.json')).read_text())['observations']
                    with np.load(directory/(key+'.npz'),allow_pickle=False) as z:
                        traces[arm]=z['noise_sha256'].copy()
                n,a=audit['normal'],audit['same_task']
                assert a['donor']==donor_plan[str(seed)][key]['same_task']
                assert a['task_at_handover_supplied']['predicted_task']==task
                h=n['handover'];assert a['handover']==h
                initial=n['initial_environment']==a['initial_environment']
                prefix=n['expert_prefix_actions_sha256']==a['expert_prefix_actions_sha256']
                common=min(len(traces['normal']),len(traces['same_task']))
                noise=bool(np.array_equal(traces['normal'][:common],traces['same_task'][:common]))
                assert initial and prefix and noise
                identical=(h<len(heads['normal']) and h<len(heads['same_task']) and
                    all(heads['normal'][h][field]==heads['same_task'][h][field]
                        for field in ('visual_sha256','state_sha256','before_payload_sha256')))
                exact[si,ki]=identical
                pre=n['task_at_handover_before']
                cohort.append(pre is not None and pre['predicted_task']==task)
                pairs.append(dict(seed=seed,episode=key,task=task,initial_environment_matched=initial,
                    expert_actions_matched=prefix,noise_matched=noise,handover_inputs_identical=identical,
                    normal_success=bool(outcomes['normal'][si,ki]),donor_success=bool(outcomes['same_task'][si,ki]),
                    normal_identity_correct=cohort[-1],donor=a['donor']))
                entry=per_task.setdefault(task,dict(trials=0,normal=0,same_task=0,expert=0))
                entry['trials']+=1
                entry['normal']+=int(outcomes['normal'][si,ki])
                entry['same_task']+=int(outcomes['same_task'][si,ki])
                entry['expert']+=int(reports['normal'][key]['replay_check']['task_success'])
            correct.append(cohort)
            per_seed[str(seed)]={arm:int(outcomes[arm][si].sum()) for arm in protocol['arms']}
        mask=eligible['normal']&eligible['same_task']
        assert mask.all(), 'Report policy-ineligible episodes explicitly before comparing'
        delta=outcomes['same_task'].astype(float)-outcomes['normal']
        correct=np.asarray(correct,dtype=bool)
        cohorts={}
        for name,selection in [('correct',correct),('incorrect',~correct)]:
            selection=selection&mask
            cohorts[name]=dict(trials=int(selection.sum()),episode_identities=int(selection.any(0).sum()),
                normal=int((outcomes['normal']&selection).sum()),same_task=int((outcomes['same_task']&selection).sum()))
        rows.append(dict(**model,trials=int(mask.sum()),normal=int((outcomes['normal']&mask).sum()),
            same_task=int((outcomes['same_task']&mask).sum()),per_seed=per_seed,
            effect=clustered_interval(delta,mask),identical_handover_effect=clustered_interval(delta,mask&exact),
            gains=int(((delta>0)&mask).sum()),losses=int(((delta<0)&mask).sum()),
            identity_cohorts=cohorts,per_task=per_task,pair_audits=pairs))
    result=dict(complete=True,protocol=protocol,models=rows,source_results=sources,
                analysis_sha256=digest(__file__),weights_updated=False)
    previous_rows={}
    if protocol.get('previous_batch'):
        previous=json.loads((Path(protocol['previous_batch'])/'result.json').read_text())
        previous_rows={(r['model'],r['variant']):r for r in previous['models']}
        result['previous_checkpoint_comparison']=[dict(model=r['model'],variant=r['variant'],
            previous_epoch=previous_rows[r['model'],r['variant']]['epoch'],new_epoch=r['epoch'],
            previous_normal=previous_rows[r['model'],r['variant']]['normal'],new_normal=r['normal'],
            previous_same_task=previous_rows[r['model'],r['variant']]['same_task'],new_same_task=r['same_task'],
            trials=r['trials']) for r in rows]
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    reused=sum(r['reused'] for r in rows)
    reuse_note=('This report consolidates completed evaluations from the linked source batches; no new rollouts were run for consolidation.'
                if protocol.get('consolidated_from') else
                f'{reused} model(s) reuse their previously completed paired runs; other paired runs are new.'
                if reused else 'All paired runs in this report are new.')
    lines=['# Correct-task donor evaluation across non-text Piper models','',
        '44 policy-validation episodes × 3 evaluation seeds (42, 43, 44) per condition. '
        '40% expert prefix; fixed handover task context; live execution context; 300-step cap; action horizon 1. '
        'Failed expert replays remain included. Every model uses the same predeclared native training donors and per-seed scene states.','',
        'GT task names select correctly decoded, same-task student tokens from training demonstrations. '
        'This is a privileged diagnostic, not deployable task-estimation performance. No weights were updated. '+reuse_note,'',
        '| Model | Variant | Epoch | CP dim | Head width | Normal | Correct-task donor | Difference (95% CI) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    def count(n,d):return f'{n}/{d} ({100*n/d:.1f}%)' if d else 'N/A'
    for r in rows:
        e=r['effect'];lo,hi=e['interval_95']
        lines.append(f"| {r['model']} | {r['variant']} | {r['epoch']} | {r['policy_spec']['compressed_dim']} | {r['head_d_model']} | "
            f"{count(r['normal'],r['trials'])} | {count(r['same_task'],r['trials'])} | "
            f"{100*e['mean']:+.1f} pp [{100*lo:+.1f}, {100*hi:+.1f}] |")
    lines+=['','Intervals resample the 44 episode identities, retaining all three seeds within each cluster. '
        'Three evaluation seeds are not three independently trained policies.']
    if previous_rows:
        lines+=['','## Updated versus previous checkpoints','',
            'Same evaluation protocol, seeds, episodes and native donors. These are different checkpoints from the same training runs.','',
            '| Model / variant | Epoch old → new | Normal old → new | GT donor old → new |',
            '|---|---:|---:|---:|']
        for r in rows:
            old=previous_rows[r['model'],r['variant']]
            if old['checkpoint_sha256']==r['checkpoint_sha256']:
                continue
            lines.append(f"| {r['model']}/{r['variant']} | {old['epoch']} → {r['epoch']} | "
                f"{count(old['normal'],old['trials'])} → {count(r['normal'],r['trials'])} | "
                f"{count(old['same_task'],old['trials'])} → {count(r['same_task'],r['trials'])} |")
    lines+=['',
        '## Per-seed completion counts','',
        '| Model / variant | Seed 42 normal → donor | Seed 43 normal → donor | Seed 44 normal → donor |',
        '|---|---:|---:|---:|']
    for r in rows:
        cells=[f"{r['per_seed'][str(s)]['normal']} → {r['per_seed'][str(s)]['same_task']}" for s in seeds]
        lines.append('| '+r['model']+'/'+r['variant']+' | '+' | '.join(cells)+' |')
    lines+=['','## Rescue when the original task identity decoded incorrectly','',
        '| Model / variant | Incorrect trials | Normal completion | Correct-task donor completion |',
        '|---|---:|---:|---:|']
    for r in rows:
        c=r['identity_cohorts']['incorrect']
        lines.append(f"| {r['model']}/{r['variant']} | {c['trials']} | {count(c['normal'],c['trials'])} | {count(c['same_task'],c['trials'])} |")
    lines+=['','## Identical-input sensitivity check','',
        'Only pairs with bit-identical visual features, state features, and original context payload at handover enter this secondary analysis. '
        'The primary analysis includes every episode.','',
        '| Model / variant | Trials | Difference (95% CI) |','|---|---:|---:|']
    for r in rows:
        e=r['identical_handover_effect'];lo,hi=e['interval_95']
        lines.append(f"| {r['model']}/{r['variant']} | {e['eligible_trials']} | {100*e['mean']:+.1f} pp [{100*lo:+.1f}, {100*hi:+.1f}] |")
    for r in rows:
        lines+=['',f"## Results by task: {r['model']}/{r['variant']}",'',
            '| Task | Expert completion | Normal completion | Correct-task donor completion |',
            '|---|---:|---:|---:|']
        for task,t in sorted(r['per_task'].items()):
            lines.append(f"| {task} | {count(t['expert'],t['trials'])} | {count(t['normal'],t['trials'])} | {count(t['same_task'],t['trials'])} |")
    lines+=['','Scope: known tasks and the existing validation split. Native donors can differ in episode-specific procedure or scene context. '
        'A donor benefit diagnoses sensitivity to inferred task context; it does not establish unseen-task generalization, '
        'a universal upper bound, or superiority over a separately trained no-token policy.','',
        '[Machine-readable results and paired audits](result.json). [Fixed donor plan](donor_plan.json).']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([{k:r[k] for k in ('model','variant','normal','same_task','trials')} for r in rows]),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    summarize(parser.parse_args().root.resolve())
