"""Summarize privileged task-identity oracles without claiming perfect semantics."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from data.task_context_dataset import file_hash


def exact_p(wins,losses):
    n=len(wins)+len(losses)
    return min(1.,2*sum(math.comb(n,j) for j in range(min(len(wins),len(losses))+1))/2**n)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root
    arms=['baseline','repair','episode']
    paths={arm:root/arm/'result.json' for arm in arms}
    if not all(path.exists() for path in paths.values()):
        raise ValueError('Wait for all three complete simulator evaluations')
    data={arm:json.loads(path.read_text()) for arm,path in paths.items()}
    rows={arm:{r['episode']:r for r in d['episodes']} for arm,d in data.items()}
    keys=sorted(rows['baseline'])
    for arm,d in data.items():
        if not d['evaluation_complete'] or len(keys)!=44 or sorted(rows[arm])!=keys:
            raise ValueError('Expected complete evaluations of the same 44 development episodes')
        if d['evaluation_protocol']['replay_filters_policy_episodes']:
            raise ValueError('Replay-filtered comparisons are not allowed')
        for field in ['checkpoint_sha256','evaluator_sha256','protocol_sha256','episode_selection','stop_on_task_success']:
            if d['evaluation_protocol'][field]!=data['baseline']['evaluation_protocol'][field]:
                raise ValueError('Protocol differs: '+field)
        for field,value in d['evaluation_args'].items():
            if field not in ['oracle_task_mode','oracle_task_bank','out_dir'] and value!=data['baseline']['evaluation_args'][field]:
                raise ValueError('Evaluation arguments differ: '+field)
        for key,r in rows[arm].items():
            b=rows['baseline'][key]
            if (r['task_name'],r['episode_seed'],r['switch_step'])!=(b['task_name'],b['episode_seed'],b['switch_step']):
                raise ValueError('Episode pairing differs: '+key)
    baseline_wrong=[k for k in keys if not rows['baseline'][k]['task_estimation']['checkpoints']['handover']['inferred']['correct']]
    replaced=[k for k in keys if rows['repair'][k]['task_context_selection']['replaced']]
    untouched=[k for k in keys if k not in replaced]
    comparisons={}
    for arm in ['repair','episode']:
        wins=[k for k in keys if rows[arm][k]['success'] and not rows['baseline'][k]['success']]
        losses=[k for k in keys if not rows[arm][k]['success'] and rows['baseline'][k]['success']]
        comparisons[arm]=dict(improved=wins,regressed=losses,exact_mcnemar_p=exact_p(wins,losses))
    changed_untouched=[k for k in untouched if rows['repair'][k]['success']!=rows['baseline'][k]['success']]
    repair_cohorts={name:dict(episodes=len(cohort),
        baseline_successes=sum(rows['baseline'][k]['success'] for k in cohort),
        repair_successes=sum(rows['repair'][k]['success'] for k in cohort),
        improved=[k for k in cohort if rows['repair'][k]['success'] and not rows['baseline'][k]['success']],
        regressed=[k for k in cohort if not rows['repair'][k]['success'] and rows['baseline'][k]['success']])
        for name,cohort in [('replaced',replaced),('untouched',untouched)]}
    failed=[k for k in keys if not rows['baseline'][k]['success']]
    audit=[]
    for key in keys:
        values={}
        for arm in arms:
            with np.load(root/arm/f'{key}_rollout.npz',allow_pickle=False) as trace:
                values[arm]=trace['initial_sim_state'].copy()
            with np.load(root/arm/f'{key}_task_estimates.npz',allow_pickle=False) as trace:
                after=trace['steps']>=rows[arm][key]['switch_step']
                if not after.any():
                    raise ValueError('Missing policy observations: '+key)
                identity=bool((trace['probabilities'][after,1].argmax(-1)==rows[arm][key]['task_estimation']['actual_group']).all())
                if arm!='baseline' and not identity:
                    raise ValueError('Oracle task identity was not correct for every policy observation')
                if arm!='baseline' and not rows[arm][key]['task_estimation']['ground_truth_is_model_input']:
                    raise ValueError('Oracle trace failed to expose GT usage')
                if trace['execution_tokens'].shape[0]!=len(trace['steps']):
                    raise ValueError('Missing live execution trace')
        audit.append(dict(episode=key,initial_states_identical=all(np.array_equal(v,values['baseline']) for v in values.values())))
    totals={arm:dict(successes=sum(r['success'] for r in rr.values()),episodes=len(rr),
        task_identity_correct=sum(r['task_estimation']['checkpoints']['handover']['policy']['correct'] for r in rr.values()),
        expert_task_successes=sum(r['replay_check']['task_success'] for r in rr.values())) for arm,rr in rows.items()}
    result=dict(totals=totals,comparisons=comparisons,baseline_task_wrong=baseline_wrong,
        repaired_episodes=replaced,untouched_episodes=untouched,untouched_changed_outcomes=changed_untouched,
        repair_cohorts=repair_cohorts,
        handover_identity_agrees_before_repair=sorted(baseline_wrong)==sorted(replaced),
        baseline_failures_with_correct_task=[k for k in failed if k not in baseline_wrong],
        baseline_failures_with_wrong_task=[k for k in failed if k in baseline_wrong],
        initial_state_audit=audit,privileged=True,deployable=False,weights_updated=False,
        confirmation_opened=False,source_results={arm:dict(path=str(path.resolve()),sha256=file_hash(path)) for arm,path in paths.items()},
        source_sha256=file_hash(__file__))
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    offline=json.loads((root/'offline/result.json').read_text())
    lines=['# Does task identification limit the policy?','',
        '**Privileged diagnostic on Policy2: GT task labels and potentially future demonstration observations select native student tokens. '
        'This guarantees correct decoder identity; it does not certify a perfect complete task representation. No policy weights changed.**','',
        'All 44 development episodes; 40% expert prefix; fixed handover task snapshot; live execution context; seed 42; '
        'action horizon 1; 300-step limit; stop at first success. Failed expert replays remain included.','',
        '| Condition | Correct actor task at takeover | Policy completion | Expert replay completion |',
        '|---|---:|---:|---:|']
    for arm,t in totals.items():
        lines.append(f"| {arm} | {t['task_identity_correct']}/44 ({100*t['task_identity_correct']/44:.1f}%) | "
            f"{t['successes']}/44 ({100*t['successes']/44:.1f}%) | {t['expert_task_successes']}/44 ({100*t['expert_task_successes']/44:.1f}%) |")
    lines+=['','- **baseline:** use the task snapshot inferred from actual simulator observations at handover.',
        '- **repair:** retain correctly decoded live snapshots; replace only wrong identities with a correctly decoded same-demonstration native snapshot.',
        '- **episode:** replace every task snapshot with the same-demonstration oracle snapshot. '
        'This also changes episode/context details on already-correct tasks.','',
        '| Oracle | Newly successful episodes | Newly failed episodes | Exact paired p |', '|---|---:|---:|---:|']
    for arm,c in comparisons.items():
        lines.append(f"| {arm} | {len(c['improved'])} | {len(c['regressed'])} | {c['exact_mcnemar_p']:.3f} |")
    lines+=['','| Repair subgroup | Episodes | Baseline successes | Repair successes | Gains / losses |',
        '|---|---:|---:|---:|---:|']
    for name,c in repair_cohorts.items():
        lines.append(f"| {name} | {c['episodes']} | {c['baseline_successes']} | {c['repair_successes']} | "
            f"{len(c['improved'])} / {len(c['regressed'])} |")
    lines+=['',f"Repair changes completion by {totals['repair']['successes']-totals['baseline']['successes']:+d} episodes "
        f"({100*(totals['repair']['successes']-totals['baseline']['successes'])/44:+.1f} percentage points); "
        f"the episode oracle changes it by {totals['episode']['successes']-totals['baseline']['successes']:+d} "
        f"({100*(totals['episode']['successes']-totals['baseline']['successes'])/44:+.1f} points). "
        'These are descriptive differences from one evaluation seed; paired tests and untouched controls above qualify their interpretation.']
    lines+=['',f"Baseline failures: {len(result['baseline_failures_with_correct_task'])} already had the correct decoded task; "
        f"{len(result['baseline_failures_with_wrong_task'])} had the wrong decoded task. "
        'Task classification errors therefore cannot explain every failure. Correct classification can still hide missing object, goal, or procedure information.','',
        f"Repair replaced {len(replaced)} snapshots and left {len(untouched)} episodes untouched. "
        f"Outcomes changed in {len(changed_untouched)} untouched cases. "
        'Changed outcomes on untouched cases measure residual simulator/inference variability and limit causal attribution. '
        f"Initial simulator states matched across all arms in {sum(r['initial_states_identical'] for r in audit)}/44 cases. "
        'Only one simulator evaluation seed was used.','',
        '## Fixed-observation action diagnostic','',
        'Four expert anchors per development episode and three matched diffusion draws. '
        'Both oracles correct 8/44 cached expert task snapshots; live simulator observations produce a slightly different identification set.','',
        '| Condition | First-action position MSE | Chunk position MSE | Action MSE |', '|---|---:|---:|---:|']
    for arm,r in offline['runs'].items():
        m=r['mean'];lines.append(f"| {arm} | {m['first_position_mse']:.6f} | {m['position_mse']:.6f} | {m['action_mse']:.6f} |")
    lines+=['','[Offline paired intervals and subgroup errors](offline/summary.md).','',
        '## Results by task','',
        '| Task | Expert completion | Baseline | Repair | Episode oracle |','|---|---:|---:|---:|---:|']
    for task in sorted({r['task_name'] for r in rows['baseline'].values()}):
        kk=[k for k in keys if rows['baseline'][k]['task_name']==task];n=len(kk)
        expert=sum(rows['baseline'][k]['replay_check']['task_success'] for k in kk)
        cells=[f'{expert}/{n} ({100*expert/n:.1f}%)']
        for arm in arms:
            count=sum(rows[arm][k]['success'] for k in kk);cells.append(f'{count}/{n} ({100*count/n:.1f}%)')
        lines.append('| '+task+' | '+' | '.join(cells)+' |')
    lines+=['','## Interpretation limits','',
        'A large improvement under repair would implicate task inference. A small improvement limits how much known-task identity mistakes explain, '
        'but does not establish that the whole context representation is adequate. '
        'Later snapshots may encode additional procedural information or a different phase. Recorded expert scenes may differ from seeded simulator scenes. '
        'Neither oracle is a mathematically guaranteed success upper bound.','',
        ('For this checkpoint and protocol, neither oracle demonstrated a statistically reliable completion change. '
         'The offline improvement shows that task information can affect action quality, but the simulator evidence does not '
         'support treating identity accuracy as the dominant performance bottleneck. '
         if all(c['exact_mcnemar_p']>=.05 for c in comparisons.values()) else
         'At least one paired comparison reached p < 0.05; inspect its gain/loss direction and untouched variability before attributing a benefit. ')+
        'The next discriminating experiment is to train a policy with correct native task context on the training partition '
        'and evaluate it with oracle context: that tests whether learning to use context, rather than inferring its identity, '
        'is limiting performance. Keep live execution context and evaluate normal inferred context alongside the oracle.','',
        'No confidence scores were forced to 1.0; oracle correctness is checked by decoding the supplied native vectors. '
        'Bank selection used no simulator policy outcomes. Source weights and the 340/44/44 split are unchanged; confirmation remains closed.','',
        '[Complete paired episode IDs, state audit, and provenance](comparison.json). '
        '[Oracle bank definition and per-episode source observations](oracle_bank.json).']
    text='\n'.join(lines)+'\n'
    (root/'summary.md').write_text(text)
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
