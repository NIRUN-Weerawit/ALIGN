"""Paired completion and semantic-direction audits for task-token counterfactuals."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ARMS=('normal','repeat','same_task','wrong_task','no_task')


def clustered_interval(deltas,masks,seed=20261026):
    """Resample episode clusters, retaining all evaluation seeds within each."""
    deltas=np.asarray(deltas,dtype=float);masks=np.asarray(masks,dtype=float)
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,deltas.shape[1],size=(10000,deltas.shape[1]))
    numerator=(deltas*masks).sum(0)[indices].sum(-1)
    denominator=masks.sum(0)[indices].sum(-1)
    valid=denominator>0
    samples=numerator[valid]/denominator[valid]
    return dict(mean=float((deltas*masks).sum()/masks.sum()) if masks.sum() else None,
        interval_95=np.quantile(samples,[.025,.975]).tolist() if len(samples) else None,
        episode_clusters=deltas.shape[1],eligible_trials=int(masks.sum()),bootstrap_draws=10000)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve()
    protocol=json.loads((root/'protocol.json').read_text())
    keys=protocol['episodes'];seeds=protocol['seeds']
    results={};traces={};audits={};data={};task_names={};source_results=[]
    for seed in seeds:
        for arm in ARMS:
            directory=root/'runs'/f'seed_{seed}'/arm
            path=directory/'result.json'
            if not path.exists():raise ValueError(f'Incomplete: seed {seed}, {arm}')
            d=json.loads(path.read_text());assert d['evaluation_complete']
            source_results.append(dict(seed=seed,arm=arm,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            assert d['evaluation_protocol']['checkpoint_sha256']==protocol['checkpoint_sha256']
            assert sorted(r['episode'] for r in d['episodes'])==keys
            results[seed,arm]={r['episode']:r for r in d['episodes']}
            data[seed,arm]=d
            for key in keys:
                audits[seed,arm,key]=json.loads((directory/'counterfactual'/(key+'_audit.json')).read_text())
                with np.load(directory/'counterfactual'/(key+'.npz'),allow_pickle=False) as t:
                    traces[seed,arm,key]={k:t[k].copy() for k in ['noise_sha256','task_contribution_active']}
                task_names[key]=results[seed,arm][key]['task_name']
    totals={};contrasts={};pair_audits=[];direction={};handover_actions={};repeat_trajectories=[]
    normal_correct=np.array([[audits[s,'normal',k]['task_at_handover_before'] is not None and
        audits[s,'normal',k]['task_at_handover_before']['predicted_task']==task_names[k] for k in keys] for s in seeds])
    normal_success=np.array([[results[s,'normal'][k]['success_after_handover'] for k in keys] for s in seeds],dtype=float)
    eligible=np.array([[results[s,'normal'][k]['policy_eligible'] for k in keys] for s in seeds])
    for arm in ARMS:
        per_seed={}
        for seed in seeds:
            rows=list(results[seed,arm].values());valid=[r for r in rows if r['policy_eligible']]
            per_seed[str(seed)]=dict(successes=sum(r['success_after_handover'] for r in valid),episodes=len(valid))
        totals[arm]=dict(successes=sum(x['successes'] for x in per_seed.values()),episodes=sum(x['episodes'] for x in per_seed.values()),per_seed=per_seed)
        if arm=='normal':continue
        outcomes=np.array([[results[s,arm][k]['success_after_handover'] for k in keys] for s in seeds],dtype=float)
        arm_eligible=np.array([[results[s,arm][k]['policy_eligible'] for k in keys] for s in seeds])
        mask=eligible&arm_eligible
        delta=outcomes-normal_success
        contrasts[arm]=dict(all_episodes=clustered_interval(delta,mask),
            correct_normal_at_handover=clustered_interval(delta,mask&normal_correct),
            gained=int(((delta>0)&mask).sum()),lost=int(((delta<0)&mask).sum()))
        handover_actions[arm]=[]
        exact_mask=np.zeros_like(mask)
        for seed in seeds:
            for key in keys:
                b=audits[seed,'normal',key];a=audits[seed,arm,key]
                bt=traces[seed,'normal',key];at=traces[seed,arm,key]
                common=min(len(bt['noise_sha256']),len(at['noise_sha256']))
                matched_noise=bool(np.array_equal(bt['noise_sha256'][:common],at['noise_sha256'][:common]))
                if not matched_noise:raise ValueError('Mismatched DDIM draws')
                geometry=b['initial_environment']['geometry']==a['initial_environment']['geometry']
                state=b['initial_environment']['state']==a['initial_environment']['state']
                if not geometry or not state:raise ValueError('Mismatched initial scene or state')
                if b['expert_prefix_actions_sha256']!=a['expert_prefix_actions_sha256']:
                    raise ValueError('Different executed expert commands')
                used=a['task_at_handover_supplied']
                if used is not None and arm in ('same_task','wrong_task'):
                    assert (used['predicted_task']==task_names[key])==(arm=='same_task')
                if arm=='no_task':
                    assert not at['task_contribution_active'][a['handover']:].any()
                baseline_dir=root/'runs'/f'seed_{seed}'/'normal'
                arm_dir=root/'runs'/f'seed_{seed}'/arm
                bh=json.loads((baseline_dir/'counterfactual'/(key+'.json')).read_text())['observations']
                ah=json.loads((arm_dir/'counterfactual'/(key+'.json')).read_text())['observations']
                h=b['handover']
                input_match=(h<len(bh) and h<len(ah) and all(bh[h][field]==ah[h][field]
                    for field in ('visual_sha256','state_sha256','before_payload_sha256')))
                exact_mask[seeds.index(seed),keys.index(key)]=input_match
                with np.load(baseline_dir/(key+'_rollout.npz'),allow_pickle=False) as br,np.load(arm_dir/(key+'_rollout.npz'),allow_pickle=False) as ar:
                    if input_match:
                        handover_actions[arm].append(dict(seed=seed,episode=key,
                            condition_changed=bh[h]['condition_sha256']!=ah[h]['condition_sha256'],
                            normalized_position_action_l2=float(np.linalg.norm(br['predicted_actions'][h,:3]-ar['predicted_actions'][h,:3]))))
                    if arm=='repeat':
                        n=min(len(br['sim_positions']),len(ar['sim_positions']))
                        repeat_trajectories.append(dict(seed=seed,episode=key,
                            common_steps=n,max_position_delta_m=float(np.linalg.norm(br['sim_positions'][:n,:3]-ar['sim_positions'][:n,:3],axis=-1).max()) if n else 0.,
                            max_predicted_action_delta=float(np.abs(br['predicted_actions'][:n]-ar['predicted_actions'][:n]).max()) if n else 0.))
                pair_audits.append(dict(seed=seed,episode=key,arm=arm,noise_matched=matched_noise,
                    initial_geometry_matched=geometry,initial_state_matched=state,
                    initial_cameras_matched=b['initial_environment']['cameras']==a['initial_environment']['cameras'],
                    expert_prefix_poses_matched=b['expert_prefix_poses_sha256']==a['expert_prefix_poses_sha256'],
                    expert_prefix_states_and_cameras_matched=b['environment_steps'][:b['handover']]==a['environment_steps'][:a['handover']]))
        contrasts[arm]['identical_handover_inputs']=clustered_interval(delta,mask&exact_mask)
    for arm in ARMS:
        rows=[audits[s,arm,k] for s in seeds for k in keys]
        valid=[r for r in rows if r['alternative_goal']['available'] and not r['alternative_goal']['true_before_handover']]
        direction[arm]=dict(evaluable=len(valid),new_wrong_goal_completions=sum(r['alternative_goal']['true_after_handover'] for r in valid),
            unavailable=sum(not r['alternative_goal']['available'] for r in rows),
            already_true_before_handover=sum(r['alternative_goal']['available'] and r['alternative_goal']['true_before_handover'] for r in rows))
    same=np.array([[results[s,'same_task'][k]['success_after_handover'] for k in keys] for s in seeds],dtype=float)
    wrong=np.array([[results[s,'wrong_task'][k]['success_after_handover'] for k in keys] for s in seeds],dtype=float)
    donor_mask=np.array([[results[s,'same_task'][k]['policy_eligible'] and results[s,'wrong_task'][k]['policy_eligible'] for k in keys] for s in seeds])
    same_vs_wrong=clustered_interval(same-wrong,donor_mask)
    cohorts={}
    for label,selection in [('correct_decoded_identity',normal_correct),('incorrect_decoded_identity',~normal_correct)]:
        cohorts[label]={}
        for arm in ARMS:
            cohort_mask=selection&eligible&np.array([[results[s,arm][k]['policy_eligible'] for k in keys] for s in seeds])
            outcomes=np.array([[results[s,arm][k]['success_after_handover'] for k in keys] for s in seeds])
            cohorts[label][arm]=dict(successes=int((outcomes&cohort_mask).sum()),trials=int(cohort_mask.sum()),
                unique_episode_identities=int(cohort_mask.any(0).sum()))
    redirections=[]
    for seed in seeds:
        for key in keys:
            a=audits[seed,'wrong_task',key];g=a['alternative_goal']
            if g['available'] and g['true_after_handover'] and not g['true_before_handover'] and not results[seed,'wrong_task'][key]['success']:
                redirections.append(dict(seed=seed,episode=key,intended_task=a['actual_task'],supplied_task=g['task'],
                    wrong_goal_first_reached_after_action=next(r['after_action_step'] for r in g['observations'] if r['after_action_step']>=a['handover'] and r['success'])))
    result=dict(complete=True,totals=totals,contrasts=contrasts,alternative_goal=direction,pair_audits=pair_audits,
        normal_handover_identity_correct=int(normal_correct.sum()),normal_handover_trials=int(normal_correct.size),
        protocol=protocol,weights_updated=False,same_task_minus_wrong_task=same_vs_wrong,
        matched_handover_action_effects=handover_actions,repeat_trajectory_variation=repeat_trajectories,
        source_results=source_results,analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        privileged_ground_truth_selected_arms=['same_task','wrong_task'],
        goal_redirection_examples=redirections,
        preintervention_identity_cohorts=cohorts,
        analysis_note='Wrong-token damage tests dependence; superiority over a policy trained without task tokens is not established by this inference-only ablation.')
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Do task tokens improve successful control?','',
        'LIBERO simulator evaluation of the frozen cp4/no-memory checkpoint from the completed Piper batch. Four task-context conditions plus a repeated normal control; '
        f"{len(seeds)} evaluation seeds × {len(keys)} episodes. All expert-replay outcomes included. "
        '40% expert prefix, fixed handover task context, live execution context, 300-step cap, action horizon 1.','',
        '| Condition | '+' | '.join('Seed '+str(s) for s in seeds)+' | Total completion |',
        '|---|'+'---:|'*(len(seeds)+1)]
    for arm,t in totals.items():
        cells=[f"{t['per_seed'][str(s)]['successes']}/{t['per_seed'][str(s)]['episodes']}" for s in seeds]
        n=t['episodes'];count=t['successes'];cells.append(f'{count}/{n} ({100*count/n:.1f}%)' if n else 'N/A')
        lines.append('| '+arm+' | '+' | '.join(cells)+' |')
    support=all(contrasts[arm]['all_episodes']['interval_95'][1]<0 for arm in ('wrong_task','no_task'))
    lines+=['',('**Finding: the explicit task-token contribution improves completion in this frozen policy.** '
        'Both wrong-task conditioning and removal reduce success, with episode-cluster intervals excluding zero.' if support else
        '**Finding: inspect the paired effects below; a benefit over both wrong-task and removed-task conditions is not established by both intervals.**'),
        'Same-task donors are selected using recorded task labels and are a privileged diagnostic, not deployment performance.']
    lines+=['','Totals are repeated trials on 44 episode identities, not independent new tasks. '
        'Intervals resample episode clusters and retain all evaluation seeds within each cluster.','',
        '| Comparison vs normal | Completion difference | Episode-cluster 95% interval | Gains / losses |',
        '|---|---:|---:|---:|']
    for arm,c in contrasts.items():
        r=c['all_episodes'];lo,hi=r['interval_95']
        lines.append(f"| {arm} | {100*r['mean']:+.1f} pp | [{100*lo:+.1f}, {100*hi:+.1f}] pp | {c['gained']} / {c['lost']} |")
    lo,hi=same_vs_wrong['interval_95']
    lines+=['',f"Same-task donor minus wrong-task donor: **{100*same_vs_wrong['mean']:+.1f} percentage points**, "
        f"episode-cluster 95% interval **[{100*lo:+.1f}, {100*hi:+.1f}] points**. "
        'Both arms replace the bundle with native training tokens, providing a closer semantic control than an unmodified-versus-corrupted comparison.']
    lines+=['','## Correct normal identity before intervention','',
        f"Normal handover tokens decode correctly in {normal_correct.sum()}/{normal_correct.size} trials. "
        'This cohort is selected before autonomous outcomes, not by policy or expert success.','',
        '| Original decoded identity | Normal completion | Same-task donor completion | Unique episode IDs |',
        '|---|---:|---:|---:|']
    for label,c in cohorts.items():
        n=c['normal'];s=c['same_task']
        lines.append(f"| {label} | {n['successes']}/{n['trials']} | {s['successes']}/{s['trials']} | {n['unique_episode_identities']} |")
    lines+=['',
        '| Comparison vs normal | Completion difference | Episode-cluster 95% interval | Trials |',
        '|---|---:|---:|---:|']
    for arm,c in contrasts.items():
        r=c['correct_normal_at_handover'];lo,hi=r['interval_95']
        lines.append(f"| {arm} | {100*r['mean']:+.1f} pp | [{100*lo:+.1f}, {100*hi:+.1f}] pp | {r['eligible_trials']} |")
    lines+=['','## Sensitivity analysis: identical inputs at handover','',
        'This secondary subset requires exact equality of visual features, state features, and the original task/execution payload '
        'before the head intervention. Selection uses pre-intervention inputs, never rollout success. All trials remain in the primary analysis.','',
        '| Comparison vs normal | Completion difference | Episode-cluster 95% interval | Trials |',
        '|---|---:|---:|---:|']
    for arm,c in contrasts.items():
        r=c['identical_handover_inputs'];lo,hi=r['interval_95']
        lines.append(f"| {arm} | {100*r['mean']:+.1f} pp | [{100*lo:+.1f}, {100*hi:+.1f}] pp | {r['eligible_trials']} |")
    lines+=['','## Did the policy reach the supplied wrong goal?','',
        'Only goals whose entities and region definitions are present in the actual scene are checked. '
        'An alternative goal already true before handover is excluded. The original task remains the rollout success criterion; '
        'alternative-goal observations never stop or modify the rollout.','',
        '| Condition | Newly achieves alternative goal | Evaluable trials |', '|---|---:|---:|']
    for arm,r in direction.items():lines.append(f"| {arm} | {r['new_wrong_goal_completions']} | {r['evaluable']} |")
    lines+=['','Examples with original-task failure and subsequent alternative-goal success:','']
    for row in redirections[:4]:
        lines.append(f"- Seed {row['seed']}, {row['episode']}: intended **{row['intended_task']}**; supplied **{row['supplied_task']}**. "
            f"The supplied goal predicate becomes true at action {row['wrong_goal_first_reached_after_action']}.")
    lines+=['','## Matching and variability controls','',
        f"All {len(pair_audits)} arm/seed/episode pairs have matching initial geometry, initial simulator state, executed expert commands, and DDIM noise draws. "
        f"Initial cameras match exactly in {sum(r['initial_cameras_matched'] for r in pair_audits)}/{len(pair_audits)}; "
        f"expert-prefix poses in {sum(r['expert_prefix_poses_matched'] for r in pair_audits)}/{len(pair_audits)}; "
        f"prefix states and camera hashes in {sum(r['expert_prefix_states_and_cameras_matched'] for r in pair_audits)}/{len(pair_audits)}. "
        'The repeated normal arm quantifies remaining simulator/inference variability.','',
        'Same-task and wrong-task arms use known task labels solely to select training donors. '
        'Donors are native student vectors from 40% training prefixes, validated with the fixed training-only decoder. '
        'No teacher vectors, evaluation future observations, synthetic random tokens, or policy outcomes select donors. '
        'The current state, visuals, live execution token, encoder state and policy weights receive no direct intervention. '
        'After actions diverge, subsequent observations and execution context naturally diverge too.','',
        '| Condition | Identical handover-input trials | Changed head conditioning | Mean first position-action L2 difference |',
        '|---|---:|---:|---:|']
    for arm,rows in handover_actions.items():
        mean=float(np.mean([r['normalized_position_action_l2'] for r in rows])) if rows else float('nan')
        lines.append(f"| {arm} | {len(rows)} | {sum(r['condition_changed'] for r in rows)} | {mean:.5f} |")
    lines+=['','Position-action differences are in normalized action units, not metres. '
        'Only trials with identical visual, state, and original payload inputs at handover enter this table.','',
        '## Results by task','',
        '| Task | Expert completion | Normal | Repeat | Same task | Wrong task | No task contribution |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for task in sorted(set(task_names.values())):
        selected=[(s,k) for s in seeds for k in keys if task_names[k]==task]
        e=[results[s,'normal'][k]['replay_check']['task_success'] for s,k in selected]
        cells=[f'{sum(e)}/{len(e)} ({100*np.mean(e):.1f}%)']
        for arm in ARMS:
            rr=[results[s,arm][k] for s,k in selected if results[s,arm][k]['policy_eligible']]
            n=len(rr);count=sum(r['success_after_handover'] for r in rr)
            cells.append(f'{count}/{n} ({100*count/n:.1f}%)' if n else 'N/A')
        lines.append('| '+task+' | '+' | '.join(cells)+' |')
    lines+=['','## Interpretation','',
        'Normal and same-task donor performance above wrong-task performance supports a semantic task-context contribution. '
        'A same-task donor regression also points to episode-specific information or donor mismatch. '
        'Wrong-task damage without a normal-over-no-task advantage only establishes that wrong context can hurt. '
        'Similar completion can also reflect redundant task information in execution/vision, or recovery after action changes. '
        'A matched policy trained without task tokens is still needed to establish an architectural benefit.','',
        'Scope: one frozen checkpoint, known tasks, and the same 44 policy-validation episodes used in earlier checkpoint selection. '
        'These results do not establish unseen-task generalization or real-robot handover performance.','',
        '[Full paired audits, effects, and source protocol](result.json). [Predeclared training donors](donor_plan.json).']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    (root/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
