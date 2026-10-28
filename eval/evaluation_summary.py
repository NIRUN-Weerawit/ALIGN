"""Small, dependency-free summaries of saved LIBERO evaluation results.

Derive counts from episode records, preserve policy/expert denominators, and
keep post-completion task-label changes separate from recognition at success.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path


POINTS = ('initial', 'handover', 'last_reference', 'final')
POINT_LABELS = dict(initial='Initial observation', handover='Handover',
                    last_reference='Last demonstration frame', final='Final evaluation frame')


def _rate(correct, total):
    return dict(correct=int(correct), total=int(total), rate=correct/total if total else None)


def _eligible(row):
    return row.get('status','evaluated') == 'evaluated' and row.get('policy_eligible',not row.get('success_before_handover',False))


def _estimate(row, point, role='inferred'):
    context = row.get('task_estimation') or {}
    if not context.get('actual_task_known',False):
        return None
    return (context.get('checkpoints',{}).get(point) or {}).get(role)


def _recognition(rows, point, role='inferred'):
    scored = [(row, _estimate(row,point,role)) for row in rows]
    scored = [(row,prediction) for row,prediction in scored
              if prediction is not None and isinstance(prediction.get('correct'),bool)]
    result = _rate(sum(p['correct'] for _,p in scored),len(scored))
    groups = {}
    for row,prediction in scored:
        group = row['task_estimation'].get('actual_task') or row.get('task_name','?')
        groups.setdefault(group,[]).append(prediction['correct'])
    result['balanced_accuracy'] = (sum(sum(values)/len(values) for values in groups.values())/len(groups)
                                   if groups else None)
    return result


def _outcome(rows):
    eligible = [row for row in rows if _eligible(row)]
    return _rate(sum(bool(row.get('success')) for row in eligible),len(eligible))


def _expert_outcome(rows):
    outcomes = [(row.get('replay_check') or {}).get('task_success') for row in rows]
    recorded = [value for value in outcomes if isinstance(value,bool)]
    return _rate(sum(recorded),len(recorded))


def _trace_diagnostics(rows, directory):
    """Optional sidecars improve interpretation; stale sidecars are never used."""
    success_rows = [row for row in rows if _eligible(row) and row.get('success')]
    result = dict(sidecars_loaded=0,missing_sidecars=[],invalid_sidecars=[],
                  success_episodes=len(success_rows),at_first_success={},first_errors_after_handover=[])
    at_success = {role:[] for role in ('inferred','policy')}
    if directory is None:
        result['available'] = False
        return result
    result['available'] = True
    for row in rows:
        if not row.get('task_estimation'):
            continue
        key = row['episode']
        path = Path(directory)/f'{key}_task_estimates.json'
        if not path.exists():
            result['missing_sidecars'].append(key)
            continue
        try:
            trace = json.loads(path.read_text())
            records = trace['observations']
            if trace.get('summary') != row['task_estimation']:
                raise ValueError('Sidecar summary does not match this evaluation episode')
            by_step = {record['step']:record for record in records}
            if len(by_step) != len(records) or len(records) != row['task_estimation']['observations']:
                raise ValueError('Duplicate steps or mismatched observation coverage')
        except (OSError,ValueError,KeyError,TypeError) as error:
            result['invalid_sidecars'].append(dict(episode=key,reason=str(error)))
            continue
        result['sidecars_loaded'] += 1
        success_step = row.get('first_success_step')
        if (_eligible(row) and row.get('success') and row['task_estimation'].get('actual_task_known')
                and success_step in by_step):
            for role in at_success:
                prediction = by_step[success_step].get(role) or {}
                if isinstance(prediction.get('correct'),bool):
                    at_success[role].append(prediction['correct'])
        handover = _estimate(row,'handover')
        if handover and handover.get('correct') is True:
            step = row.get('switch_step',0)
            first_wrong = next((r['step'] for r in records if r['step'] > step
                                and (r.get('inferred') or {}).get('correct') is False),None)
            if first_wrong is not None:
                last = (row['task_estimation']['checkpoints'].get('last_reference') or {}).get('step')
                result['first_errors_after_handover'].append(dict(episode=key,step=first_wrong,
                    after_demonstration=first_wrong > last if last is not None else None,
                    after_task_completion=first_wrong > success_step if success_step is not None else None))
    result['at_first_success'] = {role:_rate(sum(values),len(values)) for role,values in at_success.items()}
    return result


def build_evaluation_summary(report, trace_directory=None):
    if 'episodes' not in report:
        raise ValueError('Expected an evaluation result with an episodes list')
    rows = report['episodes']
    if not isinstance(rows,list):
        raise ValueError('Expected an evaluation result with an episodes list')
    keys = [row['episode'] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate episodes in evaluation result')
    protocol = report.get('evaluation_protocol',{})
    selection = protocol.get('episode_selection',{})
    args = report.get('evaluation_args',{})
    replay_only = report.get('evaluation_kind') == 'expert replay validation' or args.get('replay_only',False)
    evaluated = [] if replay_only else [row for row in rows if row.get('status','evaluated') == 'evaluated']
    eligible = [row for row in evaluated if _eligible(row)]
    requested = report.get('requested_episodes',len(rows))
    issues = []
    completion = _outcome(eligible)
    if report.get('success_count') is not None and report['success_count'] != completion['correct']:
        issues.append('Recorded policy success count disagrees with episode records; summary uses episode records.')
    if report.get('evaluation_complete') and len(rows) != requested:
        issues.append('Report claims completion but processed/requested episode counts differ.')
    tasks = {}
    for name in sorted({row.get('task_name','?') for row in rows}):
        group = [row for row in evaluated if row.get('task_name','?') == name]
        task_rows = [row for row in rows if row.get('task_name','?') == name]
        tasks[name] = dict(episodes=len(task_rows),policy_completion=_outcome(group),
            expert_completion=_expert_outcome(task_rows),
            recognition={point:_recognition(group,point) for point in POINTS})
    stages = {point:{role:_recognition(evaluated,point,role) for role in ('inferred','policy')} for point in POINTS}
    failed = [row for row in eligible if row.get('success') is False]
    successful = [row for row in eligible if row.get('success') is True]
    mistakes = []
    retained, lost, recovered = [], [], []
    for row in evaluated:
        handover,final = _estimate(row,'handover'),_estimate(row,'final')
        if handover and handover.get('correct') is False:
            mistakes.append(dict(episode=row['episode'],actual_task=row['task_name'],
                predicted_task=handover.get('predicted_task'),top1_probability=handover.get('top1_probability'),
                policy_success=row.get('success')))
        if handover and final:
            if handover.get('correct') is True and final.get('correct') is True:retained.append(row['episode'])
            if handover.get('correct') is True and final.get('correct') is False:lost.append(row['episode'])
            if handover.get('correct') is False and final.get('correct') is True:recovered.append(row['episode'])
    replay_pass = [r for r in rows if r.get('replay_check',{}).get('valid') is True]
    replay_fail = [r for r in rows if r.get('replay_check',{}).get('valid') is False]
    expert_fail = [r for r in rows if r.get('replay_check',{}).get('task_success') is False]
    preflight = {r['episode']:r for r in protocol.get('episodes',[])}
    extended = 0
    for row in evaluated:
        n = preflight.get(row['episode'],{}).get('demonstration_steps',row.get('reference_steps'))
        if n is not None and row.get('n_steps',0) > n:extended += 1
    traces = _trace_diagnostics(evaluated,trace_directory)
    if traces['invalid_sidecars']:
        issues.append(f'{len(traces["invalid_sidecars"])} stale or invalid task sidecars excluded from per-step diagnostics.')
    result = dict(schema='align-evaluation-quick-summary-v1',
        evaluation_kind='expert replay validation' if replay_only else report.get('evaluation_kind','policy rollout'),
        complete=bool(report.get('evaluation_complete',len(rows)==requested)) and len(rows)==requested,
        coverage=dict(requested=requested,processed=len(rows),policy_evaluated=len(evaluated)),
        settings=dict(checkpoint=report.get('checkpoint'),seed=args.get('seed'),
            expert_prefix_fraction=args.get('switch_at'),max_steps=args.get('max_steps'),
            stop_on_task_success=protocol.get('stop_on_task_success',False),
            task_snapshot_selection=protocol.get('task_snapshot_selection'),
            privileged_context=bool(protocol.get('task_oracle')),
            task_oracle=protocol.get('task_oracle'),
            task_decoder_controls_snapshot=bool((protocol.get('task_estimator') or {}).get('used_for_task_snapshot_selection')),
            task_context_modes=dict(Counter(row.get('task_context_mode') for row in evaluated
                                            if row.get('task_context_mode') is not None))),
        exposure=dict(selection=selection.get('selection_mode',selection.get('source','unspecified')),
            split_membership=selection.get('split_membership'),
            held_out_for_checkpoint=selection.get('held_out_for_this_checkpoint'),
            confirmation_opened=selection.get('confirmation_opened')),
        policy_completion=completion,
        overall_completion=_rate(sum(bool(r.get('episode_success',r.get('success'))) for r in evaluated),len(evaluated)),
        expert_completed_before_handover=sum(bool(r.get('success_before_handover')) for r in evaluated),
        replay=dict(passed=len(replay_pass),failed=len(replay_fail),unrecorded=len(rows)-len(replay_pass)-len(replay_fail),
            expert_task_completion=_expert_outcome(rows),
            actual_expert_task_failures=len(expert_fail),policy_completion_on_failed_checks=_outcome(replay_fail),
            recognition_at_handover_on_failed_checks=_recognition(replay_fail,'handover'),
            policy_completion_on_expert_task_failures=_outcome(expert_fail),
            policy_episodes_filtered=protocol.get('replay_filters_policy_episodes')),
        task_recognition=dict(episodes_with_estimates=sum(bool(r.get('task_estimation')) for r in evaluated),
            unknown_task_episodes=sum(bool(r.get('task_estimation')) and not r['task_estimation'].get('actual_task_known',False) for r in evaluated),
            stages=stages,failed_policy_with_correct_handover=_recognition(failed,'handover'),
            successful_policy_with_correct_handover=_recognition(successful,'handover'),
            handover_mistakes=mistakes,retained_correct_to_final=retained,
            correct_to_incorrect=lost,incorrect_to_correct=recovered,probabilities_calibrated=False),
        trajectory=dict(final_beyond_demonstration=extended,
            successful_before_final_frame=sum(r.get('first_success_step') is not None and
                r['first_success_step'] < r.get('n_steps',0)-1 for r in successful),**traces),
        per_task=tasks,issues=issues)
    return result


def compare_evaluations(current, baseline):
    a = {r['episode']:r for r in baseline.get('episodes',[])}
    b = {r['episode']:r for r in current.get('episodes',[])}
    if len(a)!=len(baseline.get('episodes',[])) or len(b)!=len(current.get('episodes',[])):
        raise ValueError('Duplicate episodes in comparison input')
    common = sorted(set(a)&set(b))
    eligible_common = [k for k in common if _eligible(a[k]) and _eligible(b[k])]
    changes = [dict(episode=k,previous_success=bool(a[k].get('success')),current_success=bool(b[k].get('success')))
               for k in eligible_common if bool(a[k].get('success')) != bool(b[k].get('success'))]
    before,after = baseline.get('evaluation_args',{}),current.get('evaluation_args',{})
    return dict(same_episode_set=set(a)==set(b),shared_episodes=len(common),shared_policy_eligible_episodes=len(eligible_common),
        only_current=len(set(b)-set(a)),only_previous=len(set(a)-set(b)),
        previous_completion=_outcome(list(a.values())),current_completion=_outcome(list(b.values())),
        shared_previous_completion=_outcome([a[k] for k in eligible_common]),
        shared_current_completion=_outcome([b[k] for k in eligible_common]),changed_outcomes=changes,
        setting_differences={k:[before.get(k),after.get(k)] for k in sorted(set(before)|set(after))
                             if k != 'out_dir' and before.get(k) != after.get(k)},
        interpretation=('Paired episode outcomes are available for the shared cases only.' if set(a)!=set(b) else
                        'Same episode IDs; compare recorded settings and scene provenance before attributing changes.'))


def _display(rate):
    return f'{rate["correct"]}/{rate["total"]} ({100*rate["rate"]:.1f}%)' if rate['total'] else 'N/A'


def render_evaluation_summary(summary, detailed=True):
    s = summary
    lines=['# Evaluation summary','',
        f'**Policy completion: {_display(s["policy_completion"])}**' if s['evaluation_kind']!='expert replay validation'
        else f'**Expert replay completion: {_display(s["replay"]["expert_task_completion"])}**',
        f'Coverage: {s["coverage"]["processed"]}/{s["coverage"]["requested"]} episodes processed; '
        f'complete={s["complete"]}; expert-completed before handover={s["expert_completed_before_handover"]}.']
    if s['settings'].get('privileged_context'):
        lines.extend(['','**PRIVILEGED ORACLE DIAGNOSTIC — not deployment performance.** '
            'Ground-truth task identity selects native task tokens; future demonstration observations may be used. '
            'Correct decoder identity does not guarantee perfect complete context.'])
    split=s['exposure']['split_membership']
    lines.append('Split: '+(', '.join(f'{split.get(k,0)} {label}' for k,label in
        [('train_episodes','training'),('development_episodes','development'),('confirmation_episodes','confirmation')])
        if split is not None else s['exposure']['selection']))
    if split and split.get('train_episodes',0):
        lines.append('Overall results include episodes used in training.')
    prefix=s['settings']['expert_prefix_fraction']
    if prefix is not None:
        modes=s['settings']['task_context_modes']
        mode=next(iter(modes)) if len(modes)==1 else ', '.join(f'{k} ({v})' for k,v in modes.items()) or 'unrecorded'
        lines.append(f'Setup: {100*prefix:g}% expert prefix; context={mode}; '
                     f'{s["settings"]["max_steps"]} steps; seed={s["settings"]["seed"]}.')
    if s['settings'].get('stop_on_task_success'):
        lines.append('Episodes end on the first task success; max-steps is a limit. Later checkpoint metrics exclude episodes that already ended.')
    if s['settings'].get('task_snapshot_selection') == 'highest_decoder_score':
        lines.append('Task tokens retain the highest decoder score observed from episode start; only strictly higher scores replace the snapshot. Execution continues updating.')
    recognition=s['task_recognition']
    if recognition['episodes_with_estimates']:
        separate=any(values['inferred']!=values['policy'] for values in recognition['stages'].values())
        lines.extend(['','| Task recognition | Student estimate |'+(' Actor-used context |' if separate else ''),
                         '|---|---:|'+('---:|' if separate else '')])
        for point,values in recognition['stages'].items():
            lines.append(f'| {POINT_LABELS[point]} | {_display(values["inferred"])} |'+
                         (f' {_display(values["policy"])} |' if separate else ''))
        lines.extend(['',f'Correct handover estimate: successful rollouts '
            f'{_display(recognition["successful_policy_with_correct_handover"])}; unsuccessful '
            f'{_display(recognition["failed_policy_with_correct_handover"])}.',
            f'Correct at handover, incorrect at final frame: {len(recognition["correct_to_incorrect"])} episodes.'])
        first=s['trajectory']['at_first_success'].get('inferred')
        if first and first['total']:
            lines.append(f'Correct estimate before the first successful action: {_display(first)} '
                         f'({first["total"]}/{s["trajectory"]["success_episodes"]} successful episodes with usable step logs).')
        elif s['trajectory']['success_episodes']:
            lines.append('Task recognition at first success: unavailable without usable step logs.')
        if s['trajectory']['final_beyond_demonstration']:
            lines.append(f'Final frame is beyond demonstration duration in {s["trajectory"]["final_beyond_demonstration"]} episodes; '
                         'late recognition can include post-completion behavior.')
        if recognition['unknown_task_episodes']:
            lines.append(f'{recognition["unknown_task_episodes"]} episodes have task names outside the decoder vocabulary; excluded from task accuracy.')
    elif s['evaluation_kind']!='expert replay validation':
        lines.extend(['','Task recognition: no estimates recorded.'])
    replay=s['replay']
    lines.extend(['',f'Expert replay checks: {replay["passed"]} pass, {replay["failed"]} fail, {replay["unrecorded"]} unrecorded; '
                  f'{replay["actual_expert_task_failures"]} actual expert task failures.',
                  f'Policy completion on failed replay checks: {_display(replay["policy_completion_on_failed_checks"])}.'])
    zero_tasks=[f'{name} (0/{values["policy_completion"]["total"]})' for name,values in s['per_task'].items()
                if values['policy_completion']['total'] and not values['policy_completion']['correct']]
    if zero_tasks:
        lines.append('Tasks with no completions: '+'; '.join(zero_tasks)+'.')
    if detailed and s['per_task']:
        lines.extend(['','## Results by task','',
            '| Recorded task | Policy completion | Expert replay completion | Task ID at handover | Task ID at final frame |',
            '|---|---:|---:|---:|---:|'])
        for task,values in s['per_task'].items():
            lines.append(f'| {task} | {_display(values["policy_completion"])} | '
                         f'{_display(values["expert_completion"])} | '
                         f'{_display(values["recognition"]["handover"])} | {_display(values["recognition"]["final"])} |')
        lines.extend(['','Expert replay completion uses the task-success predicate, independently of pose-replay checks. '
                      'Only recorded expert outcomes enter its denominator; missing outcomes are excluded (N/A if none). '
                      'Policy completion counts policy-eligible episodes.'])
        if recognition['handover_mistakes']:
            lines.extend(['','## Incorrect task estimates at handover','',
                '| Episode | Recorded task | Predicted task | Policy completes task |','|---|---|---|---|'])
            for error in recognition['handover_mistakes']:
                lines.append(f'| {error["episode"]} | {error["actual_task"]} | {error["predicted_task"]} | '
                             f'{"Yes" if error["policy_success"] else "No"} |')
    if s.get('comparison'):
        c=s['comparison']
        lines.extend(['','## Comparison with previous report','',
            f'Previous completion: {_display(c["previous_completion"])}; current: {_display(c["current_completion"])}.',
            f'Same episode set: {c["same_episode_set"]}; shared episodes: {c["shared_episodes"]}.',
            f'On shared policy-eligible episodes only: previous {_display(c["shared_previous_completion"])}, '
            f'current {_display(c["shared_current_completion"])}; outcome changes={len(c["changed_outcomes"])}.',c['interpretation']])
        if detailed and c['setting_differences']:
            lines.extend(['','Changed settings: '+', '.join(c['setting_differences'])+'.'])
    if recognition['episodes_with_estimates'] and detailed:
        lines.extend(['',('Task decoder scores select the task snapshot in this evaluation; probabilities are uncalibrated. ' if
            s['settings'].get('task_decoder_controls_snapshot') else 'Task-name decoding is diagnostic; probabilities are uncalibrated. ')+
            'Recognition alone does not establish action competence or a causal reason for failure.'])
    if s['issues']:
        lines.extend(['','Data checks:']+[f'- {issue}' for issue in s['issues']])
    return '\n'.join(lines)+'\n'


def write_evaluation_summary(result_path, output_dir=None, compare_path=None):
    result_path=Path(result_path)
    source=result_path.read_bytes()
    report=json.loads(source)
    summary=build_evaluation_summary(report,result_path.parent)
    summary['source']=dict(path=str(result_path),sha256=hashlib.sha256(source).hexdigest())
    if compare_path is not None:
        compare_path=Path(compare_path)
        raw=compare_path.read_bytes()
        summary['comparison']=compare_evaluations(report,json.loads(raw))
        summary['comparison']['source']=dict(path=str(compare_path),sha256=hashlib.sha256(raw).hexdigest())
    output=Path(output_dir) if output_dir is not None else result_path.parent
    targets=[output/'summary.json',output/'summary.md']
    if any(path.resolve()==result_path.resolve() or
           (compare_path is not None and path.resolve()==compare_path.resolve()) for path in targets):
        raise ValueError('Summary output would overwrite an evaluation input')
    output.mkdir(parents=True,exist_ok=True)
    targets[0].write_text(json.dumps(summary,indent=2)+'\n')
    targets[1].write_text(render_evaluation_summary(summary))
    return summary,targets
