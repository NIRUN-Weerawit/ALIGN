"""Reporting must preserve denominators, GT coverage and trace provenance."""
import copy
import hashlib
import json

import pytest

from eval.evaluation_summary import (build_evaluation_summary,compare_evaluations,
    render_evaluation_summary,write_evaluation_summary)


def episode(key='a',success=True,known=True):
    def record(step,correct):
        prediction=dict(predicted_task='task' if correct else 'other task',correct=correct,
                        predicted_group=0 if correct else 1,top1_probability=.8)
        return dict(step=step,phase='expert' if step<2 else 'policy',
                    inferred=prediction,policy=prediction.copy())
    observations=[record(t,t in (2,3,4)) for t in range(6)]
    context=dict(actual_task='task',actual_task_known=known,actual_group=0,observations=6,
        estimator_fingerprint='bound',checkpoints=dict(initial=observations[0],handover=observations[2],
            last_reference=observations[4],final=observations[5]))
    row=dict(episode=key,task_name='task',status='evaluated',policy_eligible=True,
        success=success,episode_success=success,success_before_handover=False,
        switch_step=2,n_steps=6,reference_steps=5,first_success_step=3 if success else None,
        task_context_mode='dynamic',task_estimation=context,
        replay_check=dict(valid=False,task_success=False))
    return row,observations


def report(rows):
    return dict(episodes=rows,evaluation_kind='policy rollout',evaluation_complete=True,
        requested_episodes=len(rows),success_count=sum(bool(r.get('success')) for r in rows if r.get('policy_eligible')),
        evaluation_args=dict(seed=42,switch_at=.4,max_steps=6),
        evaluation_protocol=dict(replay_filters_policy_episodes=False,
            episode_selection=dict(selection_mode='development',
                split_membership=dict(train_episodes=0,development_episodes=len(rows),confirmation_episodes=0))))


def test_oracle_summary_preserves_privileged_kind_and_warns_about_gt():
    row,_=episode()
    r=report([row]);r['evaluation_kind']='privileged oracle-context diagnostic'
    r['evaluation_protocol']['task_oracle']=dict(mode='repair',privileged=True,ground_truth_is_model_input=True)
    s=build_evaluation_summary(r)
    assert s['evaluation_kind']=='privileged oracle-context diagnostic'
    assert s['settings']['privileged_context']
    text=render_evaluation_summary(s)
    assert 'PRIVILEGED ORACLE DIAGNOSTIC' in text
    assert '**Policy completion: 1/1 (100.0%)**' in text


def test_policy_and_overall_denominators_include_failed_replays_but_separate_expert_completions():
    a,_=episode('a');b,_=episode('b',False);c,_=episode('c',False)
    c.update(policy_eligible=False,episode_success=True,success_before_handover=True)
    s=build_evaluation_summary(report([a,b,c]))
    assert s['policy_completion']==dict(correct=1,total=2,rate=.5)
    assert s['overall_completion']['correct']==2 and s['overall_completion']['total']==3
    assert s['expert_completed_before_handover']==1
    assert s['replay']['policy_completion_on_failed_checks']==s['policy_completion']


def test_replay_only_has_no_policy_score_and_no_decoder_is_handled():
    a,_=episode('a');b,_=episode('b',False)
    for row in (a,b):row.update(status='replay validation only',policy_eligible=False,task_estimation=None)
    a['replay_check'].update(valid=True,task_success=True)
    r=report([a,b]);r.update(evaluation_kind='expert replay validation',success_count=0)
    s=build_evaluation_summary(r)
    assert s['policy_completion']['total']==0 and s['policy_completion']['rate'] is None
    assert s['replay']['expert_task_completion']['correct']==1
    assert s['per_task']['task']['expert_completion']==dict(correct=1,total=2,rate=.5)
    assert s['per_task']['task']['policy_completion']['rate'] is None
    assert '**Expert replay completion: 1/2 (50.0%)**' in render_evaluation_summary(s)


def test_per_task_expert_completion_uses_task_outcomes_not_pose_checks_or_policy_eligibility():
    a,_=episode('a');b,_=episode('b',False);c,_=episode('c');d,_=episode('d')
    a['replay_check'].update(valid=False,task_success=True)
    a.update(policy_eligible=False,success_before_handover=True)
    c['replay_check']={}
    d.update(task_name='unrecorded task',replay_check={})
    s=build_evaluation_summary(report([a,b,c,d]))
    assert s['per_task']['task']['expert_completion']==dict(correct=1,total=2,rate=.5)
    assert s['per_task']['task']['policy_completion']==dict(correct=1,total=2,rate=.5)
    assert s['per_task']['unrecorded task']['expert_completion']==dict(correct=0,total=0,rate=None)
    rendered=render_evaluation_summary(s)
    assert '| Policy completion | Expert replay completion |' in rendered
    assert '| task | 1/2 (50.0%) | 1/2 (50.0%) |' in rendered
    assert '| unrecorded task | 1/1 (100.0%) | N/A |' in rendered


def test_unknown_labels_and_missing_decoder_do_not_become_incorrect_predictions():
    a,_=episode('a',known=False);b,_=episode('b');b['task_estimation']=None
    s=build_evaluation_summary(report([a,b]))
    assert s['task_recognition']['unknown_task_episodes']==1
    assert s['task_recognition']['stages']['handover']['inferred']['total']==0
    assert '| Handover | N/A |' in render_evaluation_summary(s)


def test_late_wrong_label_is_distinguished_from_correct_label_at_success_and_stale_logs_are_excluded(tmp_path):
    a,records=episode('a');b,b_records=episode('b');c,_=episode('c')
    (tmp_path/'a_task_estimates.json').write_text(json.dumps(dict(summary=a['task_estimation'],observations=records)))
    stale=copy.deepcopy(b['task_estimation']);stale['estimator_fingerprint']='old run'
    (tmp_path/'b_task_estimates.json').write_text(json.dumps(dict(summary=stale,observations=b_records)))
    s=build_evaluation_summary(report([a,b,c]),tmp_path)
    assert s['task_recognition']['stages']['final']['inferred']['correct']==0
    trajectory=s['trajectory']
    assert trajectory['at_first_success']['inferred']==dict(correct=1,total=1,rate=1.)
    assert trajectory['success_episodes']==3 and trajectory['sidecars_loaded']==1
    assert trajectory['invalid_sidecars'][0]['episode']=='b' and trajectory['missing_sidecars']==['c']
    assert trajectory['first_errors_after_handover'][0]['after_task_completion'] is True
    assert '1/3 successful episodes with usable step logs' in render_evaluation_summary(s)


def test_frozen_actor_context_is_reported_separately_from_the_current_student_estimate():
    row,_=episode();row['task_estimation']['checkpoints']['final']['policy']['correct']=True
    s=build_evaluation_summary(report([row]))
    assert s['task_recognition']['stages']['final']['inferred']['correct']==0
    assert s['task_recognition']['stages']['final']['policy']['correct']==1
    assert 'Actor-used context' in render_evaluation_summary(s)


def test_missing_episodes_and_inconsistent_reported_counts_are_visible():
    row,_=episode();r=report([row]);r.update(requested_episodes=3,success_count=99)
    s=build_evaluation_summary(r)
    assert not s['complete'] and len(s['issues'])==2
    assert s['policy_completion']['correct']==1
    with pytest.raises(ValueError,match='Duplicate'):
        build_evaluation_summary(report([row,row]))
    with pytest.raises(ValueError,match='episodes list'):
        build_evaluation_summary({})


def test_comparison_identifies_different_episode_sets_and_uses_shared_eligible_cases_only():
    a,_=episode('a',False);b,_=episode('b');c,_=episode('c')
    comparison=compare_evaluations(report([a,c]),report([a,b]))
    assert not comparison['same_episode_set'] and comparison['shared_episodes']==1
    assert comparison['previous_completion']['correct']==comparison['current_completion']['correct']==1
    assert comparison['shared_current_completion']==dict(correct=0,total=1,rate=0.)
    current_a=copy.deepcopy(a);current_a.update(policy_eligible=False,success_before_handover=True,episode_success=True)
    comparison=compare_evaluations(report([current_a,c]),report([a,b]))
    assert comparison['shared_episodes']==1 and comparison['shared_policy_eligible_episodes']==0
    assert comparison['shared_current_completion']['total']==0


def test_writer_preserves_raw_inputs_and_records_their_hashes(tmp_path):
    row,_=episode();raw=json.dumps(report([row])).encode();path=tmp_path/'result.json';path.write_bytes(raw)
    summary,targets=write_evaluation_summary(path)
    assert path.read_bytes()==raw
    assert summary['source']['sha256']==hashlib.sha256(raw).hexdigest()
    assert {p.name for p in targets}=={'summary.md','summary.json'}
    assert json.loads(targets[0].read_text())==summary
    with pytest.raises(ValueError,match='overwrite'):
        write_evaluation_summary(path,compare_path=tmp_path/'summary.json')


def test_peak_snapshot_summary_records_control_dependency_and_success_termination():
    row,_=episode();r=report([row])
    r['evaluation_protocol'].update(stop_on_task_success=True,task_snapshot_selection='highest_decoder_score',
        task_estimator=dict(used_for_task_snapshot_selection=True))
    s=build_evaluation_summary(r);text=render_evaluation_summary(s)
    assert s['settings']['stop_on_task_success']
    assert 'strictly higher scores replace' in text
    assert 'Task decoder scores select the task snapshot' in text
    assert 'Task-name decoding is diagnostic;' not in text
