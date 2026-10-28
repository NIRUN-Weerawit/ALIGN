"""Task diagnostics must not leak GT, alter actions, or conceal frozen snapshots."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from eval.task_estimation import (SCHEMA,TaskLabelEstimator,TaskEstimationTrace,
    context_fingerprint,summarize_task_estimation)


def write_estimator(path,branch=None,token_shape=(4,2)):
    width=int(np.prod(token_shape))
    weight=np.zeros((3,width));weight[0,0]=-2.;weight[1,0]=2.;weight[2,1]=2.
    metadata=dict(schema=SCHEMA,training_only=True,confirmation_opened=False,
        context_fingerprint=context_fingerprint(branch) if branch is not None else 'test',
        fitting_episodes=['train'],fitting_queries=[dict(fraction=.4)])
    np.savez_compressed(path,metadata=np.asarray(json.dumps(metadata)),
        task_names=np.asarray(['bowl on plate','bowl on stove','bowl on cabinet']),
        latent_mean=np.zeros(token_shape),latent_std=np.ones(token_shape),
        feature_mean=np.zeros(width),feature_scale=np.ones(width),weight=weight,bias=np.zeros(3))
    return TaskLabelEstimator(path)


def test_task_estimates_do_not_depend_on_ground_truth_and_unknown_tasks_are_not_scored(tmp_path):
    estimator=write_estimator(tmp_path/'probe.npz')
    tokens=torch.zeros(1,4,2);tokens[0,0,0]=1.
    correct=TaskEstimationTrace(estimator,'bowl on stove',2,4)
    wrong=TaskEstimationTrace(estimator,'bowl on plate',2,4)
    unknown=TaskEstimationTrace(estimator,'an unseen task',2,4)
    for trace in (correct,wrong,unknown):trace.observe(0,tokens,tokens)
    predicted=[t.records[0]['inferred']['predicted_task'] for t in (correct,wrong,unknown)]
    assert predicted==['bowl on stove']*3
    assert correct.records[0]['inferred']['correct'] is True
    assert wrong.records[0]['inferred']['correct'] is False
    assert unknown.records[0]['inferred']['correct'] is None
    assert not unknown.summary()['actual_task_known']
    assert unknown.summary()['observation_accuracy']['all']['inferred']['accuracy'] is None


def test_context_binding_rejects_a_different_encoder(tmp_path):
    branch=nn.Linear(2,2);branch.config=dict(width=2)
    estimator=write_estimator(tmp_path/'probe.npz',branch)
    estimator.validate_branch(branch)
    with torch.no_grad():branch.weight[0,0]+=1
    with pytest.raises(ValueError,match='does not match'):
        estimator.validate_branch(branch)


def test_snapshot_and_current_estimates_are_reported_separately_and_raw_tokens_saved(tmp_path):
    estimator=write_estimator(tmp_path/'probe.npz')
    trace=TaskEstimationTrace(estimator,'bowl on stove',1,3)
    inferred=torch.zeros(1,4,2);inferred[0,0,0]=1
    snapshot=-inferred
    for t in range(3):trace.observe(t,inferred,snapshot)
    summary=trace.summary()
    assert summary['checkpoints']['handover']['inferred']['correct'] is True
    assert summary['checkpoints']['handover']['policy']['correct'] is False
    trace.save(tmp_path/'trace')
    record=json.loads((tmp_path/'trace.json').read_text())
    assert len(record['observations'])==3 and record['summary']['checkpoints']['last_reference']['step']==2
    with np.load(tmp_path/'trace.npz',allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['inferred_tokens'][0],inferred[0])
        assert arrays['probabilities'].shape==(3,2,3)
    episode=dict(task_estimation=summary,replay_check=dict(valid=False,task_success=False),success=False)
    aggregate=summarize_task_estimation([episode])
    row=aggregate['groups']['expert_replay_fail']['checkpoints']['handover']
    assert row['inferred']['correct']==1 and row['policy']['correct']==0
    assert summarize_task_estimation([dict(status='replay only')]) is None


def test_wrong_shape_and_nonfinite_observations_are_rejected(tmp_path):
    estimator=write_estimator(tmp_path/'probe.npz')
    with pytest.raises(ValueError,match='finite'):
        estimator.predict(np.full((1,4,2),np.nan))
    with pytest.raises(ValueError,match='Expected'):
        estimator.predict(np.zeros((1,5,2)))


def test_online_evaluator_is_observational_and_keeps_inferred_and_used_tokens_separate(monkeypatch,tmp_path):
    import eval.eval_libero_v4_trajectory as evaluator

    class Env:
        def reset(self):
            self.position=np.zeros(3,dtype='f4');return self.obs()
        def obs(self):
            return dict(robot0_eef_pos=self.position.copy(),robot0_eef_quat=np.array([0,0,0,1.]))
        def step(self,action):
            self.position+=action[:3];return self.obs(),0.,False,{}
        def check_success(self):return False

    class Model:
        history_size=1;intention_encoder=None;use_memory_bank=False;text_condition=None;head_type='diffusion'
        def __init__(self):
            self.task_context=SimpleNamespace(freeze_task=self.freeze)
        def freeze(self,cache):cache.snapshot=cache.last_task.clone()
        def encode_step(self,frames,state,cache,produce_intent=False):
            cache=cache or SimpleNamespace(t=0,snapshot=None)
            cache.t+=1;cache.last_task=state.new_full((1,4,2),cache.t)
            task=cache.last_task if cache.snapshot is None else cache.snapshot
            payload=torch.cat([task,state.new_full((1,1,2),cache.t)],1)
            return state[:,:2],state,state[:,:1],cache,payload
        def condition_actions(self,visual,state,intent,**kwargs):return visual,state,intent
        def sample_actions(self,*args,**kwargs):return torch.randn(1,4,7)
    monkeypatch.setattr(evaluator,'get_sim_frame',lambda *a,**k:np.zeros((4,4,3),'u1'))
    estimator=write_estimator(tmp_path/'probe.npz')
    args=dict(device=torch.device('cpu'),expert_actions=np.zeros((6,7),'f4'),
        chunk_size=4,max_steps=6,switch_at=.5,task_context_mode='handover',use_camera=['image'],record_frames=False)
    torch.manual_seed(4)
    original=evaluator.run_model_in_sim(Env(),Model(),**args)
    state=torch.random.get_rng_state().clone()
    torch.manual_seed(4)
    model=Model()
    observed=evaluator.run_model_in_sim(Env(),model,**args,task_estimator=estimator,actual_task_name='bowl on plate')
    np.testing.assert_array_equal(original['executed_actions'],observed['executed_actions'])
    torch.testing.assert_close(torch.random.get_rng_state(),state)
    trace=observed['task_estimation_trace']
    assert observed['task_estimation']['observations']==6
    assert not np.array_equal(trace.inferred_tokens[-1],trace.policy_tokens[-1])
    np.testing.assert_array_equal(trace.policy_tokens[3],trace.policy_tokens[-1])
    np.testing.assert_array_equal(np.asarray(trace.execution_tokens),np.repeat(np.arange(1,7)[:,None],2,axis=1))
    assert original['task_estimation'] is None


def test_execution_trace_logging_preserves_native_values_without_changing_task_scores(tmp_path):
    estimator=write_estimator(tmp_path/'probe.npz')
    task=torch.ones(1,4,2);execution=torch.tensor([[[3.,4.]]])
    trace=TaskEstimationTrace(estimator,'bowl on stove',1,3)
    trace.observe(0,task,task,execution_token=execution)
    trace.observe(1,task,task,execution_token=execution+1)
    trace.save(tmp_path/'execution')
    with np.load(tmp_path/'execution.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['execution_tokens'],[[3.,4.],[4.,5.]])
        np.testing.assert_allclose(f['probabilities'][:,0],estimator.probabilities(task).repeat(2,axis=0))
    with pytest.raises(ValueError,match='every observation'):
        trace.observe(2,task,task)
    with pytest.raises(ValueError,match='finite execution'):
        TaskEstimationTrace(estimator,'bowl on stove',1,3).observe(0,task,task,execution_token=torch.zeros(1,1,3))


def test_peak_snapshot_replaces_only_on_strict_score_improvement_and_is_episode_local():
    from eval.task_estimation import PeakTaskSnapshot
    probabilities=np.array([[.6,.3,.1],[.95,.03,.02],[.15,.7,.15],[.02,.95,.03],[.01,.02,.97],[.1,.8,.1]])
    estimator=SimpleNamespace(task_names=['A','B','C'],probabilities=lambda x:probabilities[np.asarray(x)[:,0,0].astype(int)])
    peak=PeakTaskSnapshot(estimator)
    selections=[];used=[]
    for step in range(6):
        token=np.full((1,4,2),step,dtype='f4')
        selected,record=peak.observe(step,token)
        used.append(int(selected[0,0,0]));selections.append(record)
        token[:]=-100  # Stored snapshots must not alias caller arrays.
    assert used==[0,1,1,1,4,4]
    assert [s['selected_step'] for s in selections]==[0,1,1,1,4,4]
    assert [s['selected_score'] for s in selections]==[.6,.95,.95,.95,.97,.97]
    assert [s['updated'] for s in selections]==[True,True,False,False,True,False]
    assert peak.summary()['ground_truth_used'] is False
    assert PeakTaskSnapshot(estimator).observe(0,np.zeros((1,4,2)))[1]['updated']


def test_peak_selection_provenance_is_saved_per_observation(tmp_path):
    estimator=write_estimator(tmp_path/'probe.npz')
    from eval.task_estimation import PeakTaskSnapshot
    peak=PeakTaskSnapshot(estimator);trace=TaskEstimationTrace(estimator,'bowl on stove',1,3)
    for step,value in enumerate([1.,2.,1.]):
        token=torch.full((1,4,2),value)
        selected,selection=peak.observe(step,token)
        trace.observe(step,token,selected,snapshot_selection=selection)
    trace.save(tmp_path/'selected')
    assert trace.summary()['diagnostic_only'] is False
    assert trace.summary()['estimator_is_policy_input']
    with np.load(tmp_path/'selected.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['snapshot_steps'],[0,1,1])
        assert np.all(np.diff(f['snapshot_scores'])>=0)
