"""Evaluation must preserve held-out splits, causal handover and replay validity."""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from eval.libero_protocol import (select_evaluation_episodes, saved_development_episodes,
    context_handover_mode, assess_expert_replay, load_initial_sim_state)


def config():
    return dict(cameras=['image'], num_cameras=1, chunk_size=4,
        task_context_split=dict(train_episodes=['ep_0'], development_episodes=['ep_1'],
            confirmation_episodes=['ep_2'], manifest_sha256='audited'))


def test_saved_development_is_default_and_explicit_lists_cannot_escape_it(tmp_path):
    cfg=config()
    keys, record=select_evaluation_episodes(['ep_0','ep_1','ep_2'],cfg)
    assert keys==['ep_1'] and record['confirmation_opened'] is False
    path=tmp_path/'wrong.txt';path.write_text('ep_0\nep_2\n')
    with pytest.raises(ValueError,match='training or confirmation'):
        select_evaluation_episodes(['ep_0','ep_1','ep_2'],cfg,path)
    path.write_text('unknown\n')
    with pytest.raises(ValueError,match='Unknown'):
        select_evaluation_episodes(['ep_0','ep_1','ep_2'],cfg,path)
    path.write_text('ep_1\nep_1\n')
    with pytest.raises(ValueError,match='unique'):
        select_evaluation_episodes(['ep_0','ep_1','ep_2'],cfg,path)


def test_split_integrity_and_dataset_coverage_are_checked():
    cfg=config();cfg['task_context_split']['confirmation_episodes']=['ep_0']
    with pytest.raises(ValueError,match='overlap'):
        saved_development_episodes(cfg)
    with pytest.raises(ValueError,match='Dataset episodes differ'):
        select_evaluation_episodes(['ep_0','ep_1','ep_2','ep_3'],config())
    with pytest.raises(ValueError,match='missing saved'):
        select_evaluation_episodes(['ep_0','ep_2'],config())
    with pytest.raises(ValueError,match='positive'):
        select_evaluation_episodes(['ep_0','ep_1','ep_2'],config(),limit=0)


def test_comparison_file_matches_cases_and_reports_training_overlap(tmp_path):
    path=tmp_path/'piper.txt';path.write_text('ep_0\nep_2\n')
    keys,record=select_evaluation_episodes(['ep_0','ep_1','ep_2'],config(),path,selection='comparison')
    assert keys==['ep_0','ep_2']
    assert record['source']=='explicit comparison episode file'
    assert record['split_membership']==dict(train_episodes=1,development_episodes=0,confirmation_episodes=1)
    assert not record['held_out_for_this_checkpoint'] and record['confirmation_opened']
    with pytest.raises(ValueError,match='explicit episode file'):
        select_evaluation_episodes(['ep_0','ep_1','ep_2'],config(),selection='comparison')
    keys,record=select_evaluation_episodes(['ep_0','ep_1','ep_2'],config(),path,limit=1,selection='comparison')
    assert keys==['ep_0'] and not record['confirmation_opened']


def test_legacy_saved_episode_split_is_preferred_to_random_reconstruction():
    keys, _=select_evaluation_episodes(['a','b','c'],dict(val_episodes=['b']))
    assert keys==['b']
    assert select_evaluation_episodes(['c','a'],{})[0]==['a','c']


@pytest.mark.parametrize('requested,step,expected',[
    ('auto',0,'dynamic'),('auto',3,'freeze'),('dynamic',3,'dynamic'),('freeze',3,'freeze'),('freeze',0,'freeze'),('handover',3,'handover')])
def test_explicit_handover_mode(requested,step,expected):
    assert context_handover_mode(requested,step)==expected


def test_cannot_freeze_an_unobserved_human_prefix():
    with pytest.raises(ValueError,match='without a human prefix'):
        context_handover_mode('handover',0)


class Env:
    def __init__(self, success_at=3, done_at=None, **kwargs):
        self.success_at=success_at;self.done_at=done_at
        self.restore_calls=0;self.steps=[]
    def reset(self):
        self.t=0;self.position=np.zeros(3,dtype='f4');return self.obs()
    def obs(self):
        return dict(robot0_eef_pos=self.position.copy(),robot0_eef_quat=np.array([0,0,0,1.]))
    def get_sim_state(self):
        return self.position.copy()
    def set_init_state(self,state):
        self.restore_calls+=1;self.position=state.copy();return self.obs()
    def step(self, action):
        self.steps.append(action.copy());self.position+=action[:3];self.t+=1
        return self.obs(),0.,self.t==self.done_at,{}
    def check_success(self):
        return self.success_at is not None and self.t>=self.success_at
    def seed(self,seed):pass
    def close(self):pass


class StreamModel:
    history_size=1
    intention_encoder=None
    use_memory_bank=False
    text_condition=None
    head_type='diffusion'
    def __init__(self):
        self.payloads=[];self.freeze_calls=0
        self.task_context=SimpleNamespace(freeze_task=self.freeze)
    def freeze(self,cache):
        self.freeze_calls+=1;cache.snapshot=cache.last.clone()
    def encode_step(self,frames,state,cache,produce_intent=False):
        cache=cache or SimpleNamespace(t=0,snapshot=None)
        cache.t+=1;cache.last=state.new_full((1,4,2),cache.t);cache.last_task=cache.last
        task=cache.last if cache.snapshot is None else cache.snapshot
        execution=state.new_full((1,1,2),cache.t)
        payload=torch.cat([task,execution],1)
        self.payloads.append(payload.clone())
        return state[:,:2],state,state[:,:1],cache,payload
    def condition_actions(self,visual,state,intent,**kwargs):
        return visual,state,intent
    def sample_actions(self,*args,**kwargs):
        out=torch.zeros(1,4,7);out[:,:,6]=1;return out


@pytest.fixture
def evaluator(monkeypatch):
    import eval.eval_libero_v4_trajectory as module
    monkeypatch.setattr(module,'get_sim_frame',lambda *a,**k:np.zeros((4,4,3),dtype='u1'))
    return module


@pytest.mark.parametrize('switch,mode,freezes',[(0,'auto',0),(.5,'handover',1),(.5,'dynamic',0)])
def test_task_inference_updates_from_start_and_freezes_only_at_real_handover(evaluator,switch,mode,freezes):
    model=StreamModel()
    result=evaluator.run_model_in_sim(Env(success_at=None),model,torch.device('cpu'),np.zeros((6,7),'f4'),
        chunk_size=4,max_steps=6,switch_at=switch,task_context_mode=mode,use_camera=['image'],record_frames=False)
    assert model.freeze_calls==freezes
    assert result['n_steps']==6 and result['frames']==[]
    if freezes:
        torch.testing.assert_close(model.payloads[3][:,:4],model.payloads[5][:,:4])
    else:
        assert not torch.equal(model.payloads[0][:,:4],model.payloads[-1][:,:4])
    assert not torch.equal(model.payloads[0][:,-1],model.payloads[-1][:,-1])


def test_success_during_expert_prefix_is_not_attributed_to_policy(evaluator):
    result=evaluator.run_model_in_sim(Env(success_at=2),StreamModel(),torch.device('cpu'),np.zeros((8,7),'f4'),
        chunk_size=4,max_steps=8,switch_at=.75,task_context_mode='handover',use_camera=['image'])
    assert result['episode_success'] and result['success_before_handover']
    assert not result['success'] and not result['policy_eligible']


def test_oracle_changes_only_actor_task_context_at_handover(evaluator):
    from eval.task_context_oracle import OracleTaskSnapshot
    class Estimator:
        token_shape=(4,2)
        task_names=['task a','task b']
        name_to_group={'task a':0,'task b':1}
        metadata={'context_fingerprint':'test'}
        def probabilities(self,tokens):
            v=np.asarray(tokens.detach().cpu() if isinstance(tokens,torch.Tensor) else tokens)
            p=np.where(v[:,0,0]>0,.9,.1)
            return np.stack([1-p,p],1)
        def predict(self,tokens):
            return [dict(predicted_task=self.task_names[int(p.argmax())],predicted_group=int(p.argmax()),
                         top1_probability=float(p.max())) for p in self.probabilities(tokens)]
    class RecordingModel(StreamModel):
        def __init__(self):
            super().__init__();self.actor_payloads=[]
        def sample_actions(self,visual,state,intent,**kwargs):
            self.actor_payloads.append(intent.clone())
            return super().sample_actions()
    estimator=Estimator();model=RecordingModel()
    oracle=OracleTaskSnapshot(dict(tokens=np.full((4,2),-5.,dtype='f4'),task='task a',source_step=9),'repair',estimator)
    result=evaluator.run_model_in_sim(Env(success_at=None),model,torch.device('cpu'),np.zeros((8,7),'f4'),
        chunk_size=4,max_steps=6,switch_at=.4,task_context_mode='handover',task_estimator=estimator,
        actual_task_name='task a',task_context_oracle=oracle,use_camera=['image'],record_frames=False)
    assert all((v[:,:4]>0).all() for v in model.actor_payloads[:3])
    assert all((v[:,:4]==-5).all() for v in model.actor_payloads[3:])
    assert [v[0,-1,0].item() for v in model.actor_payloads]==[1,2,3,4,5,6]
    assert result['task_context_selection']['replaced']
    assert result['task_estimation']['ground_truth_is_model_input']


def test_horizon_done_is_not_task_success_and_expert_only_never_hands_over(evaluator):
    actions=np.zeros((3,7),'f4');actions[:,0]=.1
    result=evaluator.run_model_in_sim(Env(success_at=None,done_at=3),StreamModel(),torch.device('cpu'),actions,
        chunk_size=4,max_steps=5,switch_at=1,use_camera=['image'])
    assert not result['success'] and not result['episode_success'] and not result['policy_eligible']
    assert result['switch_step']==5
    np.testing.assert_array_equal(result['executed_actions'][3:,:6],0.)
    assert result['action_magnitude_model']==0.  # Open gripper is not pose motion.
    assert result['action_magnitude_dataset']==pytest.approx(.1)  # No padded tail in denominator.


def test_handover_uses_original_demonstration_length_not_truncated_budget(evaluator):
    with pytest.raises(ValueError,match='before requested handover'):
        evaluator.run_model_in_sim(Env(),StreamModel(),torch.device('cpu'),np.zeros((20,7),'f4'),
            max_steps=6,switch_at=.5,use_camera=['image'])


def test_replay_checks_pre_action_alignment_success_and_holds_gripper_without_video(evaluator):
    env=Env(success_at=5);actions=np.zeros((3,7),'f4');actions[:,0]=.1;actions[:,6]=1.
    poses=np.zeros((3,6),'f4');poses[:,0]=[0,.1,.2]
    result=evaluator.run_replay_in_sim(env,actions,poses,max_steps=5,record_frames=False,
                                      initial_sim_state=np.zeros(3,'f4'))
    assert result['frames']==[] and result['n_steps']==5 and result['success']
    assert env.restore_calls==1
    np.testing.assert_allclose(result['errors'],0,atol=1e-7)
    np.testing.assert_array_equal(np.asarray(env.steps)[3:,:6],0.)
    np.testing.assert_array_equal(np.asarray(env.steps)[:,6],-1.)
    assert assess_expert_replay(result,.02)['valid']
    result['errors'][0]=.1
    assert not assess_expert_replay(result,.02)['valid']
    result['errors'][:]=0.;result['success']=False
    assert not assess_expert_replay(result,.02)['valid']


@pytest.fixture
def saved_run(tmp_path):
    cfg=config();path=tmp_path/'checkpoint.pt'
    torch.save(dict(config=cfg),path)
    data=tmp_path/'libero_goal.h5'
    with h5py.File(data,'w') as f:
        for name in ['ep_0','ep_1','ep_2']:
            g=f.create_group(name);g.create_dataset('actions',data=np.zeros((6,7),'f4'))
            g.create_dataset('poses',data=np.zeros((6,6),'f4'))
            g.create_dataset('texts',data=json.dumps(['task']))
            g.create_group('frames').create_dataset('image',data=np.zeros((6,4,4,3),'u1'))
    return path,data,cfg


@pytest.mark.parametrize('replay_success',[True,False])
@pytest.mark.parametrize('replay_only',[False,True])
def test_main_evaluates_every_selected_episode_regardless_of_expert_replay(evaluator,monkeypatch,tmp_path,saved_run,replay_success,replay_only):
    path,data,cfg=saved_run;model=StreamModel();env=Env(success_at=3 if replay_success else None)
    monkeypatch.setattr(evaluator,'LIBERO_AVAILABLE',True)
    monkeypatch.setattr(evaluator,'OffScreenRenderEnv',lambda **k:env)
    def load(*args):
        assert not replay_only, 'Replay-only must not load the policy or DINO'
        return model,cfg
    monkeypatch.setattr(evaluator,'load_intention_model',load)
    monkeypatch.setattr(evaluator,'_try_load_libero_task_list',lambda *a:['task'])
    bddl=tmp_path/'task.bddl';bddl.touch()
    monkeypatch.setattr(evaluator,'get_bddl_path',lambda *a:str(bddl))
    out=tmp_path/'evaluation'
    evaluator.main(['--checkpoint',str(path),'--data',str(data),'--out-dir',str(out),
        '--max-steps','6','--switch-at','.4','--task-context-mode','handover','--no-video','--no-plot','--device','cpu']+
        (['--replay-only'] if replay_only else []))
    report=json.loads((out/'result.json').read_text())
    quick=json.loads((out/'summary.json').read_text())
    assert (out/'summary.md').exists()
    assert quick['policy_completion']['correct']==report['success_count']
    assert quick['coverage']['processed']==1
    assert report['evaluation_protocol']['episode_selection']['selected_episodes']==['ep_1']
    assert report['evaluation_complete']
    assert report['expert_replay_all_passed'] is replay_success
    evaluated=not replay_only
    assert report['n_evaluated_episodes']==int(evaluated)
    assert report['success_rate']==(('1/1' if replay_success else '0/1') if evaluated else None)
    assert report['episode_success_rate']==(('1/1' if replay_success else '0/1') if evaluated else None)
    assert report['episodes'][0]['replay_check']['valid'] is replay_success
    assert env.restore_calls==(2 if evaluated else 1)
    assert (out/'ep_1_rollout.npz').exists() is evaluated
    assert bool(model.payloads) is evaluated
    assert not list(tmp_path.glob('checkpoint.mujoco*.json'))


def test_preflight_rejects_old_validation_list_before_model_construction(evaluator,monkeypatch,tmp_path,saved_run):
    path,data,cfg=saved_run;wrong=tmp_path/'old.txt';wrong.write_text('ep_0\n')
    monkeypatch.setattr(evaluator,'load_intention_model',lambda *a:pytest.fail('must fail before DINO/model loading'))
    with pytest.raises(SystemExit):
        evaluator.main(['--checkpoint',str(path),'--data',str(data),'--val-episodes',str(wrong),'--preflight-only'])


def test_comparison_preflight_accepts_explicit_cases_and_labels_seen_episodes(evaluator,monkeypatch,tmp_path,saved_run):
    path,data,_=saved_run;cases=tmp_path/'cases.txt';cases.write_text('ep_0\nep_2\n')
    monkeypatch.setattr(evaluator,'load_intention_model',lambda *a:pytest.fail('preflight must not load model'))
    out=tmp_path/'comparison'
    evaluator.main(['--checkpoint',str(path),'--data',str(data),'--val-episodes',str(cases),
        '--episode-set','comparison','--n-episodes','2','--preflight-only','--out-dir',str(out)])
    selection=json.loads((out/'preflight.json').read_text())['protocol']['episode_selection']
    assert selection['selected_episodes']==['ep_0','ep_2']
    assert selection['split_membership']['train_episodes']==1
    assert selection['held_out_for_this_checkpoint'] is False


def test_policy_success_is_reported_even_when_expert_replay_failed(evaluator,monkeypatch,tmp_path,saved_run):
    path,data,cfg=saved_run
    class OpenGripperSuccess(Env):
        def check_success(self):
            return self.t>=3 and self.steps[-1][6]<0
    model=StreamModel();env=OpenGripperSuccess()
    monkeypatch.setattr(evaluator,'LIBERO_AVAILABLE',True)
    monkeypatch.setattr(evaluator,'OffScreenRenderEnv',lambda **k:env)
    monkeypatch.setattr(evaluator,'load_intention_model',lambda *a:(model,cfg))
    monkeypatch.setattr(evaluator,'_try_load_libero_task_list',lambda *a:['task'])
    bddl=tmp_path/'task.bddl';bddl.touch()
    monkeypatch.setattr(evaluator,'get_bddl_path',lambda *a:str(bddl))
    out=tmp_path/'all_episodes'
    evaluator.main(['--checkpoint',str(path),'--data',str(data),'--out-dir',str(out),
        '--max-steps','6','--switch-at','.4','--task-context-mode','handover','--no-video','--no-plot','--device','cpu'])
    report=json.loads((out/'result.json').read_text())
    assert not report['expert_replay_all_passed']
    assert report['evaluation_complete'] and report['n_evaluated_episodes']==1
    assert report['success_rate']=='1/1' and report['episode_success_rate']=='1/1'


def test_saved_split_export_uses_exact_keys_without_legacy_dataset_reconstruction(monkeypatch,tmp_path,saved_run):
    import scripts.save_val_episodes as export
    path,data,cfg=saved_run;out=tmp_path/'val.txt'
    monkeypatch.setattr(export,'ALIGNDataset',lambda *a,**k:pytest.fail('must use saved split'))
    export.main(['--checkpoint',str(path),'--data',str(data),'--output',str(out)])
    assert out.read_text()=='ep_1\n'


def test_missing_camera_and_pose_are_not_silently_replaced_with_black_frames_or_zeros():
    import eval.eval_libero_v4_trajectory as evaluator
    env=SimpleNamespace(_get_observations=lambda:dict(agentview_image=np.zeros((4,4,3),'u1')))
    with pytest.raises(ValueError,match='required camera'):
        evaluator.get_sim_frame(env,'wrist_image')
    with pytest.raises(KeyError):
        evaluator.get_sim_eef_pose({})
    with pytest.raises(ValueError,match='valid quaternion'):
        evaluator.get_sim_eef_pose(dict(robot0_eef_pos=np.zeros(3),robot0_eef_quat=np.zeros(4)))


def test_initial_state_sidecar_is_episode_bound_and_validated(tmp_path):
    sidecar=tmp_path/'states.npz';np.savez(sidecar,ep_1=np.arange(9.))
    state,source=load_initial_sim_state('ep_1',sidecar=sidecar)
    np.testing.assert_array_equal(state,np.arange(9.))
    assert source=='episode-keyed initial-state sidecar'
    with pytest.raises(ValueError,match='missing ep_2'):
        load_initial_sim_state('ep_2',sidecar=sidecar)
    with pytest.raises(ValueError,match='finite flat'):
        load_initial_sim_state('ep_1',embedded=np.array([np.nan]))


@pytest.mark.parametrize('flags',[
    ['--cameras','wrist_image'],
    ['--switch-at','0','--task-context-mode','handover'],
    ['--max-steps','1','--switch-at','.4'],
])
def test_invalid_evaluation_contract_fails_before_model_loading(evaluator,monkeypatch,tmp_path,saved_run,flags):
    path,data,_=saved_run
    monkeypatch.setattr(evaluator,'load_intention_model',lambda *a:pytest.fail('invalid protocol must fail before loading model'))
    with pytest.raises(SystemExit):
        evaluator.main(['--checkpoint',str(path),'--data',str(data),
            '--out-dir',str(tmp_path/'invalid'),'--preflight-only']+flags)


def test_success_ends_policy_and_replay_at_the_successful_action(evaluator):
    actions=np.zeros((8,7),'f4')
    policy_env=Env(success_at=4)
    timing=[]
    result=evaluator.run_model_in_sim(policy_env,StreamModel(),torch.device('cpu'),actions,
        chunk_size=4,max_steps=8,switch_at=.25,task_context_mode='handover',use_camera=['image'],record_frames=False,timing_log=timing)
    assert result['success'] and result['first_success_step']==3
    assert result['n_steps']==4 and len(policy_env.steps)==4 and len(timing)==4
    assert result['termination_reason']=='task_success'
    assert len(result['stored_actions'])==len(result['executed_actions'])==4
    replay_env=Env(success_at=2)
    replay=evaluator.run_replay_in_sim(replay_env,actions,max_steps=8,record_frames=False)
    assert replay['n_steps']==2 and len(replay_env.steps)==2
    assert len(replay['expert_actions'])==len(replay['sim_positions'])==2
    assert replay['first_success_step']==1 and replay['termination_reason']=='task_success'


def test_initial_success_never_executes_an_action(evaluator):
    for replay in (False,True):
        env=Env(success_at=0)
        if replay:
            result=evaluator.run_replay_in_sim(env,np.zeros((6,7),'f4'),max_steps=6,record_frames=False)
        else:
            result=evaluator.run_model_in_sim(env,StreamModel(),torch.device('cpu'),np.zeros((6,7),'f4'),
                chunk_size=4,max_steps=6,switch_at=0,task_context_mode='dynamic',use_camera=['image'],record_frames=False)
            assert not result['policy_eligible'] and result['success_before_handover']
        assert result['n_steps']==0 and not env.steps
        assert result['first_success_step']==-1 and result['termination_reason']=='task_success'


def test_success_during_expert_prefix_stops_before_handover(evaluator):
    env=Env(success_at=2)
    result=evaluator.run_model_in_sim(env,StreamModel(),torch.device('cpu'),np.zeros((8,7),'f4'),
        chunk_size=4,max_steps=8,switch_at=.75,task_context_mode='handover',use_camera=['image'],record_frames=False)
    assert result['n_steps']==2 and not result['policy_eligible']
    assert result['success_before_handover'] and not result['success_after_handover'] and not result['success']


def test_peak_task_snapshot_updates_before_and_after_handover_without_freezing_execution(evaluator,tmp_path):
    from tests.test_task_estimation import write_estimator
    estimator=write_estimator(tmp_path/'probe.npz')
    probs=np.array([[.6,.3,.1],[.95,.03,.02],[.15,.7,.15],[.02,.95,.03],[.01,.02,.97],[.1,.8,.1]])
    estimator.probabilities=lambda x:probs[np.asarray(x)[:,0,0].astype(int)-1]
    class Bank:
        _has_cognitive=False
        def reset(self,**kwargs):self.writes=[]
        def observe_only(self,visual,state,cognitive,observed_mask=None,timestamp=None):
            self.writes.extend(int(t) for t in timestamp)
    class CaptureModel(StreamModel):
        def __init__(self,memory=False):
            super().__init__();self.used=[];self.use_memory_bank=memory
            self.memory_module=Bank() if memory else None
        def condition_actions(self,visual,state,intent,**kwargs):
            if self.use_memory_bank:
                self.memory_module.observe_only(visual[:,-1],state[:,-1],None,**kwargs)
            self.used.append(intent.clone());return visual,state,intent
    for mode,memory in [('freeze',False),('auto',False),('freeze',True)]:
        model=CaptureModel(memory)
        result=evaluator.run_model_in_sim(Env(success_at=None),model,torch.device('cpu'),np.zeros((6,7),'f4'),
            chunk_size=4,max_steps=6,switch_at=.5,task_context_mode=mode,action_horizon=2,use_camera=['image'],
            record_frames=False,task_estimator=estimator,actual_task_name='bowl on plate')
        trace=result['task_estimation_trace']
        assert result['task_context_selection']['selected_step']==4
        assert [u['step'] for u in result['task_context_selection']['updates']]==[0,1,4]
        np.testing.assert_array_equal(np.asarray(trace.policy_tokens)[:,0,0],[1,2,2,2,5,5])
        np.testing.assert_array_equal(np.asarray(trace.execution_tokens)[:,0],np.arange(1,7))
        # A new high score between scheduled action plans triggers immediate replanning.
        assert [int(t[0,-1,0]) for t in model.used]==[1,2,3,4,5]
        assert [int(t[0,0,0]) for t in model.used]==[1,2,2,2,5]
        assert result['task_estimation']['diagnostic_only'] is False
        if memory:assert model.memory_module.writes==list(range(6))


def test_peak_selection_requires_a_decoder(evaluator):
    with pytest.raises(ValueError,match='requires --task-estimator'):
        evaluator.run_model_in_sim(Env(success_at=None),StreamModel(),torch.device('cpu'),np.zeros((6,7),'f4'),
            chunk_size=4,max_steps=6,switch_at=.5,task_context_mode='freeze',use_camera=['image'],record_frames=False)


def test_video_keeps_full_model_episode_when_expert_replay_ends_earlier(evaluator,monkeypatch,tmp_path):
    import imageio
    frames=[]
    class Writer:
        def append_data(self,frame):frames.append(frame)
        def close(self):pass
    monkeypatch.setattr(imageio,'get_writer',lambda *a,**k:Writer())
    image=np.zeros((32,32,3),dtype='u1')
    evaluator.save_video_3panel([image]*3,[image],[image]*3,str(tmp_path/'video.mp4'))
    assert len(frames)==3
