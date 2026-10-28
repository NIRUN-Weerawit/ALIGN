from types import SimpleNamespace
import numpy as np
import pytest
import torch

from eval.task_token_counterfactual import AuditedTaskIntervention, PassiveEnvironmentAudit
from models.task_context_policy import ContextDiffusionPolicyHead


@pytest.mark.parametrize('arm',['normal','repeat','same_task','wrong_task','no_task'])
@pytest.mark.parametrize('window',[1,4])
def test_head_intervention_preserves_prefix_execution_and_diffusion_noise(arm,window):
    torch.manual_seed(7)
    head=ContextDiffusionPolicyHead(cond_dim=6,action_dim=7,hidden_dim=16,time_dim=8,
        chunk_size=4,context_tokens=2,context_dim=3)
    head.eval()
    visual,state,payload=torch.randn(1,window,4),torch.randn(1,window,2),torch.randn(1,3,3)
    before=payload.clone();donor=np.ones((2,3),dtype=np.float32)
    build=head._build_per_step_cond
    # A simple sampler isolates the noise audit from diffusion architecture.
    head.sample=lambda cond:torch.randn(cond.shape[0],4,7)
    original_sample=head.sample
    with AuditedTaskIntervention(head,arm,1,42,None,donor) as hook:
        conditions=[];draws=[]
        for step in range(3):
            conditions.append(head._build_per_step_cond(visual,state,payload))
            draws.append(head.sample(conditions[-1]))
        assert torch.equal(conditions[0],build(visual,state,payload))
        assert torch.equal(payload,before)
        assert np.array_equal(np.asarray(hook.execution),np.repeat(payload[0,-1][None].numpy(),3,0))
        assert len(hook.noise)==3
        for step,draw in enumerate(draws):
            g=torch.Generator().manual_seed(42+step)
            assert torch.equal(draw,torch.randn(1,4,7,generator=g))
        if arm in ('same_task','wrong_task'):
            changed=payload.clone();changed[:,:-1]=torch.from_numpy(donor)
            assert torch.equal(conditions[1],build(visual,state,changed))
            assert not torch.equal(conditions[1],conditions[0])
        elif arm=='no_task':
            current,_,execution=head.conditioning_components(visual,state,payload)
            assert torch.equal(conditions[1],current+execution)
            assert hook.rows[1]['task_contribution_active'] is False
        else:
            assert torch.equal(conditions[0],conditions[1])
    assert head.sample is original_sample
    assert '_build_per_step_cond' not in head.__dict__


def test_passive_environment_does_not_refresh_or_replace_camera_observations():
    obs={'agentview_image':np.zeros((2,2,3),dtype=np.uint8),'robot0_eye_in_hand_image':np.ones((2,2,3),dtype=np.uint8)}
    geometry=SimpleNamespace(**{k:np.zeros((2,3)) for k in ('body_pos','body_quat','geom_pos','geom_quat','geom_size')})
    env=SimpleNamespace(reset=lambda:obs,set_init_state=lambda state:obs,
        step=lambda action:(obs,0,False,{}),sim=SimpleNamespace(model=geometry,
            get_state=lambda:np.zeros(5)))
    wrapper=PassiveEnvironmentAudit(env)
    assert wrapper.reset() is obs
    assert wrapper.set_init_state(np.zeros(5)) is obs
    assert wrapper.step(np.zeros(7))[0] is obs
    assert len(wrapper.steps)==1
    assert len(wrapper.initial['cameras'])==2


def test_alternative_goal_uses_existing_scene_without_changing_original_goal():
    obs={'agentview_image':np.zeros((1,1,3),dtype=np.uint8)}
    geometry=SimpleNamespace(**{k:np.zeros(1) for k in ('body_pos','body_quat','geom_pos','geom_quat','geom_size')})
    original={'problem_name':'table','regions':{},'goal_state':[['on','bowl','plate']]}
    native=SimpleNamespace(parsed_problem=original,object_states_dict={'bowl':0,'plate':1,'stove':2},
        _eval_predicate=lambda predicate:predicate==['on','bowl','stove'])
    env=SimpleNamespace(env=native,reset=lambda:obs,set_init_state=lambda state:obs,
        step=lambda action:(obs,0,False,{}),sim=SimpleNamespace(model=geometry,get_state=lambda:np.zeros(5)))
    wrong={'problem_name':'table','regions':{},'goal_state':[['on','bowl','stove']]}
    wrapper=PassiveEnvironmentAudit(env,dict(task='wrong',definition=wrong))
    wrapper.set_init_state(np.zeros(5));wrapper.step(np.zeros(7))
    assert wrapper.goal_trace[-1]['success'] is True
    assert native.parsed_problem is original
    missing={**wrong,'goal_state':[['on','bowl','cabinet']]}
    unavailable=PassiveEnvironmentAudit(env,dict(task='missing',definition=missing))
    unavailable.set_init_state(np.zeros(5))
    assert unavailable.alternative_reason and not unavailable.goal_trace
