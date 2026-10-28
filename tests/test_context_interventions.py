import numpy as np
import pytest
import torch
from eval.context_interventions import intervene, select_donors
from models.task_context_policy import ContextDiffusionPolicyHead


@pytest.mark.parametrize('fusion', ['gated_residual', 'film', 'cross_attention'])
def test_task_removal_includes_learned_interactions(fusion):
    head = ContextDiffusionPolicyHead(cond_dim=6, action_dim=7, hidden_dim=16,
        time_dim=8, chunk_size=8, context_tokens=2, context_dim=3,
        context_fusion=fusion, context_fusion_dim=8)
    # Move the optional branch off its neutral initialization.
    with torch.no_grad():
        for name, value in head.named_parameters():
            if name.startswith(('context_gate', 'context_film', 'context_attention_output')):
                value.add_(.02)
    visual, state, tokens = torch.randn(2, 1, 4), torch.randn(2, 1, 2), torch.randn(2, 3, 3)
    current, task, execution = head.conditioning_components(visual, state, tokens)
    removed = intervene(head, head._build_per_step_cond, visual, state, tokens, 'no_task')
    torch.testing.assert_close(removed, current + execution)
    assert task.abs().sum() > 0


def test_interventions_preserve_inputs_and_remove_complete_residuals():
    torch.manual_seed(7)
    head=ContextDiffusionPolicyHead(cond_dim=6,action_dim=7,hidden_dim=16,time_dim=8,
        chunk_size=8,context_tokens=2,context_dim=3)
    visual,state=torch.randn(2,1,4),torch.randn(2,1,2)
    payload=torch.randn(2,3,3);before=payload.clone()
    build=head._build_per_step_cond
    baseline=build(visual,state,payload)
    task=head.context_task_scale*head.context_task(payload[:,:-1].flatten(1))[:,None]
    execution=head.context_execution_scale*head.context_execution(payload[:,-1])[:,None]
    assert torch.equal(intervene(head,build,visual,state,payload,'repeat_control'),baseline)
    assert torch.allclose(intervene(head,build,visual,state,payload,'no_task'),baseline-task)
    assert torch.allclose(intervene(head,build,visual,state,payload,'no_execution'),baseline-execution)
    assert torch.allclose(intervene(head,build,visual,state,payload,'no_context'),torch.cat([visual,state],-1),atol=1e-6)
    replacement=torch.randn(2,2,3)
    changed=torch.cat([replacement,payload[:,-1:]],1)
    for arm in ('freeze_task','task_mean','task_swap','same_task_swap'):
        assert torch.equal(intervene(head,build,visual,state,payload,arm,replacement),build(visual,state,changed))
    assert torch.equal(payload,before)
    with pytest.raises(ValueError,match='Replacement'):
        intervene(head,build,visual,state,payload,'task_swap',torch.zeros(1,2,3))
    with pytest.raises(ValueError,match='Unknown'):
        intervene(head,build,visual,state,payload,'bad')


def test_donor_selection_is_reproducible_and_respects_task_labels():
    keys=np.array(['a','b','c','d']);groups=np.array([0,0,1,2])
    same,other=select_donors(keys,groups,0,42)
    assert (same,other)==select_donors(keys,groups,0,42)
    assert groups[same]==0 and groups[other]!=0
    with pytest.raises(ValueError):select_donors(keys,np.zeros(4),0,42)


def test_streaming_intervention_starts_at_handover_and_restores_head():
    from eval.context_interventions import StreamingIntervention
    torch.manual_seed(1)
    head=ContextDiffusionPolicyHead(cond_dim=6,action_dim=7,hidden_dim=16,time_dim=8,
        chunk_size=8,context_tokens=2,context_dim=3)
    visual,state=torch.randn(1,1,4),torch.randn(1,1,2)
    payloads=[torch.randn(1,3,3) for _ in range(4)]
    original=head._build_per_step_cond
    with StreamingIntervention(head,'freeze_task',2,42) as runner:
        for t,payload in enumerate(payloads):
            expected=payload.clone()
            if t>=2:expected[:,:-1]=payloads[2][:,:-1]
            assert torch.equal(head._build_per_step_cond(visual,state,payload),original(visual,state,expected))
        assert runner.step==4
    assert head._build_per_step_cond==original
    assert '_build_per_step_cond' not in head.__dict__
    with pytest.raises(RuntimeError):
        with StreamingIntervention(head,'no_task',0,42):raise RuntimeError('simulator failure')
    assert head._build_per_step_cond==original


def test_audit_wrapper_preserves_simulator_camera_access_and_detects_fixture_changes():
    from types import SimpleNamespace
    from tools.evaluate_context_interventions import AuditedEnvironment
    from eval.eval_libero_v4_trajectory import get_sim_frame
    images={'agentview_image':np.zeros((2,2,3),dtype=np.uint8),
            'robot0_eye_in_hand_image':np.ones((2,2,3),dtype=np.uint8)}
    geometry=SimpleNamespace(**{name:np.zeros((2,3)) for name in
        ('body_pos','body_quat','geom_pos','geom_quat','geom_size')})
    wrapped=SimpleNamespace(env=SimpleNamespace(_get_observations=lambda:images),
        sim=SimpleNamespace(model=geometry,get_state=lambda:np.zeros(5),forward=lambda:None),
        _update_observables=lambda force:None,
        set_init_state=lambda state:images)
    audit=AuditedEnvironment(wrapped)
    assert np.array_equal(get_sim_frame(audit,'image'),images['agentview_image'])
    audit.set_init_state(np.zeros(5));first=audit.initial_identity
    audit.set_init_state(np.zeros(5));assert first==audit.initial_identity
    geometry.body_pos[0,0]=.1
    audit.set_init_state(np.zeros(5));assert first!=audit.initial_identity


def test_rollout_summary_keeps_precompleted_cases_out_of_policy_denominator():
    from tools.evaluate_context_interventions import summarize
    rows=[dict(episode='a',arm='baseline',policy_eligible=True,success=False),
          dict(episode='a',arm='no_task',policy_eligible=True,success=True),
          dict(episode='b',arm='baseline',policy_eligible=False,success=False),
          dict(episode='b',arm='no_task',policy_eligible=False,success=False)]
    result=summarize(rows,['baseline','no_task'])
    assert result['no_task']['eligible']==1
    assert result['no_task']['success_rate']==1
    assert result['no_task']['paired_improvements']==['a']


def test_invalid_repeat_report_does_not_present_intervention_success_rates():
    from tools.evaluate_context_interventions import render
    result=render(dict(status='invalid_repeat_control',repeat_audit={'max_action_delta':2.0}))
    assert 'comparisons were aborted' in result
    assert '| Policy completion |' not in result
