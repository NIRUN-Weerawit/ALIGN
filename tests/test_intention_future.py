"""Future supervision is causal, masked, detached and microbatch invariant."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from models.intention_future import IntentionFuturePredictor, future_prediction_loss, freeze_intent_backbone


def example():
    torch.manual_seed(1)
    predictor=IntentionFuturePredictor(8,1,4,[2,5],16)
    model=SimpleNamespace(intention_future_predictor=predictor)
    intent=torch.randn(2,5,1,8,requires_grad=True)
    cls=torch.randn(2,8,1,4,requires_grad=True)
    state=torch.randn(2,8,7,requires_grad=True)
    mask=torch.tensor([[1,0,0,1,1],[1,1,0,1,0]],dtype=torch.bool)
    return model,intent,cls,state,torch.tensor([8,4]),mask


def test_future_targets_exclude_padding_and_receive_no_gradients():
    model,intent,cls,state,lengths,mask=example()
    with torch.no_grad():cls[1,4:]=float('nan');state[1,4:]=float('nan')
    loss,metrics=future_prediction_loss(model,intent,cls,state,lengths,mask,6)
    assert metrics['pairs_h2']==5 and metrics['pairs_h5']==1
    assert torch.isfinite(loss)
    loss.backward()
    assert cls.grad is None and state.grad is None
    assert torch.count_nonzero(intent.grad[1,3])==0
    assert torch.count_nonzero(intent.grad[0,0])>0
    assert torch.isfinite(intent.grad).all()


def test_future_loss_preserves_sample_weighting_across_microbatches():
    model,intent,cls,state,lengths,mask=example()
    mask[1,1]=False
    count=int(mask.sum())
    full,_=future_prediction_loss(model,intent,cls,state,lengths,mask,count)
    full.backward();expected=[p.grad.clone() if p.grad is not None else None for p in model.intention_future_predictor.parameters()]
    model.intention_future_predictor.zero_grad()
    parts=[]
    for i in range(2):
        micro_count=int(mask[i].sum())
        loss,_=future_prediction_loss(model,intent[i:i+1],cls[i:i+1],state[i:i+1],lengths[i:i+1],mask[i:i+1],micro_count)
        parts.append(loss*micro_count/count)
    torch.testing.assert_close(sum(parts),full)
    sum(parts).backward()
    for p,g in zip(model.intention_future_predictor.parameters(),expected):
        if g is not None:torch.testing.assert_close(p.grad,g,atol=1e-6,rtol=1e-5)


def test_unavailable_horizons_do_not_repeat_terminal_frame():
    model,intent,cls,state,lengths,mask=example()
    loss,metrics=future_prediction_loss(model,intent,cls,state,torch.tensor([1,1]),mask,6)
    assert loss.item()==0 and metrics['pairs_h2']==metrics['pairs_h5']==0


def test_rotation_change_uses_shortest_euler_displacement():
    predictor=IntentionFuturePredictor(8,1,4,[1],16)
    for p in predictor.parameters():p.data.zero_()
    state=torch.zeros(1,2,7);state[0,0,3]=3.13;state[0,1,3]=-3.13
    _,metrics=future_prediction_loss(SimpleNamespace(intention_future_predictor=predictor),
        torch.zeros(1,1,1,8),torch.ones(1,2,1,4),state,torch.tensor([2]),torch.ones(1,1,dtype=torch.bool),1)
    assert metrics['state_delta_sum_h1']<.001


def test_future_options_reject_disabled_intent_and_invalid_windows():
    from scripts.run_intention_ablation import parse_args
    assert parse_args(['--output','example']).intent_future_weight==0
    for options in [['--intent-future-weight','-1'],['--intent-future-weight','.1'],
                    ['--intent-future-horizons','0'],['--freeze-intent-backbone']]:
        with pytest.raises(SystemExit):parse_args(['--output','example',*options])
    with pytest.raises(SystemExit):
        parse_args(['--output','example','--variants','intent_no_memory',
                    '--intent-future-weight','.1','--temporal-sampling','crop'])


def small_model(weight=.1,head_type='diffusion'):
    from models.intention_encoder import HAS_MAMBA
    if not HAS_MAMBA:pytest.skip('Requires Mamba constructor')
    from models.align_intention import ALIGNIntentionModel
    with patch('models.align_intention.VisionEncoder',lambda **kw:torch.nn.Linear(1,1)):
        model=ALIGNIntentionModel(state_dim=4,num_cameras=1,compressed_dim=4,
            mamba_output_dim=4,intent_dim=8,num_intent_tokens=1,use_intent_tokens=True,
            head_d_model=8,chunk_size=4,action_dim=7,head_type=head_type,diffusion_train_steps=10,
            diffusion_loss_repeats=1,intent_future_weight=weight,intent_future_horizons=[2,5],
            intent_future_hidden_dim=16)
    model._build_head_and_bank(1024)
    return model


def test_disabled_objective_keeps_original_checkpoint_keys_and_strict_roundtrip():
    torch.manual_seed(12)
    old=small_model(0)
    assert not any(k.startswith('intention_future_predictor.') for k in old.state_dict())
    small_model(0).load_state_dict(old.state_dict(),strict=True)
    torch.manual_seed(12)
    new=small_model()
    for key,value in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[key],value)
    small_model().load_state_dict(new.state_dict(),strict=True)


@pytest.mark.parametrize('head_type',['diffusion','flow_matching'])
def test_production_loop_updates_projection_with_fixed_backbone_and_logs_auxiliary(head_type):
    from training.train_intention import train_v4_epoch,validate
    torch.set_num_threads(2)
    model=small_model(head_type=head_type);freeze_intent_backbone(model)
    model.train()
    assert not model.state_encoder.training and not model.intention_encoder.training
    assert model.intention_head.training and model.intention_future_predictor.training
    before=model.intention_encoder.intent_proj.weight.detach().clone()
    backbone=model.intention_encoder.mamba.in_proj.weight.detach().clone()
    args=SimpleNamespace(history_size=1,chunk_size=4,action_dim=7,head_type=head_type,
        skip_nan=False,grad_clip=1.,no_sample_during_train=True,head_window_batch_size=2)
    batch=dict(frames_segment=torch.randn(2,10,257,768),states_segment=torch.randn(2,10,7),
        actions_segment=torch.rand(2,10,7),segment_len=torch.tensor([10,8]))
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
    loss,_=train_v4_epoch(model,[batch],optimizer,torch.device('cpu'),args)
    assert loss>0 and args._last_future_metrics['intent_future_pairs_h5']>0
    assert not torch.equal(before,model.intention_encoder.intent_proj.weight)
    torch.testing.assert_close(backbone,model.intention_encoder.mamba.in_proj.weight)
    _,_,metrics=validate(model,[batch],torch.device('cpu'),args)
    assert metrics['intent_future_loss']>0 and metrics['total_loss']>metrics['action_loss']
