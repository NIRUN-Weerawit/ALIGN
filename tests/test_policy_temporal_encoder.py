"""Independent learned history, causal streaming, reset and frozen-observer contracts."""
import pytest
import torch

from models.policy_temporal_encoder import PolicyTemporalEncoder
from tests.test_task_context_policy import bundle, FakeVision, inputs


@pytest.mark.parametrize('device_name', ['cpu','cuda'])
def test_causal_sequence_stream_padding_gradients_and_reset(device_name):
    if device_name=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(2);torch.manual_seed(42)
    module=PolicyTemporalEncoder(16,4,width=16,layers=2).to(device_name)
    summary=torch.randn(2,9,16,device=device_name,requires_grad=True)
    lengths=torch.tensor([9,7],device=device_name)
    padded=summary.clone();padded[1,7:]=float('nan')
    actual=module(padded,lengths)
    assert torch.isfinite(actual).all() and actual.shape==(2,9,1,4)
    assert actual[1,7:].abs().sum()==0
    changed=summary.detach().clone();changed[:,5:]+=30
    torch.testing.assert_close(module(changed,lengths)[:,:5],actual[:,:5])
    cache=module.new_stream(2)
    for t in range(7):
        online=module.step(summary[:,t],cache)
        torch.testing.assert_close(online,actual[:,t],atol=3e-5,rtol=3e-5)
    assert cache.steps==7
    torch.testing.assert_close(module.step(summary[:,0],module.new_stream(2)),actual[:,0],atol=3e-5,rtol=3e-5)
    swapped=summary.detach().clone();swapped[:,:6]=0
    assert not torch.allclose(module(swapped,lengths)[:,6],actual[:,6],atol=1e-8,rtol=1e-8)
    actual[:,6].square().mean().backward()
    assert module.blocks[0].in_proj.weight.grad.abs().sum()>0
    assert summary.grad is None  # Input summaries/observer remain frozen.


def test_parallel_branch_keeps_native_tokens_decoder_identity_and_handover(monkeypatch,bundle):
    from models.align_intention import ALIGNIntentionModel
    from eval.task_estimation import context_fingerprint
    torch.set_num_threads(2)
    monkeypatch.setattr('models.align_intention.VisionEncoder',FakeVision)
    model=ALIGNIntentionModel(state_dim=8,mamba_output_dim=16,num_cameras=1,compressed_dim=4,
        action_dim=7,chunk_size=4,history_size=1,head_d_model=8,
        task_context_checkpoint=str(bundle),task_context_fusion='aci',task_context_fusion_dim=16,
        policy_temporal_mamba=True,policy_temporal_width=16)
    _,patches,cls,state,lengths,ticks=inputs()
    before=context_fingerprint(model.task_context)
    native=model.task_context.sequence(patches,cls,state,lengths,ticks)
    payload=model.context_sequence(patches,cls,state,lengths,ticks)
    assert payload.shape==(2,9,6,4)
    torch.testing.assert_close(payload[:,:,:5],native,rtol=0,atol=0)
    assert payload[:,:,-1].requires_grad
    model._build_head_and_bank(1024)
    raw=torch.cat([patches,cls],-2)
    cache=None
    for t in range(5):
        _,_,_,cache,online=model.encode_step(raw[:,t],state[:,t],cache,produce_intent=True)
        torch.testing.assert_close(online,payload[:,t],atol=4e-5,rtol=4e-5)
    assert cache.policy_temporal.steps==cache.recurrence.steps==5
    temporal_state=cache.policy_temporal
    model.task_context.freeze_task(cache)
    snapshot=cache.task_snapshot.clone()
    _,_,_,cache,online=model.encode_step(raw[:,5],state[:,5],cache,produce_intent=True)
    torch.testing.assert_close(online[:,:4],snapshot)
    assert cache.policy_temporal is temporal_state and cache.policy_temporal.steps==6
    assert context_fingerprint(model.task_context)==before


def test_six_token_context_masks_preserve_frozen_execution_and_remove_temporal():
    from models.context_aci_head import ACIContextDiffusionPolicyHead
    from eval.context_interventions import intervene
    h=ACIContextDiffusionPolicyHead(cond_dim=1032,state_dim=8,visual_token_dim=4,num_cameras=1,
        context_dim=4,temporal_tokens=1,hidden_dim=8,context_fusion_dim=16,chunk_size=4)
    v,s,c=torch.randn(2,1,1024),torch.randn(2,1,8),torch.randn(2,6,4)
    for arm,expected in [('no_task',[0,0,0,0,1,1]),('no_execution',[1,1,1,1,0,1]),
                         ('no_temporal',[1,1,1,1,1,0]),('no_context',[0]*6)]:
        packed=intervene(h,h,v,s,c,arm)
        _,_,_,mask=h.unet.unpack(packed)
        torch.testing.assert_close(mask,torch.tensor([expected]*2).to(mask))
    replacement=torch.randn(2,4,4)
    _,tokens,_,_=h.unet.unpack(intervene(h,h,v,s,c,'task_swap',replacement))
    torch.testing.assert_close(tokens[:,:4],replacement)
    torch.testing.assert_close(tokens[:,4:],c[:,4:])


def test_cli_requires_frozen_pretrained_context_and_aci():
    from scripts.run_intention_ablation import parse_args
    args=['--output','unused','--variants','context_no_memory',
        '--task-context-checkpoint','observer.pt','--task-context-split-manifest','manifest.json',
        '--task-context-fusion','aci','--policy-temporal-mamba']
    assert parse_args(args).policy_temporal_mamba
    for extra in (['--task-context-trainable'],['--task-context-fusion','additive'],
                  ['--variants','context_scratch_no_memory'],['--policy-temporal-width','0']):
        with pytest.raises(SystemExit):parse_args(args+extra)
