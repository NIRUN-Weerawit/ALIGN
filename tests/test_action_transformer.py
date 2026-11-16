import pytest
import torch
from models.action_transformer import ActionTransformer
from models.intention_head import DiffusionPolicyHead, FlowMatchingPolicyHead
from models.context_aci_head import ACIContextDiffusionPolicyHead, ACIContextFlowMatchingPolicyHead

torch.set_num_threads(2)

@pytest.mark.parametrize('flow', [False, True])
@pytest.mark.parametrize('aci', [False, True])
def test_loss_sample_masks_roundtrip(flow, aci):
    cls = ((ACIContextFlowMatchingPolicyHead if flow else ACIContextDiffusionPolicyHead) if aci
           else (FlowMatchingPolicyHead if flow else DiffusionPolicyHead))
    kwargs = dict(cond_dim=1032 if aci else 12, action_dim=7, hidden_dim=32,
                  chunk_size=3, time_dim=16, loss_repeats=2)
    if aci:kwargs.update(state_dim=8, visual_token_dim=2, num_cameras=2, temporal_tokens=1)
    else:kwargs['history_size']=4
    head = cls(**kwargs)
    head.unet = ActionTransformer(action_dim=7, cond_dim=head.cond_dim, time_dim=16,
                                 width=32, layers=2, heads=4, aci=head.unet if aci else None)
    if aci:
        tokens = torch.randn(2,6,128,requires_grad=True)
        cond = head(torch.randn(2,1,1024),torch.randn(2,1,8),tokens)
        masked = head.remove_context(cond,task=True,execution=True,temporal=True)
        assert torch.isfinite(head.sample(masked,num_steps=2)).all()
        altered=masked.clone()
        altered[:,:,head.state_dim:head.state_dim+6*128]+=17
        x=torch.randn(2,3,7);t=torch.randn(2,16)
        torch.testing.assert_close(head.unet(x,masked,t),head.unet(x,altered,t))
    else:cond=torch.randn(2,4,12,requires_grad=True)
    target=torch.randn(2,3,7)
    target[1]=float('nan')
    loss=head.loss(target,cond,sample_mask=torch.tensor([True,False]))
    assert torch.isfinite(loss)
    loss.backward()
    grad=tokens.grad if aci else cond.grad
    assert grad[0].abs().sum()>0
    assert grad[1].abs().sum()==0
    head.eval();torch.manual_seed(1);expected=head.sample(cond.detach(),num_steps=2)
    head.load_state_dict(head.state_dict(),strict=True)
    torch.manual_seed(1);actual=head.sample(cond.detach(),num_steps=2)
    torch.testing.assert_close(actual,expected)
    assert actual.shape==(2,3,7)

@pytest.mark.parametrize('inline',[False,True])
def test_optional_visual_attention(inline):
    head=DiffusionPolicyHead(cond_dim=520,action_dim=7,hidden_dim=32,chunk_size=4,time_dim=16)
    head.unet=ActionTransformer(action_dim=7,cond_dim=520,time_dim=16,width=32,layers=2,heads=4)
    if inline:head.unet.configure_visual_attention(512,1)
    else:head.unet.configure_visual_context(1,2)
    cond=torch.randn(2,1,520)
    opts={} if inline else dict(visual_context=torch.randn(2,512,1))
    loss=head.loss(torch.randn(2,4,7),cond,**opts);loss.backward()
    assert head.unet.visual_projection.input.weight.grad.abs().sum()>0
    assert torch.isfinite(head.sample(cond,num_steps=2,**opts)).all()

from tests.test_task_context_policy import bundle, FakeVision, inputs

@pytest.mark.parametrize('kind',['diffusion','flow_matching'])
@pytest.mark.parametrize('trainable',[False,True])
def test_model_memory_temporal_context_and_reload(monkeypatch,bundle,tmp_path,kind,trainable):
    from models.align_intention import ALIGNIntentionModel
    from data.task_context_policy_dataset import context_policy_options
    from eval.eval_intention import load_intention_model
    monkeypatch.setattr('models.align_intention.VisionEncoder',FakeVision)
    config=dict(state_dim=8,mamba_output_dim=16,num_cameras=1,compressed_dim=4,
        action_dim=7,chunk_size=3,history_size=1,head_d_model=32,head_type=kind,
        action_backbone='transformer',transformer_layers=2,transformer_heads=4,diffusion_clip_sample=True,
        task_context_fusion='aci',task_context_fusion_dim=16,
        policy_temporal_mamba=True,policy_temporal_width=16,
        task_context_trainable=trainable,task_context_summary_trainable=trainable,
        use_memory_bank=True,memory_mode='episodic',memory_patch_retrieval=True,
        memory_temporal_cognitive=True)
    model=ALIGNIntentionModel(**config,task_context_checkpoint=str(bundle)).train()
    model._build_head_and_bank(1024)
    model.memory_module.reset(2,torch.device("cpu"))
    _,patches,cls,state,lengths,ticks=inputs()
    context=model.context_sequence(patches,cls,state,lengths,ticks)[:,3]
    visual,encoded,context=model.condition_actions(torch.randn(2,1,1024),torch.randn(2,1,8),context)
    cond=model.action_condition(visual,encoded,context)
    model.intention_head.loss(torch.randn(2,3,7),cond,**model.action_context(visual)).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.policy_temporal_encoder.parameters())
    assert any(p.grad is not None for p in model.task_context.parameters()) == trainable
    config.update(context_policy_options(model),pool_out_dim=1024,frozen_vision_omitted=True)
    weights={k:v for k,v in model.state_dict().items() if not k.startswith('vision_encoder.')}
    path=tmp_path/'transformer.pt';torch.save(dict(config=config,model_state_dict=weights),path)
    loaded,cfg=load_intention_model(str(path),torch.device('cpu'))
    assert cfg['action_backbone']=='transformer'
    torch.manual_seed(8);expected=model.eval().sample_actions(visual,encoded,context,num_steps=2)
    torch.manual_seed(8);actual=loaded.sample_actions(visual,encoded,context,num_steps=2)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)

def test_cli_defaults_and_options():
    from scripts.run_intention_ablation import parse_args,legacy_resume_defaults
    args=parse_args(['--output','unused','--action-backbone','transformer','--transformer-layers','6','--transformer-heads','8'])
    assert (args.action_backbone,args.transformer_layers,args.transformer_heads)==('transformer',6,8)
    assert legacy_resume_defaults({})['action_backbone']=='unet'

@pytest.mark.parametrize('kind', ['diffusion','flow_matching'])
@pytest.mark.parametrize('device_name', ['cpu','cuda'])
@pytest.mark.parametrize('temporal', [False,True])
def test_transformer_epoch_and_validation(monkeypatch,bundle,kind,device_name,temporal):
    if device_name=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from types import SimpleNamespace
    from models.align_intention import ALIGNIntentionModel
    from training.train_intention import train_v4_epoch,validate
    monkeypatch.setattr('models.align_intention.VisionEncoder',FakeVision)
    device=torch.device(device_name)
    model=ALIGNIntentionModel(state_dim=8,mamba_output_dim=16,num_cameras=1,compressed_dim=4,
        action_dim=7,chunk_size=4,history_size=1,head_d_model=32,head_type=kind,action_backbone="transformer",transformer_layers=2,transformer_heads=4,
        diffusion_loss_repeats=1,task_context_checkpoint=str(bundle),
        task_context_fusion='aci',task_context_fusion_dim=16,
        policy_temporal_mamba=temporal,policy_temporal_width=16).to(device)
    model._build_head_and_bank(1024)
    frozen={k:v.detach().clone() for k,v in model.task_context.state_dict().items()}
    before=model.intention_head.unet.context[1].weight.detach().clone()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    args=SimpleNamespace(history_size=1,chunk_size=4,action_dim=7,head_type=kind,
        skip_nan=False,grad_clip=1.,no_sample_during_train=True,head_window_batch_size=2)
    batch=dict(frames_segment=torch.randn(2,9,257,768),states_segment=torch.randn(2,9,7),
        actions_segment=torch.rand(2,9,7),segment_len=torch.tensor([9,7]),
        observation_timesteps=torch.arange(9.).expand(2,-1))
    for _ in range(3):
        loss,_=train_v4_epoch(model,[batch],optimizer,device,args)
        assert 0<loss<float('inf')
    assert not torch.equal(before,model.intention_head.unet.context[1].weight)
    assert model.intention_head.unet.context[1].weight.grad.abs().sum()>0
    if temporal:
        weight=model.policy_temporal_encoder.blocks[0].in_proj.weight
        assert weight.grad is not None and weight.grad.abs().sum()>0
    for key,value in frozen.items():
        torch.testing.assert_close(model.task_context.state_dict()[key],value,rtol=0,atol=0)
    assert all(p.grad is None for p in model.task_context.parameters())
    val_loss,_,_=validate(model,[batch],device,args)
    assert 0<val_loss<float('inf')
