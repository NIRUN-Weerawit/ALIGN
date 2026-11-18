import torch
from tests.test_task_context_policy import bundle,FakeVision,inputs

def test_bag_context_gradient_stream_and_reload(monkeypatch,bundle,tmp_path):
    from models.align_intention import ALIGNIntentionModel
    from data.task_context_policy_dataset import context_policy_options
    from eval.eval_intention import load_intention_model
    torch.set_num_threads(2)
    monkeypatch.setattr('models.align_intention.VisionEncoder',FakeVision)
    cfg=dict(state_dim=8,mamba_output_dim=16,num_cameras=1,compressed_dim=4,
        head_type="diffusion",diffusion_clip_sample=True,action_dim=7,chunk_size=4,history_size=1,head_d_model=8,
        task_context_bag_text=True,use_text=True,text_encoder_type='bag',text_vocab=['bowl','stove'],text_dim=16,
        task_context_fusion='aci',task_context_fusion_dim=16,policy_temporal_mamba=True,policy_temporal_width=16)
    m=ALIGNIntentionModel(**cfg,task_context_checkpoint=str(bundle));m._build_head_and_bank(1024)
    assert not hasattr(m.task_context.observer,'encoder')
    m.encode_task_text(['bowl','stove'])
    _,p,c,s,lengths,t=inputs();context=m.context_sequence(p,c,s,lengths,t)
    torch.testing.assert_close(context[:,0,:5],context[:,3,:5])
    assert not torch.allclose(context[0,0,:5],context[1,0,:5])
    context[:,:,:5].square().mean().backward()
    assert m.text_encoder.embedding.weight.grad.abs().sum()>0
    assert m.text_context_projection.weight.grad.abs().sum()>0
    assert all(x.grad is None for x in m.task_context.observer.summary.parameters())
    stream=None
    for i in range(3):
        tok,stream=m.task_context.step(p[:,i],c[:,i],s[:,i],stream)
        torch.testing.assert_close(tok,context[:,i,:5])
    cfg.update(context_policy_options(m),pool_out_dim=1024,frozen_vision_omitted=True)
    f=tmp_path/'bag.pt';torch.save(dict(config=cfg,model_state_dict={k:v for k,v in m.state_dict().items() if not k.startswith('vision_encoder.')}),f)
    loaded,_=load_intention_model(str(f),torch.device('cpu'));loaded.encode_task_text(['bowl','stove'])
    torch.testing.assert_close(loaded.context_sequence(p,c,s,lengths,t),context)

def test_cli():
    from scripts.run_intention_ablation import parse_args
    a=parse_args(['--output','unused','--variants','context_memory','--task-context-checkpoint','observer.pt','--task-context-split-manifest','split.json','--task-context-bag-text','--use-task-text','--text-encoder-type','bag'])
    assert a.task_context_bag_text

import pytest
@pytest.mark.parametrize('kind', ['diffusion','flow_matching'])
@pytest.mark.parametrize('device_name', ['cpu','cuda'])
@pytest.mark.parametrize('temporal', [False,True])
def test_bag_context_epoch_and_validation(monkeypatch,bundle,kind,device_name,temporal):
    if device_name=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from types import SimpleNamespace
    from models.align_intention import ALIGNIntentionModel
    from training.train_intention import train_v4_epoch,validate
    monkeypatch.setattr('models.align_intention.VisionEncoder',FakeVision)
    device=torch.device(device_name)
    model=ALIGNIntentionModel(state_dim=8,mamba_output_dim=16,num_cameras=1,compressed_dim=4,
        action_dim=7,chunk_size=4,history_size=1,head_d_model=32,head_type=kind,task_context_bag_text=True,use_text=True,text_encoder_type="bag",text_vocab=["bowl","stove"],text_dim=16,
        diffusion_loss_repeats=1,task_context_checkpoint=str(bundle),
        task_context_fusion='aci',task_context_fusion_dim=16,
        policy_temporal_mamba=temporal,policy_temporal_width=16).to(device)
    model._build_head_and_bank(1024)
    frozen={k:v.detach().clone() for k,v in model.task_context.state_dict().items()}
    before=model.intention_head.unet.context_projection.weight.detach().clone()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    args=SimpleNamespace(history_size=1,chunk_size=4,action_dim=7,head_type=kind,
        skip_nan=False,grad_clip=1.,no_sample_during_train=True,head_window_batch_size=2)
    batch=dict(frames_segment=torch.randn(2,9,257,768),states_segment=torch.randn(2,9,7),
        actions_segment=torch.rand(2,9,7),segment_len=torch.tensor([9,7]),
        observation_timesteps=torch.arange(9.).expand(2,-1),texts=["bowl","stove"])
    for _ in range(3):
        loss,_=train_v4_epoch(model,[batch],optimizer,device,args)
        assert 0<loss<float('inf')
    assert not torch.equal(before,model.intention_head.unet.context_projection.weight)
    assert model.intention_head.unet.context_projection.weight.grad.abs().sum()>0
    if temporal:
        weight=model.policy_temporal_encoder.blocks[0].in_proj.weight
        assert weight.grad is not None and weight.grad.abs().sum()>0
    for key,value in frozen.items():
        torch.testing.assert_close(model.task_context.state_dict()[key],value,rtol=0,atol=0)
    assert all(p.grad is None for p in model.task_context.parameters())
    val_loss,_,_=validate(model,[batch],device,args)
    assert 0<val_loss<float('inf')
