"""ACI information routing, masked losses, gradients and production reload."""
import pytest
import torch

from models.context_aci_head import ACIContextDiffusionPolicyHead, ACIContextFlowMatchingPolicyHead
from tests.test_task_context_policy import bundle, FakeVision, inputs


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def head(kind='diffusion', cameras=1, tokens=4):
    cls = ACIContextDiffusionPolicyHead if kind == 'diffusion' else ACIContextFlowMatchingPolicyHead
    return cls(cond_dim=cameras*1024+8, state_dim=8, visual_token_dim=4,
        num_cameras=cameras, context_tokens=tokens, context_dim=4,
        context_fusion_dim=16, hidden_dim=8, chunk_size=8, loss_repeats=2)


def activate(h):
    # At initialization both residual attention outputs and the final action
    # output are zero, intentionally. Exercise information paths after opening them.
    with torch.no_grad():
        h.unet.output_proj.weight.normal_(std=.1)
        for module in h.unet.modules():
            if hasattr(module, 'film'):
                module.film[-1].weight.normal_(std=.02)
        for layer in h.unet.aci_attention.values():
            layer.output.weight.normal_(std=.1)


def values(cameras=1, tokens=4):
    return (torch.randn(2,1,cameras*1024), torch.randn(2,1,8), torch.randn(2,tokens+1,4))


@pytest.mark.parametrize('kind', ['diffusion', 'flow_matching'])
@pytest.mark.parametrize('tokens', [1,4])
def test_alternation_film_inputs_cache_gradients_and_sampling(kind,tokens):
    h = head(kind, cameras=2, tokens=tokens)
    activate(h)
    v,s,c = [x.requires_grad_() for x in values(2,tokens)]
    cond = h(v,s,c)
    events, film, prepared = [], [], []
    hooks = [m.register_forward_pre_hook(lambda m,a,name=name: events.append(name))
             for name,m in h.unet.aci_attention.items()]
    hooks += [h.unet.enc1.film.register_forward_pre_hook(lambda m,a:film.append(a[0].detach()))]
    original = h.unet.prepare_visual_context
    def prepare(condition):
        prepared.append(len(condition))
        return original(condition)
    h.unet.prepare_visual_context = prepare
    loss = h.loss(torch.randn(2,8,7),cond)
    assert prepared == [2]  # K/V built before repeating independent noise draws.
    assert events == ['context_encoder','observation_bottleneck','context_decoder','observation_output']
    assert film[0].shape == (4,72)  # ONLY 8D state + 64D time, no context or vision.
    torch.testing.assert_close(film[0][:,:8],s[:,0].repeat(2,1))
    loss.backward()
    for x in (v,s,c):
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert c.grad[:,:-1].abs().sum()>0 and c.grad[:,-1].abs().sum()>0
    for module in (h.unet.context_key,h.unet.context_value,h.unet.visual_projection.key):
        assert module.weight.grad.abs().sum()>0
    prepared.clear();events.clear()
    sampled = h.sample(cond.detach(),num_steps=3)
    assert prepared == [2]  # BOTH modalities cached over all sampler iterations.
    assert sampled.shape==(2,8,7) and torch.isfinite(sampled).all()
    assert events == ['context_encoder','observation_bottleneck','context_decoder','observation_output']*3
    for hook in hooks:hook.remove()


@pytest.mark.parametrize('kind', ['diffusion', 'flow_matching'])
def test_padding_removed_before_condition_projection(kind):
    h=head(kind);activate(h)
    v,s,c=values();v[1]=s[1]=c[1]=float('nan')
    packed=h(v,s,c);target=torch.randn(2,8,7);target[1]=float('nan')
    torch.manual_seed(17)
    actual=h.loss(target,packed,sample_mask=torch.tensor([True,False]))
    torch.manual_seed(17)
    expected=h.loss(target[:1],packed[:1])
    torch.testing.assert_close(actual,expected)
    actual.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in h.parameters())


def test_context_removal_masks_values_slot_embeddings_and_output_bias():
    from eval.context_interventions import intervene
    h=head();activate(h)
    for layer in h.unet.aci_attention.values():
        with torch.no_grad():layer.output.bias.fill_(.4)
    v,s,c=values();packed=h(v,s,c)
    x=torch.randn(2,8,7);t=torch.tensor([10,10])
    def predict(condition):return h.predict_noise(x,t,condition)
    baseline=predict(packed)
    for arm,indices in [('no_task',slice(None,-1)),('no_execution',slice(-1,None)),('no_context',slice(None))]:
        removed=intervene(h,h,v,s,c,arm)
        altered=c.clone();altered[:,indices]=torch.randn_like(altered[:,indices])*20
        alternate=intervene(h,h,v,s,altered,arm)
        torch.testing.assert_close(predict(removed),predict(alternate),rtol=0,atol=0)
        assert not torch.allclose(baseline,predict(removed))
    # Changing observation and individual native tokens reaches action prediction.
    v2=v.clone();v2[:,:,0:4]+=3
    assert not torch.allclose(baseline,predict(h(v2,s,c)))
    for index in range(5):
        c2=c.clone();c2[:,index,0]+=2
        assert not torch.allclose(baseline,predict(h(v,s,c2)))


@pytest.mark.parametrize('kind', ['diffusion','flow_matching'])
def test_production_model_frozen_encoder_and_strict_loader(monkeypatch,bundle,tmp_path,kind):
    from models.align_intention import ALIGNIntentionModel
    from data.task_context_policy_dataset import context_policy_options
    from eval.eval_intention import load_intention_model
    monkeypatch.setattr('models.align_intention.VisionEncoder',FakeVision)
    config=dict(state_dim=8,mamba_output_dim=16,num_cameras=1,compressed_dim=4,
        action_dim=7,chunk_size=8,history_size=1,head_d_model=8,head_type=kind,
        diffusion_loss_repeats=1,diffusion_clip_sample=True,task_context_fusion='aci',task_context_fusion_dim=16)
    model=ALIGNIntentionModel(**config,task_context_checkpoint=str(bundle)).train()
    model._build_head_and_bank(1024)
    activate(model.intention_head)
    _,patches,cls,state,lengths,ticks=inputs()
    context=model.task_context.sequence(patches,cls,state,lengths,ticks)[:,3]
    visual=torch.randn(2,1,1024);encoded=torch.randn(2,1,8)
    cond=model.action_condition(visual,encoded,context)
    loss=model.intention_head.loss(torch.randn(2,8,7),cond,**model.action_context(visual))
    loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in model.task_context.parameters())
    assert model.intention_head.unet.context_projection.weight.grad.abs().sum()>0
    config.update(context_policy_options(model),pool_out_dim=1024,frozen_vision_omitted=True)
    weights={k:v for k,v in model.state_dict().items() if not k.startswith('vision_encoder.')}
    path=tmp_path/'aci.pt';torch.save(dict(config=config,model_state_dict=weights),path)
    bundle.unlink()  # All required pretrained observer weights are inside the policy.
    loaded,cfg=load_intention_model(str(path),torch.device('cpu'))
    assert cfg['task_context_fusion']=='aci'
    torch.manual_seed(8);expected=model.eval().sample_actions(visual,encoded,context,num_steps=2)
    torch.manual_seed(8);actual=loaded.sample_actions(visual,encoded,context,num_steps=2)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    weights.pop('intention_head.unet.context_projection.weight')
    torch.save(dict(config=config,model_state_dict=weights),path)
    with pytest.raises(ValueError,match='missing trained'):
        load_intention_model(str(path),torch.device('cpu'))


def test_cli_exposes_aci_and_rejects_incompatible_options():
    from scripts.run_intention_ablation import parse_args
    args=['--output','unused','--variants','context_no_memory',
        '--task-context-checkpoint','observer.pt','--task-context-split-manifest','manifest.json',
        '--task-context-fusion','aci','--task-context-fusion-dim','16']
    assert parse_args(args).task_context_fusion=='aci'
    for extra in (['--task-context-fusion-dim','15'],['--history-size','2'],
                  ['--visual-token-attention'],['--visual-context-mode','multiscale'],
                  ['--variants','context_history']):
        with pytest.raises(SystemExit):parse_args(args+extra)


@pytest.mark.parametrize('kind', ['diffusion','flow_matching'])
@pytest.mark.parametrize('device_name', ['cpu','cuda'])
def test_production_epoch_and_validation_update_head_keep_observer_fixed(monkeypatch,bundle,kind,device_name):
    if device_name=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from types import SimpleNamespace
    from models.align_intention import ALIGNIntentionModel
    from training.train_intention import train_v4_epoch,validate
    monkeypatch.setattr('models.align_intention.VisionEncoder',FakeVision)
    device=torch.device(device_name)
    model=ALIGNIntentionModel(state_dim=8,mamba_output_dim=16,num_cameras=1,compressed_dim=4,
        action_dim=7,chunk_size=4,history_size=1,head_d_model=8,head_type=kind,
        diffusion_loss_repeats=1,task_context_checkpoint=str(bundle),
        task_context_fusion='aci',task_context_fusion_dim=16).to(device)
    model._build_head_and_bank(1024)
    frozen={k:v.detach().clone() for k,v in model.task_context.state_dict().items()}
    before=model.intention_head.unet.context_projection.weight.detach().clone()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    args=SimpleNamespace(history_size=1,chunk_size=4,action_dim=7,head_type=kind,
        skip_nan=False,grad_clip=1.,no_sample_during_train=True,head_window_batch_size=2)
    batch=dict(frames_segment=torch.randn(2,9,257,768),states_segment=torch.randn(2,9,7),
        actions_segment=torch.rand(2,9,7),segment_len=torch.tensor([9,7]),
        observation_timesteps=torch.arange(9.).expand(2,-1))
    for _ in range(3):
        loss,_=train_v4_epoch(model,[batch],optimizer,device,args)
        assert 0<loss<float('inf')
    assert not torch.equal(before,model.intention_head.unet.context_projection.weight)
    assert model.intention_head.unet.context_projection.weight.grad.abs().sum()>0
    for key,value in frozen.items():
        torch.testing.assert_close(model.task_context.state_dict()[key],value,rtol=0,atol=0)
    assert all(p.grad is None for p in model.task_context.parameters())
    val_loss,_,_=validate(model,[batch],device,args)
    assert 0<val_loss<float('inf')
