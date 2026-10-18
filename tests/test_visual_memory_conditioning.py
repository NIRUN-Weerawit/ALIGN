"""Explicit visual memory survives masking, batching, sampling and checkpointing."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.intention_head import DiffusionPolicyHead, FlowMatchingPolicyHead
from models.memory_bank import EpisodicMemoryModule, MemoryRetrieval


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_state_visual_encoder_excludes_cudnn_and_restores_backend(monkeypatch):
    from models.intention_encoder import StateConditionalCrossAttn
    from torch.nn.attention import SDPBackend, sdpa_kernel
    original = torch.nn.functional.scaled_dot_product_attention
    calls = []
    def checked(*args, **kwargs):
        assert not torch.backends.cuda.cudnn_sdp_enabled()
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch.nn.functional, 'scaled_dot_product_attention', checked)
    module = StateConditionalCrossAttn(compressed_dim=32, state_dim=64).cuda()
    patches = torch.randn(24, 512, 32, device='cuda', requires_grad=True)
    state = torch.randn(24, 64, device='cuda', requires_grad=True)
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = module(patches, state)
        output.square().mean().backward()
        assert torch.backends.cuda.cudnn_sdp_enabled()
    assert calls and torch.isfinite(patches.grad).all()
    assert torch.isfinite(state.grad).all()


def active_head(head_cls=DiffusionPolicyHead, device='cpu'):
    head = head_cls(cond_dim=12,hidden_dim=8,time_dim=8,chunk_size=8).to(device)
    nn.init.normal_(head.unet.output_proj.weight,std=.1)
    head.unet.configure_visual_context(4,2,width=16)
    return head.to(device)


@pytest.mark.parametrize('head_cls',[DiffusionPolicyHead,FlowMatchingPolicyHead])
def test_explicit_tokens_preserve_initial_function_and_reach_all_action_scales(head_cls):
    torch.manual_seed(9)
    reference=head_cls(cond_dim=12,hidden_dim=8,time_dim=8,chunk_size=8)
    nn.init.normal_(reference.unet.output_proj.weight,std=.1)
    head=copy.deepcopy(reference)
    head.unet.configure_visual_context(4,2,width=16)
    x,t,cond,context=torch.randn(2,8,7),torch.ones(2),torch.randn(2,1,12),torch.randn(2,8,4)
    predict='predict_noise' if head_cls is DiffusionPolicyHead else 'predict_velocity'
    expected=getattr(reference,predict)(x,t,cond)
    torch.testing.assert_close(getattr(head,predict)(x,t,cond,visual_context=context),expected)
    lengths={}
    hooks=[module.register_forward_pre_hook(
        lambda module,args,name=name:lengths.update({name:args[0].shape[-1]}))
        for name,module in head.unet.visual_context_blocks.items()]
    # Activate zero-initialized residual outputs to test upstream gradients.
    for block in head.unet.visual_context_blocks.values():nn.init.normal_(block.output.weight,std=.1)
    context.requires_grad_()
    output=getattr(head,predict)(x,t,cond,visual_context=context)
    changed=getattr(head,predict)(x,t,cond,visual_context=context.flip(1))
    assert not torch.allclose(output,changed)
    (output*torch.randn_like(output)).sum().backward()
    assert context.grad.abs().sum()>0
    assert head.unet.visual_context_projection.key.weight.grad.abs().sum()>0
    for block in head.unet.visual_context_blocks.values():assert block.query.weight.grad.abs().sum()>0
    assert lengths==dict(bottleneck=2,decoder=4,output=8)
    for hook in hooks:hook.remove()


@pytest.mark.parametrize('head_cls',[DiffusionPolicyHead,FlowMatchingPolicyHead])
def test_visual_context_excludes_invalid_samples_before_noise_generation(head_cls):
    head=active_head(head_cls)
    for block in head.unet.visual_context_blocks.values():nn.init.normal_(block.output.weight,std=.1)
    target,cond,context=torch.randn(2,8,7),torch.randn(2,1,12),torch.randn(2,8,4,requires_grad=True)
    with torch.no_grad():target[1]=float('nan');cond[1]=float('nan');context[1]=float('nan')
    torch.manual_seed(12)
    masked=head.loss(target,cond,sample_mask=torch.tensor([True,False]),visual_context=context)
    torch.manual_seed(12)
    expected=head.loss(target[:1],cond[:1],visual_context=context[:1])
    torch.testing.assert_close(masked,expected)
    masked.backward()
    assert context.grad[0].abs().sum()>0 and context.grad[1].count_nonzero()==0


@pytest.mark.parametrize('head_cls',[DiffusionPolicyHead,FlowMatchingPolicyHead])
def test_sampling_projects_visual_keys_and_values_once_and_strict_reload(head_cls):
    head=active_head(head_cls)
    clone=active_head(head_cls);clone.load_state_dict(head.state_dict(),strict=True)
    calls=[]
    hook=head.unet.visual_context_projection.register_forward_hook(lambda *args:calls.append(1))
    cond,context=torch.randn(2,1,12),torch.randn(2,8,4)
    torch.manual_seed(4);expected=head.sample(cond,num_steps=3,visual_context=context)
    torch.manual_seed(4);actual=clone.sample(cond,num_steps=3,visual_context=context)
    assert calls==[1] and torch.isfinite(actual).all()
    torch.testing.assert_close(actual,expected)
    hook.remove()
    with pytest.raises(ValueError,match='explicit visual patch context'):head.sample(cond,num_steps=1)


def test_two_reads_share_bank_keep_empty_rows_and_train_both_blocks():
    module=MemoryRetrieval(8,2,retrieval_depth=2)
    q,keys,values=torch.randn(2,8),torch.randn(2,4,8),torch.randn(2,4,8)
    mask=torch.tensor([[True,True,False,False],[False,False,False,False]])
    captured=[]
    hook=module.extra_layers[0].register_forward_pre_hook(lambda module,args:captured.append(args))
    out=module(q,keys,mask,values)
    assert captured[0][1] is keys and captured[0][3] is values
    torch.testing.assert_close(out[1],q[1])
    (out*torch.randn_like(out)).sum().backward()
    assert module.retrieval_attn.in_proj_weight.grad.abs().sum()>0
    assert module.extra_layers[0].retrieval_attn.in_proj_weight.grad.abs().sum()>0
    hook.remove()


def test_intention_changes_visual_selection_without_changing_stored_values():
    torch.manual_seed(2)
    memory=EpisodicMemoryModule(8,4,4,bank_len=4,patch_dim=4,
        patch_temporal=True,value_preserving=True,retrieval_depth=2,intent_query=True)
    with torch.no_grad():memory.visual_intent_projection.weight.copy_(torch.eye(4)*3)
    state=torch.zeros(1,4)
    query=torch.zeros(1,8)
    context=torch.tensor([[3.,0.,0.,0.]],requires_grad=True)
    outputs=[]
    for sign in [1,-1]:
        memory.reset(1,torch.device('cpu'))
        memory.observe_only(torch.tensor([[1.,0.,0.,0.]*2]),state,torch.zeros(1,1,4))
        memory.observe_only(torch.tensor([[-1.,0.,0.,0.]*2]),state,torch.zeros(1,1,4))
        original=memory.perceptual_bank[:,:2].clone()
        outputs.append(memory(query,state,sign*context[:,None])[0])
        torch.testing.assert_close(memory.perceptual_bank[:,:2],original)
        torch.testing.assert_close(memory.perceptual_bank[:,2].flatten(1),query)
    assert not torch.allclose(*outputs)
    outputs[-1].sum().backward()
    assert context.grad.abs().sum()>0
    assert memory.visual_intent_projection.weight.grad.abs().sum()>0
    # All reads, including extra layers, preserve the selected raw value space.
    assert memory.perceptual_retrieval.extra_layers[0].value_preserving


def test_no_intent_memory_remains_active_and_legacy_has_no_new_parameters(monkeypatch):
    from models import align_intention as align
    monkeypatch.setattr(align,'VisionEncoder',lambda **kw:nn.Linear(1,1))
    legacy=align.ALIGNIntentionModel(state_dim=4,compressed_dim=4,mamba_output_dim=0,
        head_d_model=8,action_dim=7,chunk_size=8,use_memory_bank=True)
    legacy._build_head_and_bank(32)
    assert not any('extra_layers.' in k or 'visual_context_' in k for k in legacy.state_dict())
    model=align.ALIGNIntentionModel(state_dim=4,compressed_dim=4,mamba_output_dim=0,
        num_cameras=2,head_d_model=8,action_dim=7,chunk_size=8,use_memory_bank=True,
        visual_context_mode='multiscale',memory_retrieval_depth=2,memory_intent_query=True)
    model._build_head_and_bank(32)
    assert model.memory_patch_retrieval and not model.memory_intent_query
    assert model.intention_encoder is None and model.memory_module.cognitive_retrieval is None
    model.memory_module.reset(2,torch.device('cpu'))
    p,s=torch.randn(2,1,32),torch.randn(2,1,4)
    model.condition_actions(p,s)
    fused=model.condition_actions(p+1,s)
    assert model.memory_module._count.tolist()==[2,2]
    tokens=model.action_context(fused[0])['visual_context']
    assert tokens.shape==(2,8,4)
    torch.testing.assert_close(tokens.flatten(1),fused[0][:,0])
    assert torch.isfinite(model.sample_actions(*fused,num_steps=2)).all()


@pytest.mark.parametrize('head_type',['diffusion','flow_matching'])
@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_real_training_validation_and_reload_with_intent_memory(monkeypatch,head_type,device):
    from models.intention_encoder import HAS_MAMBA
    if not HAS_MAMBA:pytest.skip('Requires Mamba constructor')
    from models import align_intention as align
    from training.train_intention import train_v4_epoch,validate
    monkeypatch.setattr(align,'VisionEncoder',lambda **kw:nn.Linear(1,1))
    def build():
        model=align.ALIGNIntentionModel(state_dim=4,compressed_dim=4,mamba_output_dim=4,
            num_cameras=2,
            intent_dim=8,num_intent_tokens=1,use_intent_tokens=True,use_memory_bank=True,
            visual_context_mode='multiscale',memory_retrieval_depth=2,memory_intent_query=True,
            head_type=head_type,head_d_model=8,action_dim=7,chunk_size=4,
            memory_bank_len=3,diffusion_loss_repeats=1,diffusion_train_steps=10)
        model=model.to(device)
        model._build_head_and_bank(2048)
        return model
    model=build()
    args=SimpleNamespace(history_size=1,chunk_size=4,action_dim=7,head_type=head_type,
        skip_nan=False,grad_clip=1.,no_sample_during_train=True,head_window_batch_size=2)
    batch=dict(frames_segment=torch.randn(2,8,514,768),states_segment=torch.randn(2,8,7),
        actions_segment=torch.rand(2,8,7),segment_len=torch.tensor([8,6]),
        observation_timesteps=torch.arange(8).expand(2,-1),
        loss_anchor_mask=torch.tensor([[True,False,True,False,True,False,False,False],
                                      [True,False,True,False,False,False,False,False]]))
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    for _ in range(3):
        loss,_=train_v4_epoch(model,[batch],opt,torch.device(device),args)
        assert loss>0
    assert model.memory_module.visual_intent_projection.weight.grad.abs().sum()>0
    assert model.intention_head.unet.visual_context_projection.key.weight.grad.abs().sum()>0
    clone=build();clone.load_state_dict(model.state_dict(),strict=True)
    torch.manual_seed(7);actual=validate(model,[batch],torch.device(device),args)
    torch.manual_seed(7);expected=validate(clone,[batch],torch.device(device),args)
    assert actual[0]==pytest.approx(expected[0])
    for metric in ['pos_mse','rot_mse','grip_mse']:assert actual[2][metric]==pytest.approx(expected[2][metric])


def test_evaluator_reconstructs_new_architecture_from_checkpoint(monkeypatch,tmp_path):
    from models import align_intention as align
    from eval.eval_intention import load_intention_model
    class Vision(nn.Module):
        def __init__(self,**kw):super().__init__();self.anchor=nn.Linear(1,1)
        def forward(self,x):return torch.zeros(len(x),5,768,device=x.device)
    monkeypatch.setattr(align,'VisionEncoder',Vision)
    config=dict(state_dim=4,compressed_dim=4,mamba_output_dim=0,use_history=False,
        num_cameras=1,head_type='diffusion',head_d_model=8,action_dim=7,chunk_size=8,
        history_size=1,use_intent_tokens=False,use_memory_bank=True,
        memory_mode='episodic',memory_bank_len=3,memory_detach_writes=True,memory_write_fused=False,
        visual_context_mode='multiscale',memory_retrieval_depth=2,memory_intent_query=True,
        diffusion_train_steps=100,diffusion_loss_repeats=4,diffusion_clip_sample=True)
    model=align.ALIGNIntentionModel(**{k:v for k,v in config.items() if k!='use_history'})
    model._build_head_and_bank(16)
    path=tmp_path/'model.pt';torch.save(dict(config=config,model_state_dict=model.state_dict()),path)
    loaded,_=load_intention_model(str(path),torch.device('cpu'))
    assert loaded.memory_module.retrieval_depth==2
    assert loaded.memory_patch_retrieval and loaded.visual_context_mode=='multiscale'
    for key,value in model.state_dict().items():torch.testing.assert_close(loaded.state_dict()[key],value)


@pytest.mark.parametrize('history',[1,4])
def test_plain_sparse_training_and_validation_carry_explicit_context(monkeypatch,history):
    from models import align_intention as align
    from tests.test_sparse_plain_history import Features
    from scripts.run_intention_ablation import CachedEpisodes
    from training.plain_history_training import SparseHistoryEpisodes,collate_sparse_histories,plain_history_epoch
    monkeypatch.setattr(align,'VisionEncoder',lambda **kw:nn.Linear(1,1))
    model=align.ALIGNIntentionModel(state_dim=4,compressed_dim=4,mamba_output_dim=0,
        head_d_model=8,action_dim=7,chunk_size=4,history_size=history,
        head_history_mode='full' if history>1 else 'mean',visual_context_mode='multiscale',
        use_text=True,text_dim=4,text_encoder_type='bag',text_vocab=['move','object'])
    model._build_head_and_bank(1024)
    parent=CachedEpisodes(Features(),[0,1],20,42,True,'episode',2,4)
    dataset=SparseHistoryEpisodes(parent,history)
    batch=collate_sparse_histories([dataset[0],dataset[1]])
    args=SimpleNamespace(plain_head_batch_size=2,gripper_threshold=.5,action_dim=7,grad_clip=1.)
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    for _ in range(3):assert plain_history_epoch(model,[batch],torch.device('cpu'),args,opt)[0]>0
    assert model.intention_head.unet.visual_context_projection.value.weight.grad.abs().sum()>0
    assert plain_history_epoch(model,[batch],torch.device('cpu'),args)[0]>0


def test_cli_depth_and_conflicting_context_are_validated():
    from scripts.run_intention_ablation import parse_args
    args=parse_args(['--output','example','--visual-context-mode','multiscale','--memory-intent-query'])
    assert args.memory_retrieval_depth==2 and args.memory_intent_query
    for options in [['--memory-retrieval-depth','0'],['--visual-context-mode','multiscale','--visual-token-attention']]:
        with pytest.raises(SystemExit):parse_args(['--output','example',*options])
