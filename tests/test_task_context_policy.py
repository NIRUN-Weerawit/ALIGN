"""Actual policy integration: causal inputs, frozen weights, split safety and reload."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from data.task_context_dataset import geometric_state, spatial_summary, json_hash, SCHEMA
from models.task_context_observer import TaskContextObserver
from models.task_context_policy import TaskContextPolicyBranch, rotation_columns, visual_slots


class FakeVision(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = nn.Linear(1, 1)

    def forward(self, features):
        return features


@pytest.fixture
def bundle(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    summary = dict(cameras=1, raw_dim=768, width=16)
    encoder = dict(width=16, tokens=4, token_dim=4, modes=4, layers=1, d_state=4,
        d_conv=3, expand=2, visual_width=768, horizons=[2,5], execution_dim=4)
    observer = TaskContextObserver(summary, encoder)
    observer.state_mean.copy_(torch.arange(10) / 10)
    observer.state_std.fill_(2.)
    path = tmp_path / 'observer.pt'
    torch.save(dict(schema='align-causal-task-observer-v1',summary_config=summary,
        encoder_config=encoder,summary=observer.summary.state_dict(),encoder=observer.encoder.state_dict(),
        state_mean=observer.state_mean,state_std=observer.state_std,
        metadata=dict(fps=20.,target_manifest_sha256='test',cameras=['image'])),path)
    return path


def make_model(monkeypatch, bundle, memory=False, initialization='pretrained', head_type='diffusion'):
    import models.align_intention as module
    monkeypatch.setattr(module,'VisionEncoder',FakeVision)
    model=module.ALIGNIntentionModel(state_dim=8,mamba_output_dim=16,num_cameras=1,
        compressed_dim=4,action_dim=7,chunk_size=4,history_size=1,head_d_model=8,
        head_type=head_type,diffusion_loss_repeats=1,use_memory_bank=memory,memory_bank_len=3,
        task_context_checkpoint=str(bundle),task_context_initialization=initialization)
    model._build_head_and_bank(256*4)
    return model


def inputs():
    torch.manual_seed(9)
    raw=torch.randn(2,9,257,768)
    state=torch.randn(2,9,7)*.1
    lengths=torch.tensor([9,7])
    cls=raw[:,:,-1,None]
    patches=raw[:,:,:-1]
    ticks=torch.arange(9.).expand(2,-1)
    return raw,patches,cls,state,lengths,ticks


def test_visual_and_geometric_conversion_match_pretraining():
    raw,patches,cls,state,_,_=inputs()
    expected=spatial_summary(raw[0].numpy(),1)
    np.testing.assert_allclose(visual_slots(patches,cls,1)[0].numpy(),expected,atol=2e-6)
    expected=geometric_state(state[0,:,:6].numpy(),state[0,:,6].numpy())[:,3:9]
    np.testing.assert_allclose(rotation_columns(state[0,:,3:6]).numpy(),expected,atol=2e-7)
    zeros=torch.zeros(1,3,requires_grad=True)
    rotation_columns(zeros).sum().backward()
    assert torch.isfinite(zeros.grad).all()


def test_frozen_context_sequence_matches_stream_and_ignores_future(bundle):
    branch=TaskContextPolicyBranch(str(bundle)).eval()
    _,patches,cls,state,lengths,ticks=inputs()
    before=branch.sequence(patches,cls,state,lengths,ticks)
    cache=None
    for t in range(6):
        result,cache=branch.step(patches[:,t],cls[:,t],state[:,t],cache)
        torch.testing.assert_close(result,before[:,t],atol=3e-6,rtol=3e-5)
    changed=patches.clone();changed[:,6:]=float('nan')
    torch.testing.assert_close(branch.sequence(changed,cls,state,lengths,ticks)[:,:6],before[:,:6])
    branch.freeze_task(cache)
    snapshot=cache.task_snapshot.clone()
    out,cache=branch.step(patches[:,6],cls[:,6],state[:,6],cache)
    torch.testing.assert_close(out[:,:4],snapshot)
    assert cache.recurrence.steps==7


@pytest.mark.parametrize('memory',[False,True])
@pytest.mark.parametrize('head_type', ['diffusion', 'flow_matching'])
def test_policy_context_gradients_memory_and_self_contained_reload(monkeypatch,bundle,memory,head_type):
    from data.task_context_policy_dataset import context_policy_options
    import models.align_intention as module
    model=make_model(monkeypatch,bundle,memory,head_type=head_type)
    model.train()
    assert model.intention_encoder is None
    assert not model.task_context.observer.encoder.training
    raw,patches,cls,state,lengths,ticks=inputs()
    context=model.task_context.sequence(patches,cls,state,lengths,ticks)[:,3]
    if memory:model.memory_module.reset(batch_size=2,device=torch.device('cpu'))
    visual=torch.randn(2,1,1024);encoded=torch.randn(2,1,8)
    v,s,c=model.condition_actions(visual,encoded,context)
    torch.testing.assert_close(c,context)
    cond=model.action_condition(v,s,c)
    assert cond.shape==(2,1,1032)
    cond.square().mean().backward()
    assert model.intention_head.context_task[1].weight.grad.abs().sum()>0
    assert model.intention_head.context_execution[1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.task_context.parameters())
    cfg=context_policy_options(model)
    # Reconstruction uses config and policy weights, never the original artifact path.
    clone=module.ALIGNIntentionModel(state_dim=8,mamba_output_dim=16,num_cameras=1,
        compressed_dim=4,action_dim=7,chunk_size=4,history_size=1,head_d_model=8,
        head_type=head_type,diffusion_loss_repeats=1,use_memory_bank=memory,memory_bank_len=3,**cfg)
    clone._build_head_and_bank(1024)
    clone.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(clone.task_context.sequence(patches,cls,state,lengths,ticks),
        model.task_context.sequence(patches,cls,state,lengths,ticks))


def test_action_only_scratch_has_selector_and_backbone_gradients(bundle):
    branch=TaskContextPolicyBranch(str(bundle),initialization='scratch').train()
    _,patches,cls,state,lengths,ticks=inputs()
    result=branch.sequence(patches,cls,state,lengths,ticks)
    result.square().mean().backward()
    assert branch.observer.encoder.blocks[0].in_proj.weight.grad.abs().sum()>0
    assert branch.observer.encoder.logits.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in branch.observer.summary.parameters())


def test_fixed_episode_split_rejects_overlap_wrong_cameras_or_binding(tmp_path):
    from data.task_context_policy_dataset import policy_episode_split
    ds=SimpleNamespace(_episode_keys=['ep_0','ep_1','ep_2'],cameras=['image'],_get_episode_length=lambda i:9)
    m=dict(schema=SCHEMA,split=dict(train=['ep_0'],development=['ep_1'],confirmation=['ep_2']),
        episodes={k:dict(length=9) for k in ds._episode_keys},cameras=['image'],fps=20.)
    def write():
        m['manifest_sha256']=json_hash({k:v for k,v in m.items() if k!='manifest_sha256'})
        p=tmp_path/'manifest.json';p.write_text(json.dumps(m));return p
    path=write();train,val,record=policy_episode_split(ds,path)
    assert train==[0] and val==[1] and record['confirmation_episodes']==['ep_2']
    with pytest.raises(ValueError,match='bound'):
        policy_episode_split(ds,path,dict(target_manifest_sha256='other'))
    ds.cameras=['wrist']
    with pytest.raises(ValueError,match='camera'):
        policy_episode_split(ds,path)
    ds.cameras=['image'];m['split']['confirmation']=['ep_0'];path=write()
    with pytest.raises(ValueError,match='overlap'):
        policy_episode_split(ds,path)


def test_context_cli_requires_checkpoint_and_fixed_split():
    from scripts.run_intention_ablation import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--output','unused','--variants','context_no_memory'])
    args=parse_args(['--output','unused','--variants','context_no_memory','context_scratch_memory',
        '--task-context-checkpoint','observer.pt','--task-context-split-manifest','manifest.json'])
    assert args.temporal_sampling=='episode' and args.history_size==1


def test_main_trainer_builds_complete_episode_loaders_with_pinned_tensors(bundle,tmp_path,monkeypatch):
    import training.train_intention as trainer
    import data.align_dataset as data_module
    from scripts.run_intention_ablation import CachedEpisodes
    # Exercise the main entry point's dataset/collator dispatch, not a separate trainer.
    fake=SimpleNamespace(_episode_keys=['a','b','c'],cameras=['image'],
        _get_episode_length=lambda ep:9)
    monkeypatch.setattr(trainer,'ALIGNDataset',lambda *a,**k:fake)
    import data.task_context_policy_dataset as split_module
    monkeypatch.setattr(split_module,'policy_episode_split',lambda *a,**k:([0],[1],dict(confirmation_episodes=['c'])))
    args=SimpleNamespace(task_context_checkpoint=str(bundle),task_context_split_manifest='split.json',
        data=['fake.h5'],use_precomputed_dinov2=True,dinov2_path='cache',cameras=['image'],
        chunk_size=4,seed=42,device='cuda',use_intent_tokens=False,use_memory_bank=False,
        segment_min_mult=0,segment_max_mult=0,history_size=1,batch_size=1,num_workers=0,
        persistent_workers=False,prefetch_factor=1)
    _,train,val=trainer.build_datasets(args)
    assert isinstance(train,CachedEpisodes) and train.temporal_sampling=='episode'
    assert train.episodes==[0] and val.episodes==[1]
    sample=dict(segment_len=9,frames_segment=np.zeros((9,257,768),dtype='f4'),
        states_segment=np.zeros((9,7),dtype='f4'),actions_segment=np.zeros((9,7),dtype='f4'),
        loss_anchor_mask=np.ones(9,dtype=bool),observation_timesteps=np.arange(9,dtype='f4'))
    loader,_=trainer.build_loaders([sample],[sample],args)
    out=loader.collate_fn([sample])
    assert isinstance(out['frames_segment'],torch.Tensor) and out['frames_segment'].shape==(1,9,257,768)


def test_handover_freezes_task_preserves_execution_and_replans_queue(monkeypatch,bundle):
    import eval.eval_libero_v4_trajectory as evaluator
    model=make_model(monkeypatch,bundle).eval()
    monkeypatch.setattr(evaluator,'get_sim_eef_pose',lambda obs,rotation_convention='libero':np.asarray([obs['step']*.01,0,0,0,0,0],dtype='f4'))
    monkeypatch.setattr(evaluator,'get_sim_frame',lambda *a,**k:np.zeros((4,4,3),dtype='u1'))
    raw=torch.randn(1,257,768)
    monkeypatch.setattr(model,'_vision_forward',lambda frames:raw.expand(len(frames),-1,-1))
    clock=[0];calls=[];payloads=[];steps=[]
    original=model.encode_step
    def encoded(*a,**k):
        result=original(*a,**k)
        payloads.append(result[4].clone());steps.append(result[3].recurrence.steps)
        return result
    monkeypatch.setattr(model,'encode_step',encoded)
    def sample(*args,**kwargs):
        calls.append(clock[0]);return torch.zeros(1,4,7)
    monkeypatch.setattr(model,'sample_actions',sample)
    class Env:
        def reset(self):clock[0]=0;return dict(step=0)
        def step(self,action):clock[0]+=1;return dict(step=clock[0]),0.,False,{}
    result=evaluator.run_model_in_sim(Env(),model,torch.device('cpu'),np.zeros((6,7),dtype='f4'),
        chunk_size=4,max_steps=6,switch_at=.5,task_context_mode='handover',action_horizon=2,use_camera=['image'])
    assert result['switch_step']==3 and calls==[0,2,3,4]
    assert steps==list(range(1,7))
    torch.testing.assert_close(payloads[3][:,:4],payloads[4][:,:4])
    torch.testing.assert_close(payloads[3][:,:4],payloads[5][:,:4])
    assert not torch.allclose(payloads[3][:,-1],payloads[5][:,-1])


def test_common_policy_warm_start_never_replaces_context(monkeypatch,bundle,tmp_path):
    from data.task_context_policy_dataset import warm_start_policy
    model=make_model(monkeypatch,bundle)
    before={k:v.clone() for k,v in model.task_context.state_dict().items()}
    source={k:v.clone() for k,v in model.state_dict().items()}
    source['intention_head.context_task_scale'].fill_(.9)
    source['task_context.observer.state_mean'].fill_(100.)
    key=next(k for k in source if k.startswith('intention_head.unet.') and source[k].is_floating_point())
    source[key].fill_(.125)
    path=tmp_path/'policy.pt'
    torch.save(dict(config=dict(action_dim=7,chunk_size=4,num_cameras=1,compressed_dim=4,head_type='diffusion'),model_state_dict=source),path)
    record=warm_start_policy(model,path)
    assert record['tensors_copied']>0
    assert (model.state_dict()[key]==.125).all()
    assert model.intention_head.context_task_scale.item()<.02
    for k,v in model.task_context.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)


def test_production_checkpoint_loader_needs_no_original_observer(monkeypatch,bundle,tmp_path):
    from eval.eval_intention import load_intention_model
    from data.task_context_policy_dataset import context_policy_options
    model=make_model(monkeypatch,bundle).eval()
    cfg=dict(state_dim=8,mamba_output_dim=16,action_dim=7,chunk_size=4,history_size=1,
        num_cameras=1,compressed_dim=4,head_type='diffusion',head_d_model=8,
        diffusion_train_steps=100,diffusion_loss_repeats=1,pool_out_dim=1024,
        frozen_vision_omitted=True,**context_policy_options(model))
    state={k:v for k,v in model.state_dict().items() if not k.startswith('vision_encoder.')}
    path=tmp_path/'policy.pt'
    torch.save(dict(config=cfg,model_state_dict=state),path)
    bundle.unlink()
    loaded,_=load_intention_model(str(path),torch.device('cpu'))
    _,patches,cls,raw,lengths,ticks=inputs()
    torch.testing.assert_close(loaded.task_context.sequence(patches,cls,raw,lengths,ticks),
        model.task_context.sequence(patches,cls,raw,lengths,ticks))
    state.pop(next(k for k in state if k.startswith('task_context.observer.encoder.blocks.')))
    damaged=tmp_path/'damaged.pt';torch.save(dict(config=cfg,model_state_dict=state),damaged)
    with pytest.raises(ValueError,match='missing trained'):
        load_intention_model(str(damaged),torch.device('cpu'))
