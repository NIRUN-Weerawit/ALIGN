"""Memory contracts: real time, raw values, detached writes, causal targets."""
import numpy as np
import torch
from torch import nn
import pytest
from models.memory_bank import EpisodicMemoryModule
from models.intention_head import DiffusionPolicyHead
from scripts.run_intention_ablation import CachedEpisodes,collate_segments

@pytest.mark.parametrize('patch_dim',[None,4])
def test_raw_detached_writes_preserve_features_and_train_retrieval(patch_dim):
    m=EpisodicMemoryModule(8,4,4,bank_len=3,patch_dim=patch_dim)
    m.reset(2,torch.device('cpu'))
    p,s,c=torch.randn(2,8,requires_grad=True),torch.randn(2,4),torch.randn(2,1,4)
    out=m(p,s,c,timestamp=torch.tensor([10.,20.]))
    torch.testing.assert_close(out[0],p)
    assert not m.perceptual_bank.requires_grad
    torch.testing.assert_close(m.perceptual_bank[:,0].reshape(2,8),p)
    q=torch.randn(2,8,requires_grad=True)
    fused=m(q,s,c,timestamp=torch.tensor([15.,25.]))
    fused[0].square().sum().backward()
    assert p.grad is None and q.grad is not None
    assert m.perceptual_retrieval.retrieval_attn.in_proj_weight.grad.abs().sum()>0
    assert m.timestamps[:,:2].tolist()==[[10,15],[20,25]]


def test_values_are_raw_and_time_shift_does_not_change_age_attention():
    torch.manual_seed(7)
    m=EpisodicMemoryModule(8,0,4,bank_len=3)
    p,s=torch.randn(2,8),torch.randn(2,4)
    outputs=[]
    for offset in [0.,1000.]:
        m.reset(2,torch.device('cpu'))
        m.observe_only(p,s,timestamp=torch.full((2,),offset+10))
        captured=[]
        hook=m.perceptual_retrieval.register_forward_pre_hook(lambda module,args:captured.append(args[3].clone()))
        outputs.append(m(p+1,s,timestamp=torch.full((2,),offset+30))[0])
        hook.remove()
        torch.testing.assert_close(captured[0][:,0],p)
    torch.testing.assert_close(*outputs)


def test_merge_preserves_timestamps_and_padding_does_not_advance_clock():
    m=EpisodicMemoryModule(4,0,4,bank_len=2)
    m.reset(2,torch.device('cpu'))
    for t in [10.,20.,40.]:
        m.observe_only(torch.ones(2,4),torch.zeros(2,4),observed_mask=torch.tensor([True,False]),timestamp=torch.full((2,),t))
    assert m.timestamps[0].tolist()==[15.,40.]
    assert m._count.tolist()==[2,0] and m._next_timestep.tolist()==[41.,0.]


def test_long_diffusion_schedule_and_strict_legacy_load():
    head=DiffusionPolicyHead(cond_dim=4,hidden_dim=8,time_dim=8,num_train_timesteps=100,loss_repeats=4)
    assert head.sampling_timesteps().tolist()[0]==100 and len(head.sampling_timesteps())==10
    captured=[]
    head.predict_noise=lambda x,t,c:(captured.append((len(x),int(t.max()))) or torch.zeros_like(x))
    head.loss(torch.randn(2,8,7),torch.zeros(2,1,4))
    assert captured[0][0]==8
    legacy=DiffusionPolicyHead(cond_dim=4,hidden_dim=8,time_dim=8)
    head.load_state_dict(legacy.state_dict(),strict=True)
    assert head.num_train_timesteps==10 and len(head.alpha_bar)==11


class FakeEpisode:
    def _get_episode_length(self,ep):return 120
    def _read_frames_dinov2(self,ep,start,count):return np.arange(start,start+count,dtype=np.float32)[:,None,None]
    def _read_poses(self,ep,start,count):return np.zeros((count,6),np.float32)
    def _read_poses_gripper(self,ep,start,count):return np.zeros(count,np.float32)
    def _read_actions(self,ep,start,count):return np.repeat(np.arange(start,start+count,dtype=np.float32)[:,None],7,axis=1)


def test_episode_sampling_keeps_intervening_frames_and_future_targets():
    data=CachedEpisodes(FakeEpisode(),[0],20,42,False,'episode',16,8)
    sample=data[0]
    assert sample['segment_len']==120 and sample['loss_anchor_mask'].sum()==16
    assert np.flatnonzero(sample['loss_anchor_mask']).max()>90
    np.testing.assert_array_equal(sample['frames_segment'][:,0,0],np.arange(120))
    assert not sample['loss_anchor_mask'][113:].any()
    for t in np.flatnonzero(sample['loss_anchor_mask']):
        np.testing.assert_array_equal(sample['actions_segment'][t:t+8,0],np.arange(t,t+8))
    batch=collate_segments([sample,sample])
    assert batch['loss_anchor_mask'].dtype==torch.bool

@pytest.mark.parametrize('write_fused',[False,True])
def test_differentiable_write_ablation_survives_consolidation(write_fused):
    m=EpisodicMemoryModule(8,0,4,bank_len=2,detach_writes=False,write_fused=write_fused)
    m.reset(2,torch.device('cpu'))
    first=torch.randn(2,8,requires_grad=True)
    s=torch.randn(2,4)
    m(first,s)
    m(torch.randn(2,8),s)
    output=m(torch.randn(2,8),s)[0]
    output.square().sum().backward()
    assert first.grad is not None and first.grad.abs().sum()>0


def test_explicit_visual_attention_preserves_initial_function_and_learns():
    head=DiffusionPolicyHead(cond_dim=12,hidden_dim=8,time_dim=8,chunk_size=8)
    nn.init.normal_(head.unet.output_proj.weight,std=.02)
    x,t,cond=torch.randn(2,8,7),torch.ones(2,dtype=torch.long),torch.randn(2,1,12)
    reference=head.predict_noise(x,t,cond).detach()
    head.unet.configure_visual_attention(8,4)
    torch.testing.assert_close(head.predict_noise(x,t,cond),reference)
    opt=torch.optim.AdamW(head.parameters(),lr=.001)
    for step in range(2):
        loss=head.predict_noise(x,t,cond).square().mean()
        opt.zero_grad();loss.backward();opt.step()
    assert head.unet.visual_attention.in_proj_weight.grad.abs().sum()>0
    assert head.unet.visual_projection.weight.grad.abs().sum()>0


def test_sampling_clips_only_when_explicitly_configured():
    head=DiffusionPolicyHead(cond_dim=4,hidden_dim=8,time_dim=8,clip_denoised=True)
    head.predict_noise=lambda x,t,c:torch.zeros_like(x)
    action=head.sample(torch.zeros(2,1,4))
    assert action[:,:,:6].abs().max()<=1
    assert action[:,:,6].min()>=0 and action[:,:,6].max()<=1


def test_single_slot_consolidation_is_rejected():
    with pytest.raises(ValueError,match="at least two"):
        EpisodicMemoryModule(4,0,4,bank_len=1)


def test_stream_retains_intermediate_observations_without_double_write():
    from models.intention_stream import IntentionStream
    class Model:
        def __init__(self):
            self.memory_module=EpisodicMemoryModule(4,4,4,bank_len=3)
            self.memory_module.reset(1,torch.device('cpu'))
            self.readouts=[]
        def encode_step(self,p,s,cache,produce_intent):
            self.readouts.append(produce_intent)
            out=(p,s,s,cache)
            return out+(p.unsqueeze(1),) if produce_intent else out
    model=Model(); stream=IntentionStream(model,1)
    p=s=torch.ones(1,4)
    stream.observe(p,s,store_memory=True)
    out=stream.observe(p*2,s,produce_intent=True)
    assert model.memory_module._count.item()==1
    model.memory_module(out['z_v_pooled_seq'][:,0],out['z_s_seq'][:,0],out['intent_emb'])
    assert model.memory_module.timestamps[0,:2].tolist()==[0,1]
    assert model.readouts==[True,True]


def test_memory_rejects_replayed_or_future_leaking_timestamps():
    m=EpisodicMemoryModule(4,0,4,bank_len=3)
    m.reset(2,torch.device('cpu')); p=s=torch.ones(2,4)
    m.observe_only(p,s,timestamp=torch.tensor([10.,20.]))
    with pytest.raises(ValueError,match="strictly increasing"):
        m(p,s,timestamp=torch.tensor([10.,21.]))
    # Padded rows may repeat a timestamp without being stored.
    m.observe_only(p,s,observed_mask=torch.tensor([False,True]),timestamp=torch.tensor([10.,21.]))
    assert m._count.tolist()==[1,2]
    m.reset(2,torch.device('cpu'))
    m.observe_only(p,s,timestamp=torch.zeros(2))

@pytest.mark.parametrize('patch_dim',[None,2])
def test_batch_consolidation_chooses_each_episodes_pair_and_preserves_padding(patch_dim):
    m=EpisodicMemoryModule(4,0,4,bank_len=3,patch_dim=patch_dim)
    m.reset(3,torch.device('cpu'))
    observations=[[[1,0],[1,0],[1,0]],[[2,0],[0,1],[0,1]],[[0,1],[0,2],[9,9]],[[1,1],[1,1],[9,9]]]
    for t,values in enumerate(observations):
        p=torch.tensor([v+[0,0] for v in values],dtype=torch.float32)
        mask=torch.tensor([True,True,t<2])
        m.observe_only(p,p*10,observed_mask=mask)
    expected=torch.tensor([[[1.5,0,0,0],[0,1,0,0],[1,1,0,0]],
                           [[1,0,0,0],[0,1.5,0,0],[1,1,0,0]],
                           [[1,0,0,0],[0,1,0,0],[0,0,0,0]]])
    torch.testing.assert_close(m.perceptual_bank.reshape(3,3,4),expected)
    torch.testing.assert_close(m.state_bank,expected*10)
    assert m.timestamps.tolist()==[[.5,2,3],[0,1.5,3],[0,1,-1]]
    assert m._count.tolist()==[3,3,2]


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
@pytest.mark.parametrize('patch_dim',[None,4])
def test_empty_memory_is_bitwise_identity_under_bfloat16(patch_dim,device):
    m=EpisodicMemoryModule(8,4,4,bank_len=3,patch_dim=patch_dim).to(device)
    m.reset(2,torch.device(device))
    p,s,c=[torch.randn(*shape,device=device).bfloat16() for shape in [(2,8),(2,4),(2,1,4)]]
    with torch.amp.autocast(device,dtype=torch.bfloat16):
        result=m(p,s,c)
        second=m(p,s,c)
    assert all(torch.isfinite(x).all() for x in second)
    assert all(torch.equal(a,b) for a,b in zip(result,(p,s,c)))


def test_missing_view_training_is_deterministic_causal_and_validation_is_complete():
    class Episode(FakeEpisode):cameras=['image','wrist_image']
    data=CachedEpisodes(Episode(),[0],20,42,True,'episode',16,8,1.)
    sample=data[0];repeat=data[0]
    np.testing.assert_array_equal(sample['observation_camera_mask'],repeat['observation_camera_mask'])
    mask=sample['observation_camera_mask']
    assert mask[0].all()
    assert mask[~sample['loss_anchor_mask']].all()
    assert not mask[np.flatnonzero(sample['loss_anchor_mask'])[1:]].all(axis=1).any()
    np.testing.assert_array_equal(sample['actions_segment'],repeat['actions_segment'])
    validation=CachedEpisodes(Episode(),[0],20,42,False,'episode',16,8,1.)[0]
    assert 'observation_camera_mask' not in validation
    batch=collate_segments([sample,sample])
    assert batch['observation_camera_mask'].shape==(2,120,2)


def test_complete_observation_outages_hide_only_inputs_not_targets():
    class Episode(FakeEpisode):cameras=['image','wrist_image']
    data=CachedEpisodes(Episode(),[0],20,42,True,'episode',16,8,1.,True)
    sample=data[0]
    np.testing.assert_array_equal(sample['observation_state_mask'],sample['observation_camera_mask'].any(1))
    assert not sample['observation_state_mask'].all()
    np.testing.assert_array_equal(sample['actions_segment'][:,0],np.arange(120))
    assert sample['observation_state_mask'][0]
    assert collate_segments([sample,sample])['observation_state_mask'].shape==(2,120)


def test_context_only_retrieval_has_no_direct_query_shortcut():
    # With exactly one historical token, attention selection is fixed. Changing
    # the current query must not change the retrieved context branch.
    from models.memory_bank import MemoryRetrieval
    torch.manual_seed(11)
    retrieval=MemoryRetrieval(4,2).eval()
    retrieval.context_only=True
    values=torch.randn(2,1,4)
    query=torch.randn(2,4)
    first=retrieval(query,values)
    second=retrieval(query+torch.randn_like(query)*5,values)
    torch.testing.assert_close(first,second)
    changed=retrieval(query,values+torch.randn_like(values))
    assert not torch.allclose(first,changed)
    changed.square().sum().backward()
    assert retrieval.retrieval_attn.in_proj_weight.grad.abs().sum()>0


def test_context_only_is_optional_and_empty_bank_is_identity():
    old=EpisodicMemoryModule(8,0,4,bank_len=3)
    new=EpisodicMemoryModule(8,0,4,bank_len=3,context_only=True)
    new.load_state_dict(old.state_dict(),strict=True)
    assert not old.perceptual_retrieval.context_only
    assert new.perceptual_retrieval.context_only and new.state_retrieval.context_only
    new.reset(2,torch.device('cpu'))
    p,s=torch.randn(2,8),torch.randn(2,4)
    out=new(p,s)
    torch.testing.assert_close(out[0],p,rtol=0,atol=0)
    torch.testing.assert_close(out[1],s,rtol=0,atol=0)


def test_temporal_patch_lookup_preserves_camera_grid_slots():
    torch.manual_seed(17)
    memory=EpisodicMemoryModule(8,0,4,bank_len=3,patch_dim=4,
                               context_only=True,patch_temporal=True).eval()
    memory.reset(2,torch.device('cpu'))
    values=torch.randn(2,8)
    memory.observe_only(values,torch.randn(2,4),observed_mask=torch.tensor([True,False]))
    query=torch.randn(2,2,4)
    mask=torch.arange(3)[None]<memory._count[:,None]
    age=torch.ones(2,3)
    original=memory._retrieve(memory.perceptual_retrieval,query,memory.perceptual_bank,mask,age)
    changed_bank=memory.perceptual_bank.clone()
    changed_bank[0,0,0]+=torch.tensor([2.,-1.,3.,-4.])
    changed=memory._retrieve(memory.perceptual_retrieval,query,changed_bank,mask,age)
    assert not torch.allclose(original[0,0],changed[0,0])
    torch.testing.assert_close(original[0,1],changed[0,1],rtol=0,atol=0)
    torch.testing.assert_close(original[1],query[1],rtol=0,atol=0)
    changed[0].square().sum().backward()
    assert memory.perceptual_retrieval.retrieval_attn.in_proj_weight.grad.abs().sum()>0


def test_temporal_patch_mode_requires_patch_layout():
    with pytest.raises(ValueError,match='patch-preserving'):
        EpisodicMemoryModule(8,0,4,patch_temporal=True)


def test_value_preserving_retrieval_returns_raw_values_and_trains_selection():
    from models.memory_bank import MemoryRetrieval
    torch.manual_seed(23)
    module=MemoryRetrieval(4,2).eval();module.value_preserving=True
    query=torch.randn(2,4,requires_grad=True)
    keys=torch.randn(2,3,4)
    values=torch.randn(2,3,4)*3+7
    single_mask=torch.tensor([[True,False,False],[False,True,False]])
    out=module(query,keys,single_mask,values)
    torch.testing.assert_close(out,torch.stack([values[0,0],values[1,1]]),rtol=0,atol=0)
    selected=module(query,keys,torch.ones(2,3,dtype=torch.bool),values)
    selected.square().sum().backward()
    grad=module.retrieval_attn.in_proj_weight.grad
    assert grad[:8].abs().sum()>0 and grad[8:].abs().sum()==0
    assert module.ffn[-1].weight.grad is None and module.out_norm.weight.grad is None
    assert module.retrieval_attn.out_proj.weight.grad is None


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_raw_value_temporal_lookup_mixed_empty_rows_and_dtype(dtype):
    memory=EpisodicMemoryModule(8,0,4,bank_len=3,patch_dim=4,
                               patch_temporal=True,value_preserving=True).eval()
    memory.reset(2,torch.device('cpu'))
    p,s=torch.randn(2,8).to(dtype),torch.randn(2,4).to(dtype)
    memory.observe_only(p,s,observed_mask=torch.tensor([True,False]))
    query=torch.randn(2,2,4).to(dtype)
    mask=torch.arange(3)[None]<memory._count[:,None]
    with torch.autocast('cpu',dtype=dtype,enabled=dtype==torch.bfloat16):
        out=memory._retrieve(memory.perceptual_retrieval,query,memory.perceptual_bank,mask,torch.ones(2,3))
    assert out.dtype==dtype
    torch.testing.assert_close(out[0],p[0].reshape(2,4),rtol=0,atol=0)
    torch.testing.assert_close(out[1],query[1],rtol=0,atol=0)
    assert not any(x.requires_grad for x in memory.perceptual_retrieval.ffn.parameters())


def test_value_preserving_mode_rejects_conflicting_context_or_spatial_layout():
    with pytest.raises(ValueError,match='combine'):
        EpisodicMemoryModule(8,0,4,context_only=True,value_preserving=True)
    with pytest.raises(ValueError,match='alignment'):
        EpisodicMemoryModule(8,0,4,patch_dim=4,value_preserving=True)


@pytest.mark.parametrize('field_masks',[False,True])
def test_pre_state_visual_memory_stores_state_free_features_and_uses_live_state(monkeypatch,field_masks):
    import models.align_intention as module
    from models.align_intention import ALIGNIntentionModel
    from tests.test_intention_training_contracts import FakeVision
    monkeypatch.setattr(module,'VisionEncoder',FakeVision)
    torch.manual_seed(31)
    model=ALIGNIntentionModel(state_dim=4,mamba_output_dim=0,compressed_dim=4,
        num_cameras=1,use_memory_bank=True,memory_bank_len=3,
        memory_patch_retrieval=True,memory_patch_temporal=True,
        memory_value_preserving=True,memory_pre_state_visual=True,memory_field_masks=field_masks,
        head_d_model=8,chunk_size=8,history_size=1).eval()
    model._build_head_and_bank(16);model.memory_module.reset(2,torch.device('cpu'))
    patches=torch.randn(2,4,768)
    old_state,new_state=torch.randn(2,4),torch.randn(2,4)
    visual=model.encode_visual_features(patches,old_state)
    torch.testing.assert_close(visual,model.encode_visual_features(patches,new_state))
    model.condition_actions(visual.flatten(1)[:,None],old_state[:,None])
    torch.testing.assert_close(model.memory_module.perceptual_bank[:,0],visual)
    recalled=model.condition_actions(visual.flatten(1)[:,None],new_state[:,None])
    # Same image, changed gripper/pose: visual modulation must use the live
    # state, not the old state fused into the state-memory conditioning slot.
    expected=model.vision_patch_encoder(patches,new_state).flatten(1)[:,None]
    torch.testing.assert_close(recalled[0],expected)
    bypass=model.prepare_head_inputs(visual.flatten(1)[:,None],new_state[:,None])
    torch.testing.assert_close(bypass[0],expected)
    model.use_memory_bank=False
    torch.testing.assert_close(model.encode_visual_features(patches,new_state),expected[:,0].reshape(2,4,4))


def test_state_modulation_masks_missing_tokens_without_phantom_bias_features():
    from models.intention_encoder import StateConditionalCrossAttn
    torch.manual_seed(37)
    mod=StateConditionalCrossAttn(4,4,2).eval()
    mod.norm.bias.data.fill_(.8)
    visual=torch.randn(2,4,4);state=torch.randn(2,4)
    mask=torch.tensor([[True,True,False,False],[False,False,False,False]])
    first=mod(visual,state,token_mask=mask)
    changed=visual.clone();changed[~mask]=1000
    second=mod(changed,state,token_mask=mask)
    torch.testing.assert_close(first[mask],second[mask])
    assert torch.count_nonzero(first[~mask])==0 and torch.isfinite(first).all()


def test_field_masks_preserve_vision_during_long_state_only_observation_runs():
    bank=EpisodicMemoryModule(8,0,4,bank_len=16,patch_dim=4,
        patch_temporal=True,value_preserving=True,mask_missing_fields=True)
    bank.reset(1,torch.device('cpu'))
    p=torch.arange(1,9,dtype=torch.float32)[None];s=torch.ones(1,4)
    for t in range(16):bank.observe_only(p,s,timestamp=torch.tensor([float(t)]))
    for t in range(16,80):bank.observe_only(torch.zeros_like(p),s,timestamp=torch.tensor([float(t)]))
    valid=bank.perceptual_times>=0
    assert valid.any() and bank.perceptual_times.max()<16
    torch.testing.assert_close(bank.perceptual_bank[valid],p.reshape(2,4))
    query=torch.zeros(1,2,4)
    recalled=bank._retrieve(bank.perceptual_retrieval,query,bank.perceptual_bank,
        valid,80-bank.perceptual_times)
    torch.testing.assert_close(recalled,p.reshape(1,2,4))
    assert bank._next_timestep.item()==80


def test_field_merge_copies_single_valid_values_and_preserves_their_time():
    bank=EpisodicMemoryModule(4,0,4,bank_len=2,mask_missing_fields=True)
    bank.reset(1,torch.device('cpu'));p=torch.ones(1,4);s=2*p
    bank.observe_only(p,s,timestamp=torch.tensor([0.]))
    bank.observe_only(0*p,s,timestamp=torch.tensor([1.]))
    bank.observe_only(0*p,s,timestamp=torch.tensor([2.]))
    torch.testing.assert_close(bank.perceptual_bank[:,0],p)
    assert bank.perceptual_times.tolist()==[[0.,-1.]]
    assert bank.state_times.tolist()==[[.5,2.]]
    torch.testing.assert_close(bank.state_bank[:,0],s)


def test_field_masks_empty_stream_is_identity_even_with_other_stream_present():
    bank=EpisodicMemoryModule(8,4,4,bank_len=2,patch_dim=4,
        patch_temporal=True,value_preserving=True,mask_missing_fields=True)
    bank.reset(1,torch.device('cpu'))
    bank.observe_only(torch.zeros(1,8),torch.ones(1,4),torch.zeros(1,4))
    p=torch.randn(1,8);s=torch.randn(1,4);c=torch.randn(1,4)
    actual=bank(p,s,c)
    assert torch.equal(actual[0],p) and torch.equal(actual[2],c)


def test_field_masks_average_two_valid_observations_and_reset_masks():
    bank=EpisodicMemoryModule(4,0,4,bank_len=2,mask_missing_fields=True)
    bank.reset(1,torch.device('cpu'));p=torch.ones(1,4)
    for t,k in enumerate([1,3,5]):bank.observe_only(k*p,p,timestamp=torch.tensor([float(t)]))
    torch.testing.assert_close(bank.perceptual_bank[:,0],2*p)
    assert bank.perceptual_times.tolist()==[[.5,2.]]
    bank.reset(2,torch.device('cpu'))
    assert bank.perceptual_times.shape==(2,2) and (bank.perceptual_times==-1).all()


def test_missing_patch_slots_are_excluded_and_empty_slots_keep_their_query():
    bank=EpisodicMemoryModule(8,0,4,bank_len=2,patch_dim=4,
        patch_temporal=True,value_preserving=True,mask_missing_fields=True)
    bank.reset(1,torch.device('cpu'))
    bank.observe_only(torch.tensor([[1.,2.,3.,4.,0.,0.,0.,0.]]),torch.ones(1,4))
    query=torch.randn(1,8)
    out=bank(query,torch.ones(1,4))[0].reshape(1,2,4)
    assert torch.equal(out[:,1],query.reshape(1,2,4)[:,1])
    assert bank.perceptual_times[:,0].tolist()==[[0.,-1.]]


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_field_masks_full_observation_parity_and_bfloat16_writes(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(42)
    kwargs=dict(perceptual_dim=8,cognitive_dim=4,state_dim=4,bank_len=4,
        patch_dim=4,patch_temporal=True,value_preserving=True)
    old=EpisodicMemoryModule(**kwargs).to(device).eval()
    new=EpisodicMemoryModule(**kwargs,mask_missing_fields=True).to(device).eval()
    new.load_state_dict(old.state_dict())
    for bank in [old,new]:bank.reset(2,torch.device(device))
    with torch.autocast(device_type=device,dtype=torch.bfloat16),torch.no_grad():
        for t in range(12):
            p=torch.randn(2,8,device=device).bfloat16()
            s=torch.randn(2,4,device=device).bfloat16();c=torch.randn_like(s)
            a,z=old(p,s,c),new(p,s,c)
            for first,second in zip(a,z):torch.testing.assert_close(first,second)
    torch.testing.assert_close(new.perceptual_times,old.timestamps[:,:,None].expand(2,4,2))
    new.observe_only(torch.zeros_like(p),s,c)
    assert torch.isfinite(new.perceptual_bank).all()
