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
