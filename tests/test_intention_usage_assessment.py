"""Protect causal counterfactuals and meaningful intervention controls."""
import numpy as np
import pytest
import torch

import tools.assess_intention_usage as assessment


def test_counterfactual_history_never_reads_future_and_keeps_current_last():
    cls=torch.arange(8.).reshape(8,1,1);state=torch.arange(8.).reshape(8,1)
    recent,_=assessment.prefix_inputs(cls,state,4,window=3)
    assert recent.flatten().tolist()==[2,3,4]
    reverse,_=assessment.prefix_inputs(cls,state,4,mode='reverse_past')
    assert reverse.flatten().tolist()==[3,2,1,0,4]
    repeat,_=assessment.prefix_inputs(cls,state,4,mode='repeat_current')
    assert repeat.flatten().tolist()==[4]*5


def test_donors_are_distinct_episodes_with_required_task_relation():
    labels=np.array(['a','b','a','b','c','c'])
    for same in [True,False]:
        indices=assessment.donor_indices(labels,42,same)
        assert np.all(indices!=np.arange(len(labels)))
        assert np.all((labels[indices]==labels)==same)
        np.testing.assert_array_equal(indices,assessment.donor_indices(labels,42,same))
    with pytest.raises(ValueError):assessment.donor_indices(['a','b'],42,True)


def test_prefix_readout_hidden_capture_uses_each_prefix_only():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.intent_proj=torch.nn.Identity()
        def forward(self,cls,state):
            h=cls.flatten(2).cumsum(1)+state.cumsum(1)
            return h,self.intent_proj(h[:,-1:])
    cls=torch.arange(1.,7.).reshape(6,1,1);state=torch.zeros(6,1)
    full,pre,obs=assessment.readouts(Encoder(),cls,state,[0,2,4],capture=True)
    np.testing.assert_array_equal(full,[[1],[6],[15]])
    np.testing.assert_array_equal(pre,full);np.testing.assert_array_equal(obs,full)
    recent=assessment.readouts(Encoder(),cls,state,[0,2,4],window=2)
    np.testing.assert_array_equal(recent,[[1],[5],[9]])


def test_paired_interval_keeps_episode_level_differences():
    assert assessment.paired_interval(np.zeros(10),42)==[0.,0.]
    assert assessment.paired_interval(np.ones(10)*.5,42)==[.5,.5]


def test_cached_policy_loader_requires_complete_trained_head(tmp_path):
    from models.intention_encoder import HAS_MAMBA
    if not HAS_MAMBA:pytest.skip('Requires Mamba constructor')
    config=dict(use_intent_tokens=True,use_memory_bank=False,head_type='diffusion',
        history_size=1,num_cameras=1,state_dim=4,raw_dim=4,compressed_dim=4,
        mamba_output_dim=4,intent_dim=4,num_intent_tokens=1,head_d_model=16,
        chunk_size=8,diffusion_train_steps=10)
    with assessment.patch('models.align_intention.VisionEncoder',lambda **kw:torch.nn.Linear(1,1,bias=False)):
        original=assessment.ALIGNIntentionModel(action_dim=7,**config)
    original._build_head_and_bank(1024)
    weights=original.state_dict();path=tmp_path/'checkpoint.pt'
    torch.save(dict(config=config,model_state_dict=weights,epoch=2),path)
    restored,_,epoch,digest=assessment.load_policy(path,torch.device('cpu'))
    assert epoch==2 and len(digest)==64
    torch.testing.assert_close(restored.intention_head.unet.input_proj.weight,
                               original.intention_head.unet.input_proj.weight)
    del weights['intention_head.unet.input_proj.weight']
    torch.save(dict(config=config,model_state_dict=weights,epoch=2),path)
    with pytest.raises(RuntimeError,match='Missing key'):
        assessment.load_policy(path,torch.device('cpu'))
