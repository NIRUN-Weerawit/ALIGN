"""Protect causal extraction, exact checkpoint loading and probe split isolation."""
import numpy as np
import pytest
import torch

import tools.probe_intent_clustering as probe


def test_episode_split_rejects_overlap_and_unknown_keys():
    cfg=dict(train_episodes=['train'],val_episodes=['val'])
    assert probe.validate_split(cfg,['train','val']) == (['train'],['val'])
    with pytest.raises(ValueError):
        probe.validate_split(dict(train_episodes=['same'],val_episodes=['same']),['same'])
    with pytest.raises(ValueError):
        probe.validate_split(cfg,['train'])


def test_prefix_features_are_causal_and_reset_control_has_no_history():
    class CumulativeEncoder:
        def forward_sequence(self, cls, states):
            signal=cls.sum((-1,-2))+states.sum(-1)
            # Two distinct token slots verify concatenation, rather than mean.
            return torch.stack([signal.cumsum(1),2*signal.cumsum(1)],-1).unsqueeze(-1)
    cls=torch.arange(1,7,dtype=torch.float32).reshape(6,1,1)
    states=torch.zeros(6,1)
    times=np.array([0,2,4])
    full,reset=probe.prefix_readouts(CumulativeEncoder(),cls,states,times,torch.device('cpu'))
    np.testing.assert_allclose(full,[[1,2],[6,12],[15,30]])
    np.testing.assert_allclose(reset,[[1,2],[3,6],[5,10]])
    cls[5]=10000
    changed,_=probe.prefix_readouts(CumulativeEncoder(),cls,states,times,torch.device('cpu'))
    np.testing.assert_array_equal(changed,full)


def test_loader_strictly_restores_current_checkpoint_encoder_keys(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'RobotStateEncoder',lambda **kwargs: torch.nn.Linear(7,2))
    def tiny_encoder(config=None):
        module=torch.nn.Linear(3,2)
        module.register_buffer('scalar',torch.tensor(1.))
        return module
    monkeypatch.setattr(probe,'make_encoder',tiny_encoder)
    state=torch.nn.Linear(7,2);encoder=tiny_encoder()
    cfg=dict(use_intent_tokens=True,state_dim=2,mamba_output_dim=1)
    weights={**{'state_encoder.'+k:v for k,v in state.state_dict().items()},
             **{'intention_encoder.'+k:v for k,v in encoder.state_dict().items()},
             'intention_head.ignored':torch.ones(1)}
    path=tmp_path/'model.pt'
    torch.save(dict(config=cfg,model_state_dict=weights,epoch=3),path)
    actual,_,_,metadata=probe.load_encoders(path,torch.device('cpu'))
    torch.testing.assert_close(actual.weight,encoder.weight)
    assert metadata['epoch']==3 and len(metadata['encoder_sha256'])==64
    del weights['intention_encoder.bias']
    torch.save(dict(config=cfg,model_state_dict=weights),path)
    with pytest.raises(RuntimeError,match='Missing key'):
        probe.load_encoders(path,torch.device('cpu'))


def test_decoder_scaling_uses_training_episodes_only():
    train=np.array([[-2.,0.],[-1.,1.],[1.,0.],[2.,1.]])
    val=np.array([[-100.,0.],[100.,1.]])
    decoder,pred=probe.fit_decoder(train,np.array([0,0,1,1]),val)
    np.testing.assert_allclose(decoder.named_steps['standardscaler'].mean_,train.mean(0))
    np.testing.assert_array_equal(pred,[0,1])
