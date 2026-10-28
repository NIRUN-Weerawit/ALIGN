"""Frozen contextual language, policy gradients, and legacy text compatibility."""
import copy
from types import SimpleNamespace
import pytest
import torch

from models.task_text_encoder import DistilBertTaskEncoder


@pytest.fixture
def pretrained(monkeypatch):
    from transformers import AutoTokenizer, AutoModel, DistilBertConfig, DistilBertModel
    calls=[]
    class Tokenizer:
        def __call__(self,texts,**kwargs):
            calls.append(list(texts))
            rows=[[1]+[3+sum(map(ord,w))%50 for w in text.split()]+[2] for text in texts]
            rows=[r[:kwargs['max_length']] for r in rows]
            ids=torch.zeros(len(rows),max(map(len,rows)),dtype=torch.long)
            for i,row in enumerate(rows):ids[i,:len(row)]=torch.tensor(row)
            return dict(input_ids=ids,attention_mask=ids.ne(0).long())
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**k:Tokenizer())
    def load(*args,**kwargs):
        return DistilBertModel(DistilBertConfig(vocab_size=64,dim=24,hidden_dim=48,
            n_layers=1,n_heads=4,max_position_embeddings=32,dropout=.5,attention_dropout=.5))
    monkeypatch.setattr(AutoModel,'from_pretrained',load)
    return calls


def test_frozen_contextual_features_and_trainable_projection(pretrained):
    encoder=DistilBertTaskEncoder(8,max_length=32)
    encoder.train()
    assert not encoder.backbone.training
    texts=['put bowl stove','put stove bowl','unseen ceramic vessel']
    out=encoder(texts)
    assert out.shape==(3,8) and not torch.equal(out[0],out[1])
    out.square().sum().backward()
    assert encoder.projection.weight.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in encoder.backbone.parameters())
    # Batch padding cannot change the frozen text feature.
    cached=out.detach().clone();encoder._features.clear()
    single=encoder([texts[0]])
    torch.testing.assert_close(single[0],cached[0],atol=1e-6,rtol=1e-5)


def test_cache_retains_raw_features_not_projected_outputs(pretrained):
    encoder=DistilBertTaskEncoder(8,max_length=32,cache_size=2)
    first=encoder(['a','a','b','c'])
    assert pretrained==[['a','b','c']]
    assert len(encoder._features)==2
    before=encoder(['b']).detach()
    with torch.no_grad():encoder.projection.bias.add_(1)
    after=encoder(['b'])
    torch.testing.assert_close(after,before+1)
    assert len(pretrained)==1
    state=copy.deepcopy(encoder.state_dict())
    encoder.load_state_dict(state)
    assert not encoder._features
    encoder(['b']);assert len(pretrained)==2


@pytest.mark.parametrize('head_type', ['diffusion', 'flow_matching'])
def test_distilbert_conditions_head_and_roundtrips(pretrained,monkeypatch,head_type):
    import models.align_intention as align
    from tests.test_task_context_policy import FakeVision
    monkeypatch.setattr(align,'VisionEncoder',FakeVision)
    def build():
        m=align.ALIGNIntentionModel(state_dim=4,num_cameras=1,compressed_dim=4,
            mamba_output_dim=0,head_type=head_type,head_d_model=8,chunk_size=4,
            use_text=True,text_dim=8,text_encoder_type='distilbert',text_max_length=32)
        m._build_head_and_bank(1024)
        return m
    model=build().train();v=torch.randn(2,1,1024);s=torch.randn(2,1,4)
    optim=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
    for step in range(2):
        optim.zero_grad()
        language=model.encode_task_text(['put bowl stove','put stove bowl'])
        condition=model.action_condition(v,s,text_emb=language)
        condition.square().mean().backward()
        assert model.text_condition.weight.grad.abs().sum()>0
        if step:assert model.text_encoder.projection.weight.grad.abs().sum()>0
        optim.step()
    model.eval();language=model.encode_task_text(['put bowl stove','put stove bowl'])
    expected=model.action_condition(v,s,text_emb=language).detach()
    other=build().eval();other.load_state_dict(model.state_dict())
    actual=other.action_condition(v,s,text_emb=other.encode_task_text(['put bowl stove','put stove bowl']))
    torch.testing.assert_close(actual,expected)
    assert not torch.equal(expected[0],expected[1])
    model.unfreeze_all()
    assert not any(p.requires_grad for p in model.text_encoder.backbone.parameters())


def test_cli_defaults_and_explicit_legacy_text():
    from scripts.run_intention_ablation import parse_args
    args=parse_args(['--output','/tmp/not-created','--use-task-text'])
    assert args.text_encoder_type=='distilbert'
    args=parse_args(['--output','/tmp/not-created','--use-task-text','--text-encoder-type','bag'])
    assert args.text_encoder_type=='bag'
