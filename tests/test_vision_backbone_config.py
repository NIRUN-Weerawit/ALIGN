"""Backbone contracts and real downstream 1024-D paths without pretrained weights."""
import sys

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from models.vision_config import feature_width, cache_spec, validate_cache
from scripts import precompute_dinov2 as precompute


@pytest.mark.parametrize('backbone,width',[('dinov2_vitb14',768),('dinov2_vitl14',1024)])
def test_actual_vision_wrapper_uses_backbone_width(monkeypatch,backbone,width):
    from models.align_model import VisionEncoder
    class Backbone(nn.Module):
        embed_dim=width
        def forward_features(self,images):
            return dict(x_norm_patchtokens=images.new_ones(len(images),256,width),
                        x_norm_clstoken=images.new_full((len(images),width),2))
    names=[]
    def load(repo,name,pretrained):
        names.append(name)
        return Backbone()
    monkeypatch.setattr(torch.hub,'load',load)
    encoder=VisionEncoder(backbone=backbone,num_cameras=2,fusion_type='linear')
    result=encoder(torch.zeros(1,2,224,224,3,dtype=torch.uint8))
    assert names==[backbone] and result.shape==(1,514,width)
    assert result[0,256,0]==2 and result[0,513,0]==2
    assert not result.requires_grad


def test_legacy_cache_and_mismatch_rejection():
    assert cache_spec({})==('dinov2_vitb14',768)
    with pytest.raises(ValueError,match='Cache'):
        validate_cache({},'dinov2_vitl14',1024)
    with pytest.raises(ValueError,match='1024-D'):
        feature_width('dinov2_vitl14',768)
    assert cache_spec(dict(backbone='dinov2_vitl14',feature_dim=1024))==('dinov2_vitl14',1024)


def test_vitl_cached_stream_and_sequence_match_and_backprop(monkeypatch):
    import models.align_intention as align
    # No image forward is needed: features enter through the cache interface.
    class CachedOnlyVision(nn.Module):
        def __init__(self,**kwargs):
            super().__init__()
            self.marker=nn.Parameter(torch.zeros(1))
    monkeypatch.setattr(align,'VisionEncoder',CachedOnlyVision)
    model=align.ALIGNIntentionModel(vision_backbone='dinov2_vitl14',raw_dim=1024,
        num_cameras=2,state_dim=4,compressed_dim=4,mamba_output_dim=0,
        head_type='diffusion',head_d_model=8,chunk_size=4)
    features=torch.randn(1,2,514,1024);states=torch.randn(1,2,7)
    model.eval()
    sequence=model(features,states)
    for t in range(2):
        visual,state,*_=model.encode_step(features[:,t],states[:,t])
        torch.testing.assert_close(visual,sequence['z_v_pooled_seq'][:,t])
        torch.testing.assert_close(state,sequence['z_s_seq'][:,t])
    sequence['z_v_pooled_seq'].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.vision_patch_encoder.parameters())


def test_vitl_cache_dry_run_never_creates_output_or_loads_weights(tmp_path,monkeypatch):
    source=tmp_path/'tiny.h5'
    with h5py.File(source,'w') as f:
        f.create_dataset('ep_000000/frames/image',data=np.zeros((1,2,2,3),dtype=np.uint8))
    output=tmp_path/'not_created'
    monkeypatch.setattr(precompute,'build_encoder',lambda *args: pytest.fail('Must not load backbone'))
    monkeypatch.setattr(sys,'argv',['precompute','--data',str(source),'--cameras','image',
        '--output',str(output),'--device','cpu','--dry-run'])
    precompute.main()
    assert not output.exists()


def test_vitl_teacher_reconstructs_1024d_targets():
    from models.task_context_teacher import TaskContextTeacher
    model=TaskContextTeacher(cameras=1,raw_dim=1024,width=16,heads=4,layers=1,
        diffusion_width=8,pooling='quarters',decoder_style='temporal')
    batch=dict(features=torch.randn(1,4,1,5,1024),states=torch.randn(1,4,10),
        times=torch.arange(4)[None].float(),lengths=torch.tensor([4]),anchors=torch.tensor([[0,2]]))
    result=model(batch)
    assert result['tokens'].shape==(1,2,4,128)
    assert result['terminal'].shape==(1,2,1,5,1024)
    result['terminal'].square().mean().backward()
    assert model.summary.cls[0].weight.grad is not None


def test_new_training_defaults_to_large_and_explicit_legacy_remains_available():
    from scripts.run_intention_ablation import parse_args
    large=parse_args(["--output","/tmp/not-created"])
    assert large.vision_backbone=="dinov2_vitl14" and large.raw_dim==1024
    base=parse_args(["--output","/tmp/not-created","--vision-backbone","dinov2_vitb14"])
    assert base.raw_dim==768
