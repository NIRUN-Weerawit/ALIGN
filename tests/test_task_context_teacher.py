"""Source provenance, causal controls, padding isolation, and real gradients."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from data.task_context_dataset import (build_targets, spatial_summary, geometric_state,
    TaskContextEpisodes, collate_task_context, json_hash)
from models.task_context_teacher import TaskContextTeacher, teacher_loss


@pytest.fixture
def targets(tmp_path):
    data=tmp_path/'data.h5';cache=tmp_path/'cache';cache.mkdir()
    index={'__meta__':dict(format='align-dinov2-per-camera-v2',source=str(data),cameras=['image'])}
    rng=np.random.default_rng(5)
    with h5py.File(data,'w') as h5:
        for i in range(8):
            key=f'ep_{i:06d}';g=h5.create_group(key)
            g['poses']=np.zeros((9,6),dtype='f4')+i*.01
            action=rng.normal(0,.1,(9,7)).astype('f4');action[:,6]=np.arange(9)%2
            g['actions']=action;g['texts']=json.dumps([f'objective-{i%2}'])
            np.save(cache/f'{key}.npy',rng.normal(size=(9,257,768)).astype('f4'))
            index[key]=dict(length=9,tokens=257,dim=768)
    (cache/'index.json').write_text(json.dumps(index))
    output=tmp_path/'targets'
    build_targets(data,cache,output,cameras=['image'],fps=20,action_horizon=4,procedure_points=3,terminal_tail=2)
    return output,data,cache


def test_target_split_hashes_and_unknown_success_are_explicit(targets):
    root,data,cache=targets
    ds=TaskContextEpisodes(root,'train',anchors=2,verify=True)
    assert len(ds)==4
    assert all(e['success'] is None and not e['terminal_approved'] for e in ds.manifest['episodes'].values())
    key=ds.keys[0]
    with h5py.File(data) as f:
        raw=f[key]['actions'][()]
    values=np.load(root/f'{key}.npz')
    assert values['states'][0,-1]==0
    np.testing.assert_array_equal(values['states'][1:,-1],raw[:-1,6])
    original=np.load(root/f'{key}.npy')
    np.testing.assert_allclose(values['terminal'],original[-2:].mean(0))
    assert np.isclose(values['times'][1],.05)
    manifest=ds.manifest
    manifest['split']['confirmation'][0]=manifest['split']['train'][0]
    manifest['manifest_sha256']=json_hash({k:v for k,v in manifest.items() if k!='manifest_sha256'})
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='overlap'):
        TaskContextEpisodes(root,'train')


def test_cache_camera_mismatch_is_rejected_before_output(targets,tmp_path):
    _,data,cache=targets
    with pytest.raises(ValueError,match='camera order'):
        build_targets(data,cache,tmp_path/'bad',cameras=['wrist'],fps=20)
    assert not (tmp_path/'bad').exists()


def test_normalization_fits_only_training_episodes(targets):
    root,_,_=targets
    ds=TaskContextEpisodes(root,'train')
    train=np.concatenate([np.load(root/f'{k}.npz')['states'] for k in ds.keys])
    np.testing.assert_allclose(ds.norm['state_mean'],train.mean(0))
    key=ds.manifest['split']['confirmation'][0]
    # An altered held-out target never causes normalization to be refitted.
    with (root/f'{key}.npz').open('ab') as f:f.write(b'modified')
    with pytest.raises(ValueError,match='hash mismatch'):
        TaskContextEpisodes(root,'confirmation',verify=True)


def test_spatial_slots_and_rotation_equivalence():
    raw=np.zeros((1,257,4),dtype='f4')
    patches=raw[:,:256].reshape(1,16,16,4)
    for row in range(2):
        for col in range(2):patches[:,row*8:(row+1)*8,col*8:(col+1)*8,row*2+col]=1
    summary=spatial_summary(raw,1)
    assert summary[0,0,1:].argmax(-1).tolist()==[0,1,2,3]
    poses=np.zeros((2,6),dtype='f4');poses[:,3]=[3.2,3.2-2*np.pi]
    state=geometric_state(poses,np.zeros(2))
    np.testing.assert_allclose(state[0],state[1],atol=2e-7)


def toy():
    torch.set_num_threads(2);torch.manual_seed(42)
    model=TaskContextTeacher(cameras=1,raw_dim=8,width=32,tokens=4,token_dim=8,
        layers=1,heads=4,horizon=4,procedure_points=3,diffusion_width=8)
    batch=dict(features=torch.randn(3,7,1,5,8),states=torch.randn(3,7,10),
        times=torch.arange(7.).expand(3,-1),lengths=torch.tensor([7,5,6]),
        anchors=torch.tensor([[0,2],[0,1],[1,2]]),actions=torch.randn(3,2,4,7),
        terminal=torch.randn(3,1,5,8),procedure=torch.randn(3,3,1,5,8),
        procedure_states=torch.randn(3,3,10),group=torch.tensor([0,0,1]))
    return model,batch


@pytest.mark.parametrize('mode',['full','current','constant'])
def test_teacher_real_optimizer_and_checkpoint_roundtrip(mode):
    model,batch=toy();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    loss,_=teacher_loss(model,model(batch,mode),batch)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.task_condition[1].weight.grad.abs().sum()>0
    if mode!='constant':assert model.summary.cls[0].weight.grad.abs().sum()>0
    optimizer.step()
    clone=TaskContextTeacher(**model.config);clone.load_state_dict(model.state_dict(),strict=True)
    model.eval();clone.eval()
    torch.testing.assert_close(model(batch,mode)['tokens'],clone(batch,mode)['tokens'])


def test_padding_nan_isolated_and_targets_and_labels_are_not_model_inputs():
    model,batch=toy();model.eval()
    expected=model(batch)['tokens']
    for key in ['features','states','times']:
        batch[key]=batch[key].clone()
        for i,n in enumerate(batch['lengths']):batch[key][i,n:]=float('nan')
    batch['group']=torch.tensor([8,9,10]);batch['actions'].fill_(float('nan'))
    batch['terminal'].fill_(float('nan'));batch['procedure'].fill_(float('nan'))
    torch.testing.assert_close(model(batch)['tokens'],expected)


def test_current_control_cannot_read_other_frames():
    model,batch=toy();model.eval()
    before=model(batch,'current')['tokens']
    batch['features'][:,-2:]=1000
    batch['states'][:,-2:]=-1000
    torch.testing.assert_close(model(batch,'current')['tokens'],before)
    assert not torch.equal(model(batch,'full')['tokens'],model(batch,'current')['tokens'])
    for mode in ['current','constant']:
        original=model(batch,mode)['tokens']
        model.config['pooling']='quarters'
        torch.testing.assert_close(model(batch,mode)['tokens'],original,rtol=0,atol=0)
        model.config['pooling']='learned'


def test_ordered_pooling_handles_short_episodes_and_padding():
    model,batch=toy()
    model.config['pooling']='quarters';model.eval()
    batch['lengths']=torch.tensor([7,1,3])
    batch['anchors'][1]=0;batch['anchors'][2]=0
    expected=model(batch,'full')['tokens']
    assert torch.isfinite(expected).all()
    for name in ['features','states','times']:
        batch[name]=batch[name].clone()
        for i,n in enumerate(batch['lengths']):batch[name][i,n:]=float('nan')
    torch.testing.assert_close(model(batch,'full')['tokens'],expected)


def test_temporal_target_decoder_has_gradients_and_strict_roundtrip():
    model,batch=toy()
    model=TaskContextTeacher(**dict(model.config,pooling='quarters',decoder_style='temporal'))
    loss,_=teacher_loss(model,model(batch),batch)
    loss.backward()
    assert model.local_decoder[0].weight.grad.abs().sum()>0
    clone=TaskContextTeacher(**model.config);clone.load_state_dict(model.state_dict(),strict=True)
    model.eval();clone.eval()
    torch.testing.assert_close(model(batch)['procedure'],clone(batch)['procedure'])


def test_actual_dataset_collation_has_no_target_repetition(targets):
    root,_,_=targets
    ds=TaskContextEpisodes(root,'development',anchors=3)
    batch=collate_task_context([ds[0],ds[1]])
    assert batch['features'].shape==(2,9,1,5,768)
    assert batch['actions'].shape==(2,3,4,7)
    assert (batch['anchors']+4<=batch['lengths'][:,None]).all()


def test_assessment_pairs_episodes_and_rejects_changed_confirmation_checkpoint(tmp_path):
    from tools.assess_task_context_teacher import paired_difference, donor_indices, validate_confirmation
    from data.task_context_dataset import file_hash
    same,wrong=donor_indices(np.array([0,0,1,1]))
    group=np.array([0,0,1,1])
    assert np.all(group[same]==group) and np.all(group[wrong]!=group)
    assert np.all(same!=np.arange(4))
    result=paired_difference(np.ones(5),np.zeros(5),repeats=100)
    assert result['mean']==1 and result['ci95']==[1,1]
    p=tmp_path/'checkpoint';p.write_bytes(b'original')
    report=dict(split='development',gate=dict(passed=True),manifest_sha256='abc',
        checkpoints=dict(full=dict(sha256=file_hash(p))))
    validate_confirmation(report,{'full':p},'abc')
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint changed'):
        validate_confirmation(report,{'full':p},'abc')


def test_export_rejects_development_only_approval_before_loading_weights(tmp_path,monkeypatch):
    from tools.export_task_context_teacher import main
    report=tmp_path/'gate.json'
    report.write_text(json.dumps(dict(split='development',gate=dict(passed=True))))
    output=tmp_path/'latents'
    monkeypatch.setattr('sys.argv',['export_task_context_teacher.py','--targets',str(tmp_path),
        '--checkpoint',str(tmp_path/'does_not_exist.pt'),'--gate-report',str(report),'--output',str(output),'--device','cpu'])
    with pytest.raises(ValueError,match='passing confirmation'):
        main()
    assert not output.exists()
