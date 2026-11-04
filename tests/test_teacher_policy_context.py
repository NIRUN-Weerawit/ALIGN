import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.teacher_policy_context import load_policy_teacher_targets, replace_task_tokens
from data.task_context_dataset import file_hash
from tests.test_task_context_policy import bundle, make_model
from tests.test_sparse_plain_history import Features
from scripts.run_intention_ablation import CachedEpisodes, collate_segments, parse_args
from training.cached_loading import collate_cached_segments
from training.train_intention import train_v4_epoch, validate


@pytest.fixture
def targets(bundle, tmp_path):
    b = torch.load(bundle, weights_only=True)
    b['metadata'].update(teacher_sha256='teacher', target_manifest_sha256='manifest')
    torch.save(b, bundle)
    root = tmp_path / 'labels'; root.mkdir()
    partitions = dict(train=['a'], development=['b'], confirmation=['c'])
    files = {}
    for n, (part, keys) in enumerate(partitions.items()):
        path = root / f'{part}.npz'
        np.savez(path, keys=np.array(keys), task_tokens=np.full((1,4,4), n, np.float32))
        files[part] = dict(episodes=1, sha256=file_hash(path))
    manifest = dict(schema='align-privileged-task-tokens-v1', teacher_sha256='teacher',
        target_manifest_sha256='manifest', shape=[4,4], splits=files)
    (root/'manifest.json').write_text(json.dumps(manifest))
    split = tmp_path/'split.json'
    split.write_text(json.dumps(dict(manifest_sha256='manifest', split=partitions)))
    args = SimpleNamespace(task_context_source='teacher', task_context_checkpoint=str(bundle),
        task_context_teacher_targets=str(root), task_context_split_manifest=str(split))
    return args, root, manifest


def test_bank_provenance_partitions_and_hashes(targets):
    args, root, manifest = targets
    dataset = SimpleNamespace(_episode_keys=['c','a','b'])
    bank = load_policy_teacher_targets(args, dataset, [1], [2])
    np.testing.assert_array_equal(bank['b'], np.ones((4,4)))
    assert args.task_context_teacher_provenance['teacher_sha256'] == 'teacher'
    with pytest.raises(ValueError, match='Policy episodes'):
        load_policy_teacher_targets(args, dataset, [0], [2])
    manifest['teacher_sha256'] = 'wrong'
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='teacher differ'):
        load_policy_teacher_targets(args, dataset, [1], [2])
    manifest['teacher_sha256'] = 'teacher'
    (root/'manifest.json').write_text(json.dumps(manifest))
    with (root/'train.npz').open('ab') as f: f.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_policy_teacher_targets(args, dataset, [1], [2])


@pytest.mark.parametrize('collate', [collate_segments, collate_cached_segments])
def test_batch_episode_identity_and_static_targets(collate):
    data = Features(); data._episode_keys=['a','b']
    bank = dict(a=np.ones((4,4),np.float32), b=np.full((4,4),7,np.float32))
    episodes = CachedEpisodes(data, [1,0], 20, 42, True, 'episode', 4, 4, teacher_targets=bank)
    batch = collate([episodes[0],episodes[1]])
    assert batch['teacher_task_tokens'].shape == (2,4,4)
    assert batch['teacher_task_tokens'][0].eq(7).all()
    assert batch['teacher_task_tokens'][1].eq(1).all()
    native = torch.randn(2,17,6,4,requires_grad=True)
    replaced = replace_task_tokens(native, batch['teacher_task_tokens'], 4)
    assert replaced[0,:,:4].eq(7).all()
    torch.testing.assert_close(replaced[:,:,4:],native[:,:,4:])
    replaced.sum().backward()
    assert native.grad[:,:,:4].eq(0).all() and native.grad[:,:,4:].eq(1).all()


@pytest.mark.parametrize('head', ['diffusion','flow_matching'])
@pytest.mark.parametrize('memory', [False,True])
def test_train_and_validation_use_fixed_teacher_plus_live_execution(monkeypatch,bundle,head,memory):
    model = make_model(monkeypatch,bundle,memory=memory,head_type=head)
    model.task_context_source='teacher'
    data=Features(); data._episode_keys=['a','b']
    bank=dict(a=np.ones((4,4),np.float32),b=np.full((4,4),7,np.float32))
    ds=CachedEpisodes(data,[0,1],20,42,False,'episode',2,4,teacher_targets=bank)
    batch=collate_cached_segments([ds[0],ds[1]])
    batch['loss_anchor_mask'][:]=False
    batch['loss_anchor_mask'][:,[0,6]]=True
    calls=[]
    hook=model.intention_head.register_forward_pre_hook(lambda module,args:calls.append(args[2].detach().clone()))
    model.intention_head.loss=lambda target,cond,**kwargs: cond.square().mean()
    model.intention_head.sample=lambda cond: cond.new_zeros((len(cond),4,7))
    args=SimpleNamespace(history_size=1,chunk_size=4,action_dim=7,head_type=head,
        skip_nan=False,grad_clip=1.,no_sample_during_train=True,head_window_batch_size=2,gripper_threshold=.5)
    optimizer=torch.optim.SGD(model.parameters(),lr=0.)
    train_v4_epoch(model,[batch],optimizer,torch.device('cpu'),args)
    validate(model,[batch],torch.device('cpu'),args)
    hook.remove()
    assert calls
    for context in calls:
        assert context[0,:4].eq(1).all() and context[1,:4].eq(7).all()
    assert not torch.equal(calls[0][:,-1], calls[1][:,-1])
    assert all(p.grad is None for p in model.task_context.parameters())
    bad=dict(batch);bad.pop('teacher_task_tokens')
    with pytest.raises(ValueError,match='requires episode'):
        validate(model,[bad],torch.device('cpu'),args)


def test_cli_teacher_requires_artifacts():
    with pytest.raises(SystemExit):parse_args(['--output','unused','--task-context-source','teacher'])
    args=parse_args(['--output','unused','--variants','context_history','--task-context-source','teacher',
        '--task-context-teacher-targets','labels','--task-context-checkpoint','observer.pt',
        '--task-context-split-manifest','split.json'])
    assert args.task_context_source=='teacher'


@pytest.mark.parametrize('head', ['diffusion','flow_matching'])
def test_self_contained_checkpoint_and_live_reference_stream(monkeypatch,bundle,tmp_path,head):
    from data.task_context_policy_dataset import context_policy_options
    from eval.eval_intention import load_intention_model
    from tests.test_task_context_policy import inputs
    model=make_model(monkeypatch,bundle,head_type=head).eval()
    model.task_context_source='teacher'
    config=dict(state_dim=8,mamba_output_dim=16,num_cameras=1,compressed_dim=4,
        action_dim=7,chunk_size=4,history_size=1,head_d_model=8,head_type=head,
        pool_out_dim=1024,frozen_vision_omitted=True,**context_policy_options(model))
    path=tmp_path/'policy.pt'
    torch.save(dict(config=config,model_state_dict={k:v for k,v in model.state_dict().items()
        if not k.startswith('vision_encoder.')}),path)
    bundle.unlink()
    loaded,_=load_intention_model(str(path),torch.device('cpu'))
    assert loaded.task_context_source=='teacher'
    loaded.eval()
    _,patches,cls,state,_,_=inputs()
    branch=loaded.task_context
    cache=branch.new_stream(2)
    reference=torch.randn(2,4,4)
    with pytest.raises(ValueError,match='batch differs'):
        branch.set_reference_task_tokens(cache,reference[:1])
    branch.set_reference_task_tokens(cache,reference)
    first,cache=branch.step(patches[:,0],cls[:,0],state[:,0],cache)
    second,cache=branch.step(patches[:,1],cls[:,1],state[:,1],cache)
    torch.testing.assert_close(first[:,:4],reference)
    torch.testing.assert_close(second[:,:4],reference)
    assert not torch.equal(first[:,-1],second[:,-1])
