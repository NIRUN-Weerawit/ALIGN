"""Checkpoint I/O may overlap training but never observe mutating live state."""
import copy
import threading
import time

import pytest
import torch

from training.checkpoint_io import CheckpointWriter,cpu_snapshot,latest_resume_checkpoint


def payload(epoch):
    return dict(epoch=epoch,model_state_dict={'weight':torch.full((16,),float(epoch))},
                optimizer_state_dict={'state':{'step':torch.tensor(epoch)}})


def test_cpu_snapshot_freezes_tensor_aliases_and_nested_metadata():
    weight=torch.randn(3,requires_grad=True)
    source={'first':weight,'other':(weight,{'epoch':[1]})}
    saved=cpu_snapshot(source)
    assert saved['first'] is saved['other'][0] and not saved['first'].requires_grad
    expected=weight.detach().clone()
    with torch.no_grad():weight.add_(100)
    source['other'][1]['epoch'][0]=9
    torch.testing.assert_close(saved['first'],expected)
    assert saved['other'][1]['epoch']==[1]


def test_slow_mirror_does_not_block_next_local_save_and_coalesces_latest(tmp_path):
    output,staging=tmp_path/'output',tmp_path/'local'
    output.mkdir();torch.save(payload(0),output/'training_last.pt')
    writer=CheckpointWriter(output,staging)
    started,release=threading.Event(),threading.Event()
    epochs=[]
    original=writer._copy_atomic
    def slow(source,destination):
        epoch=torch.load(source,weights_only=True)['epoch'];epochs.append(epoch)
        if epoch==1:
            started.set();assert release.wait(10)
        original(source,destination)
    writer._copy_atomic=slow
    try:
        with writer:
            writer.save('training_last.pt',payload(1))
            assert started.wait(10)
            writer.save('training_last.pt',payload(2))
            writer.save('training_last.pt',payload(3))
            assert writer.status()['pending_files']==1
            assert torch.load(output/'training_last.pt',weights_only=True)['epoch']==0
            assert torch.load(writer.local/'training_last.pt',weights_only=True)['epoch']==3
            release.set()
    finally:release.set()
    assert epochs==[1,3]
    assert torch.load(output/'training_last.pt',weights_only=True)['epoch']==3
    assert not list(writer.local.glob('*.ready'))


def test_mirror_failure_retains_previous_remote_and_new_local_checkpoint(tmp_path):
    output=tmp_path/'output';output.mkdir();torch.save(payload(0),output/'training_last.pt')
    writer=CheckpointWriter(output,tmp_path/'stage')
    def fail(*args):raise OSError('simulated network failure')
    writer._copy_atomic=fail
    with pytest.raises(RuntimeError,match='local files retained'):
        with writer:writer.save('training_last.pt',payload(1))
    assert torch.load(output/'training_last.pt',weights_only=True)['epoch']==0
    assert torch.load(writer.local/'training_last.pt',weights_only=True)['epoch']==1


def test_resume_selects_saved_epoch_and_skips_incomplete_candidate(tmp_path):
    a,b=tmp_path/'a.pt',tmp_path/'b.pt'
    torch.save(payload(5),a);torch.save(payload(3),b)
    assert latest_resume_checkpoint([a,b])==a
    b.write_bytes(b'incomplete')
    assert latest_resume_checkpoint([a,b])==a
    with pytest.raises(RuntimeError,match='No complete resume checkpoint'):
        latest_resume_checkpoint([b])
    assert latest_resume_checkpoint([tmp_path/'missing.pt']) is None


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_synchronous_and_staged_checkpoints_preserve_optimizer_continuation(tmp_path,device):
    torch.manual_seed(1)
    model=torch.nn.Linear(4,2).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.01,fused=device=='cuda')
    x,target=torch.randn(3,4,device=device),torch.randn(3,2,device=device)
    def step(m,o):
        o.zero_grad();(m(x)-target).square().mean().backward();o.step()
    step(model,opt)
    snapshot=cpu_snapshot(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),epoch=1))
    for staging in [None,tmp_path/'stage']:
        output=tmp_path/('async' if staging else 'sync')
        with CheckpointWriter(output,staging) as writer:writer.save('training_last.pt',snapshot)
        saved=torch.load(output/'training_last.pt',map_location='cpu',weights_only=True,mmap=True)
        clone=torch.nn.Linear(4,2).to(device);clone.load_state_dict(saved['model_state_dict'])
        clone_opt=torch.optim.AdamW(clone.parameters(),lr=.01,fused=device=='cuda');clone_opt.load_state_dict(saved['optimizer_state_dict'])
        reference=copy.deepcopy(model);reference_opt=torch.optim.AdamW(reference.parameters(),lr=.01,fused=device=='cuda')
        reference_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        step(clone,clone_opt);step(reference,reference_opt)
        for a,b in zip(clone.parameters(),reference.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_staging_excludes_concurrent_writers_and_reclaims_interrupted_generations(tmp_path):
    first=CheckpointWriter(tmp_path/'output',tmp_path/'stage')
    second=CheckpointWriter(tmp_path/'output',tmp_path/'stage')
    torch.save(payload(1),first.local/'training_last.pt')
    (first.local/'.training_last.pt.abandoned.ready').write_bytes(b'orphan')
    with first:
        assert not list(first.local.glob('*.ready'))
        with pytest.raises(RuntimeError,match='Another checkpoint writer'):second.__enter__()
        assert torch.load(first.local/'training_last.pt',weights_only=True)['epoch']==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_gpu_weights_can_change_while_immutable_checkpoint_is_mirrored(tmp_path):
    live=torch.randn(1024,device='cuda')
    expected=live.cpu()
    writer=CheckpointWriter(tmp_path/'output',tmp_path/'stage')
    started,release=threading.Event(),threading.Event()
    original=writer._copy_atomic
    def slow(source,destination):
        started.set();assert release.wait(10);original(source,destination)
    writer._copy_atomic=slow
    try:
        with writer:
            writer.save('training_last.pt',cpu_snapshot(dict(epoch=1,weight=live)))
            assert started.wait(10)
            live.add_(10);torch.cuda.synchronize()
            release.set()
    finally:release.set()
    saved=torch.load(writer.output/'training_last.pt',weights_only=True)
    torch.testing.assert_close(saved['weight'],expected)


@pytest.mark.parametrize('staged', [False, True])
def test_independent_metric_winners_and_last_survive_resume(tmp_path, staged):
    from training.checkpoint_io import METRIC_CHECKPOINTS, save_metric_checkpoints, restore_metric_scores
    output = tmp_path / 'run'
    staging = tmp_path / 'stage' if staged else None
    rows = [dict(zip(METRIC_CHECKPOINTS, values)) for values in
            [(3., 2., 4.), (2., 3., 5.), (4., 1., 3.), (5., 4., 6.)]]
    scores = {}
    with CheckpointWriter(output, staging) as writer:
        for epoch, metrics in enumerate(rows, 1):
            checkpoint = payload(epoch)
            scores = save_metric_checkpoints(writer, checkpoint, metrics, scores)
            writer.save('intention_last.pt', checkpoint)
            writer.save('training_last.pt', dict(checkpoint, best_metric_scores=scores))
        assert restore_metric_scores(writer, dict(epoch=4)) == dict(zip(METRIC_CHECKPOINTS, [2., 1., 3.]))
    for metric, expected_epoch in zip(METRIC_CHECKPOINTS, [2, 3, 3]):
        saved = torch.load(output / METRIC_CHECKPOINTS[metric], weights_only=True)
        assert saved['epoch'] == expected_epoch and saved['selection_metric'] == metric
    assert torch.load(output / 'intention_last.pt', weights_only=True)['epoch'] == 4
    assert torch.load(output / 'training_last.pt', weights_only=True)['epoch'] == 4
    with CheckpointWriter(output, staging) as writer:
        restored = restore_metric_scores(writer, dict(epoch=4))
        assert restored == scores
        save_metric_checkpoints(writer, payload(5), rows[-1], restored)
    assert torch.load(output / 'intention_best_val_loss.pt', weights_only=True)['epoch'] == 2


def test_metric_recovery_ignores_missing_corrupt_and_future_winners(tmp_path):
    from training.checkpoint_io import METRIC_CHECKPOINTS, save_metric_checkpoints, restore_metric_scores
    writer = CheckpointWriter(tmp_path)
    assert restore_metric_scores(writer, dict(epoch=2)) == {}
    metrics = dict(zip(METRIC_CHECKPOINTS, [3., 2., 1.]))
    scores = save_metric_checkpoints(writer, payload(3), metrics, {})
    assert scores == metrics and restore_metric_scores(writer, dict(epoch=2)) == {}
    (tmp_path / 'intention_best_val_loss.pt').write_bytes(b'incomplete')
    assert 'val/loss' not in restore_metric_scores(writer, dict(epoch=3))
    assert save_metric_checkpoints(writer, payload(4), {'pos_mse': float('nan')}, {}) == {}
