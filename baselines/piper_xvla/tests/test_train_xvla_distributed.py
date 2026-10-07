"""Exercise the trainer's synchronized update and exact validation reduction."""

import json
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from piper_xvla.train_xvla_piper import _mean_loss, _train_step, setup_distributed


class _Frames(Dataset):
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {"action": torch.tensor([float(index)]), "task": "move"}


class _Tokenizer:
    def __call__(self, tasks, **kwargs):
        return {"input_ids": torch.zeros((len(tasks), 1), dtype=torch.long)}


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        return ((self.weight - batch["action"][:, 0]) ** 2).mean(), {}


def _worker(rank, store_path, output_dir):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        policy = _Policy()
        wrapped = DistributedDataParallel(policy)
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        sampler = DistributedSampler(_Frames(4), num_replicas=2, rank=rank, shuffle=False)
        batch = next(iter(DataLoader(_Frames(4), batch_size=2, sampler=sampler)))
        _train_step(wrapped, batch, _Tokenizer(), "cpu", optimizer, scheduler, 10.0)

        validation = Subset(_Frames(5), range(rank, 5, 2))
        val_loss = _mean_loss(policy, DataLoader(validation, batch_size=2), _Tokenizer(), "cpu", distributed=True)
        (output_dir / f"rank-{rank}.json").write_text(
            json.dumps({"weight": policy.weight.item(), "val_loss": val_loss})
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_two_process_update_and_validation_cover_each_frame_once(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "store"), tmp_path), nprocs=2, join=True)
    results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]

    assert results[0]["weight"] == pytest.approx(results[1]["weight"])
    assert results[0]["weight"] == pytest.approx(0.3)
    expected_val_loss = sum((0.3 - index) ** 2 for index in range(5)) / 5
    assert all(result["val_loss"] == pytest.approx(expected_val_loss) for result in results)


def test_torchrun_worker_uses_its_local_cuda_device(monkeypatch):
    calls = []
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "init_process_group", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: calls.append(index))

    assert setup_distributed("cuda") == (1, 2, 1, "cuda:1")
    assert calls == [1, {"backend": "nccl", "init_method": "env://"}]
