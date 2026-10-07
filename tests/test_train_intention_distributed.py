"""Exercise intention training across two independent processes on CPU."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_two_process_training_keeps_models_in_sync(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(textwrap.dedent("""
        from types import SimpleNamespace

        import torch
        import torch.distributed as dist
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        from training.train_intention import (
            setup_distributed, sync_and_step, train_one_epoch,
            train_v4_batched_epoch, train_v4_epoch, validate,
        )

        args = SimpleNamespace(device="cpu", action_dim=7, head_type="transformer",
                               skip_nan=True, grad_clip=1.0, history_size=1,
                               chunk_size=2)
        device = setup_distributed(args)
        rank = dist.get_rank()

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(7, 4)
                self.head = torch.nn.Linear(4, 7)

            def forward(self, frames, state):
                features = self.encoder(state)
                return {"z_v_pooled_seq": features, "z_s_seq": features,
                        "h_seq": features}

            def predict_actions(self, vision, state, intent):
                return self.head(vision)

        torch.manual_seed(11)
        model = TinyModel()
        initial = [p.detach().clone() for p in model.parameters()]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        samples = [
            {"frames_window": torch.zeros(2, 1),
             "robot_state_window": torch.full((2, 7), float(i + 1)),
             "actions_window": torch.full((2, 7), float(i) / 10)}
            for i in range(8)
        ]
        sampler = DistributedSampler(samples, num_replicas=2, rank=rank,
                                     shuffle=False)
        loader = DataLoader(samples, batch_size=2, sampler=sampler)
        loss, _ = train_one_epoch(model, loader, optimizer, device, args)
        assert any(not torch.equal(p, before) for p, before in zip(model.parameters(), initial))

        # One rank has no local gradient; both must still take the same step.
        optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            model.head.weight.sum().backward()
        assert sync_and_step(model, optimizer, device, grad_clip=1.0)

        state = ([p.detach().clone() for p in model.parameters()], loss)
        gathered = [None, None]
        dist.all_gather_object(gathered, state)
        assert gathered[0][1] == gathered[1][1]
        for left, right in zip(gathered[0][0], gathered[1][0]):
            assert torch.allclose(left, right, atol=1e-7)

        class PatchEncoder(torch.nn.Module):
            def forward(self, patches, states):
                return patches[..., :1] * 0

        class ValidationModel(TinyModel):
            def __init__(self):
                super().__init__()
                self.state_encoder = self.encoder
                self.vision_patch_encoder = PatchEncoder()
                self.use_memory_bank = False
                self.use_history = False
                self._built = True

            def predict_actions(self, vision, state, intent):
                return self.head(state[:, -1:].expand(-1, 2, -1))

            def condition_actions(self, vision, state, intent=None, observed_mask=None):
                return vision, state, intent

        val_samples = [
            {"frames_segment": torch.zeros(3, 257, 768),
             "states_segment": torch.full((3, 7), float(i + 1)),
             "actions_segment": torch.full((3, 7), float(i) / 10),
             "segment_len": 3}
            for i in range(2)
        ]
        val_loader = DataLoader(val_samples[rank::2], batch_size=1)
        val_loss, _, val_metrics = validate(ValidationModel(), val_loader, device, args)
        gathered_val = [None, None]
        dist.all_gather_object(gathered_val, (val_loss, val_metrics))
        assert gathered_val[0] == gathered_val[1]
        assert val_metrics["gripper_genuine_batches"] == 4

        for train_epoch in (train_v4_epoch, train_v4_batched_epoch):
            torch.manual_seed(21)
            v4_model = ValidationModel()
            v4_optimizer = torch.optim.SGD(v4_model.parameters(), lr=0.01)
            v4_loss, _ = train_epoch(v4_model, val_loader, v4_optimizer, device, args)
            gathered_v4 = [None, None]
            dist.all_gather_object(
                gathered_v4,
                ([p.detach().clone() for p in v4_model.parameters()], v4_loss),
            )
            assert gathered_v4[0][1] == gathered_v4[1][1]
            for left, right in zip(gathered_v4[0][0], gathered_v4[1][0]):
                assert torch.allclose(left, right, atol=1e-7)
        dist.destroy_process_group()
    """))
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(project_root), env.get("PYTHONPATH")) if path
    )
    subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc-per-node=2", str(worker)],
        cwd=project_root, env=env, check=True, timeout=90,
    )
