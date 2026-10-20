"""Causality, exact state contracts, privileged-label isolation, and mixture learning."""
import copy
import json

import numpy as np
import pytest
import torch

from data.task_context_dataset import json_hash, file_hash
from data.task_context_prefix_dataset import TaskContextPrefixes, collate_prefixes, SCHEMA
from models.task_context_encoder import (TaskContextEncoder, belief_loss, execution_loss,
    mixture_log_prob, selector_loss)


def toy(history="full", device="cpu"):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = TaskContextEncoder(width=16, tokens=4, token_dim=4, modes=4, layers=2,
        d_state=4, d_conv=3, expand=2, visual_width=8, horizons=(2, 5), history=history, window=3).to(device)
    x = torch.randn(2, 9, 16, device=device)
    lengths = torch.tensor([9, 7], device=device)
    anchors = torch.tensor([[0, 2, 5], [0, 1, 5]], device=device)
    return model, x, lengths, anchors


@pytest.mark.parametrize("history", ["full", "current", "short"])
def test_prefix_mutation_padding_and_streaming_parity(history):
    model, x, lengths, anchors = toy(history)
    expected = model(x, lengths, anchors)
    changed = x.clone()
    changed[:, 6:] = float("nan")
    # Even unmasked future NaNs cannot influence earlier Mamba readouts.
    actual = model(changed, lengths, anchors)
    for name in ("tokens", "logits", "execution_token"):
        torch.testing.assert_close(expected[name], actual[name])
    state = model.new_stream(batch_size=2)
    for t in range(6):
        streamed = model.step(x[:, t], state)
        batched = model(x[:, :t + 1], torch.full_like(lengths, t + 1), torch.full((2, 1), t))
        for name in ("tokens", "logits", "execution_token"):
            torch.testing.assert_close(streamed[name], batched[name][:, 0], atol=2e-6, rtol=2e-5)
    assert state.steps == 6
    fresh = model.new_stream(batch_size=2)
    torch.testing.assert_close(model.step(x[:, 0], fresh)["tokens"], expected["tokens"][:, 0])


@pytest.mark.parametrize("history,cut", [("current", 5), ("short", 3)])
def test_control_cannot_read_older_observations(history, cut):
    model, x, lengths, _ = toy(history)
    anchors = torch.full((2, 1), 5)
    before = model(x, lengths, anchors)["tokens"]
    x[:, :cut] = float("nan")
    torch.testing.assert_close(model(x, lengths, anchors)["tokens"], before)


def test_real_gradients_execution_mask_and_checkpoint(tmp_path):
    model, x, lengths, anchors = toy()
    batch = dict(anchors=anchors, lengths=lengths, cls=torch.randn(2, 9, 8), states=torch.randn(2, 9, 10))
    out = model(x, lengths, anchors)
    loss = belief_loss(model, out, torch.randn(2, 4, 4))
    future, metrics = execution_loss(model, out, batch)
    assert metrics["pairs_h2"] == 5 and metrics["pairs_h5"] == 4
    (loss + .2 * future).backward()
    assert torch.isfinite(loss + future)
    assert model.blocks[0].in_proj.weight.grad.abs().sum() > 0
    assert model.execution[0].weight.grad.abs().sum() > 0
    assert model.means.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.optim.AdamW(model.parameters(), lr=.001).step()
    path = tmp_path / "checkpoint.pt"
    torch.save(dict(model=model.state_dict(), model_config=model.config), path)
    from training.train_task_context_encoder import load_checkpoint
    cloned, _ = load_checkpoint(path)
    torch.testing.assert_close(model(x, lengths, anchors)["tokens"], cloned(x, lengths, anchors)["tokens"])
    # No valid future: zero finite loss with a gradient path, rather than terminal replication.
    batch["anchors"] = (lengths - 1)[:, None]
    out = model(x, lengths, batch["anchors"])
    future, metrics = execution_loss(model, out, batch)
    assert future == 0 and all(v == 0 for v in metrics.values())


def test_balanced_indistinguishable_prefix_can_retain_two_outcomes():
    # Identical inputs imply shared parameters; the two teacher labels differ.
    # Fit only a tiny readout distribution to isolate the probability objective.
    torch.manual_seed(5)
    means = torch.nn.Parameter(torch.tensor([-.5, .5]).reshape(1, 1, 2, 1, 1))
    log_stds = torch.nn.Parameter(torch.zeros_like(means))
    logits = torch.nn.Parameter(torch.zeros(1, 1, 2))
    target = torch.tensor([-2., 2.]).reshape(2, 1, 1, 1)
    opt = torch.optim.Adam([means, log_stds, logits], lr=.05)
    for _ in range(180):
        opt.zero_grad()
        out = dict(means=means.expand(2, -1, -1, -1, -1),
            log_stds=log_stds.clamp(-3, 1).expand(2, -1, -1, -1, -1), logits=logits.expand(2, -1, -1))
        loss = -torch.logsumexp(mixture_log_prob(out, target), -1).mean()
        loss.backward()
        opt.step()
    torch.testing.assert_close(means.detach().flatten(), torch.tensor([-2., 2.]), atol=.06, rtol=0)
    torch.testing.assert_close(logits.softmax(-1).detach(), torch.full_like(logits, .5), atol=.02, rtol=0)
    assert float(loss.detach()) < 0


def test_top_weight_selection_ignores_teacher_assignment():
    model, x, lengths, anchors = toy()
    with torch.no_grad():
        model.logits.weight.zero_()
        model.logits.bias.copy_(torch.tensor([-3., -3., 4., -3.]))
    out = model(x, lengths, anchors)
    assert (out["mode"] == 2).all()
    expected = out["means"][:, :, 2] * model.latent_std + model.latent_mean
    torch.testing.assert_close(out["tokens"], expected)


def test_selector_responsibility_is_detached_and_balanced_ambiguity_is_preserved():
    model, _, _, _ = toy()
    logits = torch.zeros(2, 1, 4, requires_grad=True)
    means = torch.zeros(2, 1, 4, 4, 4, requires_grad=True)
    with torch.no_grad():
        means[:, :, 0] = -2
        means[:, :, 1] = 2
        means[:, :, 2:] = 10
    target = torch.stack([-2 * torch.ones(4, 4), 2 * torch.ones(4, 4)])
    out = dict(logits=logits, means=means, log_stds=torch.zeros_like(means))
    loss = selector_loss(model, out, target)
    loss.backward()
    assert means.grad is None
    # Each indistinguishable input has an opposite label: equal aggregate
    # selector gradients on the two plausible modes, unlike collapsing to one.
    torch.testing.assert_close(logits.grad[:, :, 0].sum(), logits.grad[:, :, 1].sum())
    assert logits.grad[:, :, :2].sum() < 0


def test_input_duration_is_padding_only_and_unavailable_anchors_rejected():
    model, x, lengths, anchors = toy()
    out = model(x, lengths, anchors)
    shorter = torch.tensor([6, 6])
    torch.testing.assert_close(model(x[:, :6], shorter, anchors)["tokens"], out["tokens"])
    with pytest.raises(ValueError, match="Anchor"):
        model(x, lengths, lengths[:, None])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_fused_scan_streaming_and_cpu_equations_match(monkeypatch):
    # The production trainer enables TF32 at import. This equation-parity test
    # compares FP32 CPU/CUDA math independently of other tests' import order.
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    model, x, lengths, anchors = toy(device="cuda")
    model.eval()
    out = model(x, lengths, anchors)
    # Real CUDA fused scan vs online update kernels and independent CPU equations.
    state = model.new_stream(2)
    for t in range(6):
        streamed = model.step(x[:, t], state)
        expected = model(x[:, :t + 1], torch.full_like(lengths, t + 1), torch.full((2, 1), t, device="cuda"))
        torch.testing.assert_close(streamed["tokens"], expected["tokens"][:, 0], atol=2e-5, rtol=2e-4)
    cpu = copy.deepcopy(model).cpu()
    torch.testing.assert_close(cpu(x.cpu(), lengths.cpu(), anchors.cpu())["tokens"], out["tokens"].cpu(), atol=2e-5, rtol=2e-4)
    belief_loss(model, out, torch.randn(2, 4, 4, device="cuda")).backward()
    assert model.blocks[0].in_proj.weight.grad.abs().sum() > 0


def test_prefix_cache_split_hash_and_deterministic_sampling(tmp_path):
    keys = ["ep_000000", "ep_000001", "ep_000002"]
    files = {}
    for k in keys:
        n = 24
        np.savez(tmp_path / f"{k}.npz", summary=np.zeros((n, 16), dtype="f4"), states=np.zeros((n, 10), dtype="f4"),
            cls=np.zeros((n, 8), dtype="f4"), times=np.arange(n, dtype="f4") / 20,
            actions=np.zeros((n, 7), dtype="f4"), terminal=np.zeros((1, 5, 8), dtype="f4"),
            procedure=np.zeros((8, 1, 5, 8), dtype="f4"), procedure_states=np.zeros((8, 10), dtype="f4"),
            task_tokens=np.zeros((4, 4), dtype="f4"), group=0)
        files[k] = file_hash(tmp_path / f"{k}.npz")
    np.savez(tmp_path / "normalization.npz", latent_mean=np.zeros((4, 4)), latent_std=np.ones((4, 4)))
    m = dict(schema=SCHEMA, split=dict(train=keys[:1], development=keys[1:2], confirmation=keys[2:]), files=files,
        action_horizon=8, normalization_sha256=file_hash(tmp_path / "normalization.npz"))
    m["manifest_sha256"] = json_hash(m)
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    ds = TaskContextPrefixes(tmp_path, "train", verify=True)
    a = ds[0]["anchors"].copy()
    np.testing.assert_array_equal(a, ds[0]["anchors"])
    ds.epoch = 1
    assert not np.array_equal(a, ds[0]["anchors"])
    batch = collate_prefixes([ds[0]])
    assert batch["anchors"].shape == (1, 16) and batch["task_tokens"].shape == (1, 4, 4)
    with (tmp_path / f"{keys[0]}.npz").open("ab") as f:
        f.write(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        TaskContextPrefixes(tmp_path, "train", verify=True)


def test_auxiliary_decoder_keeps_teacher_frozen_but_trains_student():
    from models.task_context_teacher import TaskContextTeacher
    from training.train_task_context_encoder import assigned_tokens, outcome_loss
    model, x, lengths, anchors = toy()
    teacher = TaskContextTeacher(cameras=1, raw_dim=4, width=16, tokens=4, token_dim=4,
        layers=1, heads=4, horizon=4, procedure_points=3, diffusion_width=8).requires_grad_(False).eval()
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    batch = dict(summary=x, anchors=anchors, task_tokens=torch.randn(2, 4, 4),
        terminal=torch.randn(2, 1, 5, 4), procedure=torch.randn(2, 3, 1, 5, 4),
        procedure_states=torch.randn(2, 3, 10))
    out = model(x, lengths, anchors)
    loss = outcome_loss(teacher, assigned_tokens(model, out, batch), batch)
    loss.backward()
    assert model.means.weight.grad.abs().sum() > 0
    assert model.blocks[0].in_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())
    for k, v in teacher.state_dict().items():
        torch.testing.assert_close(v, before[k], atol=0, rtol=0)


def test_student_confirmation_requires_bound_passing_development(tmp_path):
    from tools.assess_task_context_encoder import validate_confirmation
    p = tmp_path / "student.pt"
    p.write_bytes(b"selected-student")
    paths = dict(full=p)
    report = dict(split="development", manifest_sha256="cache",
        gates=dict(causal_information=dict(passed=True)),
        checkpoints=dict(full=dict(sha256=file_hash(p))))
    validate_confirmation(report, paths, "cache")
    with pytest.raises(ValueError, match="manifest"):
        validate_confirmation(report, paths, "other-cache")
    report["gates"]["causal_information"]["passed"] = False
    with pytest.raises(ValueError, match="passing"):
        validate_confirmation(report, paths, "cache")
    report["gates"]["causal_information"]["passed"] = True
    p.write_bytes(b"changed-student")
    with pytest.raises(ValueError, match="checkpoint"):
        validate_confirmation(report, paths, "cache")


def test_portable_observer_normalization_reset_and_roundtrip(tmp_path):
    from models.task_context_observer import TaskContextObserver, load_observer
    encoder, _, _, _ = toy()
    cfg = dict(cameras=1, raw_dim=4, width=16)
    observer = TaskContextObserver(cfg, encoder.config).requires_grad_(False).eval()
    observer.state_mean.fill_(.5)
    observer.state_std.fill_(2.)
    visual = torch.randn(1, 1, 5, 4)
    state = torch.randn(1, 10)
    t = torch.zeros(1)
    with torch.no_grad():
        summary = observer.summary(visual[:, None], ((state - .5) / 2)[:, None], t[:, None])[:, 0]
    expected = observer.encoder.step(summary, observer.encoder.new_stream())["tokens"]
    stream = observer.new_stream()
    out = observer.observe(visual, state, t, stream)
    torch.testing.assert_close(expected, out["tokens"])
    snapshot = out["tokens"].detach().clone()
    observer.observe(visual + .1, state, t + .05, stream)
    torch.testing.assert_close(snapshot, out["tokens"])
    with pytest.raises(ValueError, match="increase"):
        observer.observe(visual, state, t, stream)
    with pytest.raises(ValueError, match="finite"):
        observer.observe(visual * float("nan"), state, t + .1, stream)
    # New episode starts with zero recurrence, independent of the previous stream.
    torch.testing.assert_close(observer.observe(visual, state, t, observer.new_stream())["tokens"], expected)
    path = tmp_path / "observer.pt"
    torch.save(dict(schema="align-causal-task-observer-v1", summary_config=cfg,
        encoder_config=encoder.config, summary=observer.summary.state_dict(),
        encoder=observer.encoder.state_dict(), state_mean=observer.state_mean,
        state_std=observer.state_std, metadata=dict(contract="test")), path)
    clone, metadata = load_observer(path)
    assert metadata["contract"] == "test" and all(not p.requires_grad for p in clone.parameters())
    torch.testing.assert_close(clone.observe(visual, state, t, clone.new_stream())["tokens"], expected)


def test_observer_export_rejects_development_only_report(tmp_path):
    from tools.export_task_context_observer import export_observer
    p = tmp_path / "development.json"
    p.write_text(json.dumps(dict(split="development", gates=dict(causal_information=dict(passed=True)))))
    with pytest.raises(ValueError, match="confirmation"):
        export_observer("missing", "missing", "missing", "missing", p, tmp_path / "out.pt")
    assert not (tmp_path / "out.pt").exists()
