"""Train-only adapter experiment on frozen ALIGN expert-observation features.

This is a bounded continuation experiment, not a from-scratch architecture
comparison or a closed-loop success benchmark. Confirmation stays closed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from models.task_context_policy import ContextDiffusionPolicyHead
from data.task_context_dataset import file_hash

ARMS = {
    'additive_dynamic': ('additive', False),
    'additive_mixed': ('additive', True),
    'gated_mixed': ('gated_residual', True),
    'film_mixed': ('film', True),
    'attention_mixed': ('cross_attention', True),
    'execution_only': ('additive', True),
    'task_only_mixed': ('additive', True),
}


def make_head(checkpoint, fusion, width, device):
    cfg = checkpoint['config']
    state = {k.removeprefix('intention_head.'): v for k, v in checkpoint['model_state_dict'].items()
             if k.startswith('intention_head.')}
    if cfg.get('visual_context_mode', 'none') != 'none' or cfg.get('visual_token_attention', False):
        raise ValueError('Packed experiment currently requires the ordinary visual condition')
    cond = state['context_task.1.weight'].shape[0]
    context_dim = state['context_execution.0.weight'].numel()
    head = ContextDiffusionPolicyHead(cond_dim=cond,
        action_dim=cfg.get('action_dim', 7), hidden_dim=cfg['head_d_model'],
        num_inference_steps=cfg.get('diffusion_inference_steps', 10),
        time_dim=state['time_emb.1.weight'].shape[1], chunk_size=cfg['chunk_size'],
        num_train_timesteps=len(state['alpha_bar']) - 1, loss_repeats=1,
        clip_denoised=cfg.get('diffusion_clip_sample', False), history_size=cfg['history_size'],
        context_tokens=state['context_task.0.weight'].numel() // context_dim,
        context_dim=context_dim, context_fusion=fusion, context_fusion_dim=width)
    missing, unexpected = head.load_state_dict(state, strict=False)
    assert not unexpected and all(k.startswith('context_') for k in missing)
    head.requires_grad_(False)
    for name, value in head.named_parameters():
        if name.startswith('context_'):
            value.requires_grad_(True)
    return head.to(device)


def payload(data, index, mode, device):
    field = 'task' if mode == 'dynamic' else 'frozen_task40'
    return torch.cat((data[field][index], data['execution'][index, None]), -2).to(device)


def summarize(rows):
    result = {}
    for field in ('action_mse', 'position_mse', 'first_position_mse', 'rotation_mse', 'gripper_error'):
        result[field] = float(np.mean([r[field] for r in rows]))
    result['anchors'] = len(rows)
    return result


@torch.no_grad()
def evaluate(head, data, mode, device, seed, batch_size, intervention='normal', swaps=None):
    head.eval()
    rows = []
    # Every candidate receives exactly the same initial DDIM draws.
    torch.manual_seed(seed)
    for start in range(0, len(data['actions']), batch_size):
        idx = slice(start, start + batch_size)
        visual, states = data['visual'][idx].to(device), data['states'][idx].to(device)
        tokens = payload(data, idx, mode, device)
        if intervention == 'task_swap':
            if swaps is None:
                raise ValueError('Task swap requires training-only donor tokens')
            tokens[:, :-1] = swaps[idx].to(device)
        if intervention in ('normal', 'task_swap'):
            cond = head(visual, states, tokens)
        elif intervention == 'no_task':
            current, _, execution = head.conditioning_components(visual, states, tokens)
            cond = current + execution
        else:
            raise ValueError(intervention)
        prediction = head.sample(cond).float()
        target = data['actions'][idx].to(device)
        error = (prediction - target).square()
        metrics = dict(action_mse=error.mean((1, 2)), position_mse=error[:, :, :3].mean((1, 2)),
                       first_position_mse=error[:, 0, :3].mean(1), rotation_mse=error[:, :, 3:6].mean((1, 2)),
                       gripper_error=((prediction[:, :, 6] > .5) != (target[:, :, 6] > .5)).float().mean(1))
        for j in range(len(prediction)):
            row = dict(episode=data['episodes'][start+j], task=data['tasks'][start+j],
                       anchor=int(data['anchors'][start+j]))
            row.update({k: float(v[j]) for k, v in metrics.items()})
            rows.append(row)
    return dict(aggregate=summarize(rows), rows=rows)


def paired_interval(candidate, baseline, field='position_mse'):
    episodes = sorted({r['episode'] for r in baseline})
    assert [(r['episode'], r['anchor']) for r in candidate] == [(r['episode'], r['anchor']) for r in baseline]
    diff = {ep: [] for ep in episodes}
    for a, b in zip(candidate, baseline):
        diff[a['episode']].append(a[field] - b[field])
    values = np.array([np.mean(diff[ep]) for ep in episodes])
    rng = np.random.default_rng(20261022)
    bootstrap = rng.choice(values, (3000, len(values)), replace=True).mean(1)
    return dict(mean=float(values.mean()), episode_bootstrap_95pct=np.quantile(bootstrap, [.025, .975]).tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seeds', type=int, nargs='+', default=[123, 124, 125])
    p.add_argument('--arms', nargs='+', choices=list(ARMS), default=list(ARMS))
    p.add_argument('--fusion-dim', type=int, default=128)
    p.add_argument('--device', default='cuda')
    p.add_argument('--gpu-memory-fraction', type=float, default=.20)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Positive training limits and distinct seeds required')
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.enabled = False
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    packed = torch.load(args.features, map_location='cpu', weights_only=True)
    assert packed['metadata']['checkpoint_sha256'] == file_hash(args.checkpoint)
    train, dev = packed['train'], packed['development']
    split = checkpoint['config']['task_context_split']
    assert set(train['episodes']) == set(split['train_episodes'])
    assert set(dev['episodes']) == set(split['development_episodes'])
    assert not set(train['episodes']) & set(dev['episodes'])
    assert not (set(train['episodes']) | set(dev['episodes'])) & set(split['confirmation_episodes'])
    args.output.mkdir(parents=True)
    started = time.monotonic()
    protocol = dict(schema='align-context-fusion-adapter-experiment-v1',
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=file_hash(args.checkpoint),
        features=str(args.features.resolve()), features_sha256=file_hash(args.features),
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, train_seeds=args.seeds,
        arms=args.arms, fusion_dim=args.fusion_dim,
        train_episodes=len(set(train['episodes'])), development_episodes=len(set(dev['episodes'])),
        train_anchors=len(train['actions']), development_anchors=len(dev['actions']),
        confirmation_opened=False, action_decoder_frozen=True, observer_frozen=True,
        training_objective='Uniform-dimension diffusion epsilon MSE; fixed number of train-only optimizer steps',
        context_schedule='dynamic: current task; mixed: per-example 50% current, 50% task at 40% prefix; execution always current',
        evaluation='One matched DDIM draw per anchor, dynamic and task frozen at 40%; expert states, not simulator success',
        task_swap='Different-task donor from training partition at the matching fraction; labels used only to define this diagnostic control',
        evaluation_noise_seed=20261022, precision='FP32 features, conditioning, denoiser and optimizer',
        source_sha256={name: file_hash(ROOT/name) for name in ['tools/experiment_context_fusion.py', 'models/task_context_policy.py']})
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    donor_rng = np.random.default_rng(20261022)
    donors = []
    for task, fraction in zip(dev['tasks'], dev['fractions']):
        candidates = [i for i, (name, f) in enumerate(zip(train['tasks'], train['fractions']))
                      if name != task and f == fraction]
        donors.append(int(donor_rng.choice(candidates)))
    swapped_task = train['frozen_task40'][donors]
    baseline_head = make_head(checkpoint, 'additive', args.fusion_dim, device)
    baseline = {mode: evaluate(baseline_head, dev, mode, device, 20261022, args.batch_size)
                for mode in ('dynamic', 'freeze40')}
    baseline['freeze40_no_task'] = evaluate(baseline_head, dev, 'freeze40', device, 20261022, args.batch_size, 'no_task')
    baseline['freeze40_task_swap'] = evaluate(baseline_head, dev, 'freeze40', device, 20261022, args.batch_size, 'task_swap', swapped_task)
    del baseline_head
    outcomes = []
    for arm in args.arms:
        fusion, mixed = ARMS[arm]
        for seed in args.seeds:
            torch.manual_seed(seed)
            head = make_head(checkpoint, fusion, args.fusion_dim, device)
            if arm == 'execution_only':
                for name, value in head.named_parameters():
                    value.requires_grad_(name.startswith('context_execution'))
            elif arm == 'task_only_mixed':
                for name, value in head.named_parameters():
                    value.requires_grad_(name.startswith('context_task'))
            optimizer = torch.optim.AdamW([v for v in head.parameters() if v.requires_grad], lr=args.lr)
            trainable = sum(v.numel() for v in head.parameters() if v.requires_grad)
            # Separate sampler means extra module initialization cannot change training batches or noise.
            generator = torch.Generator().manual_seed(seed)
            losses = []
            head.train()
            for step in range(args.steps):
                idx = torch.randint(len(train['actions']), (args.batch_size,), generator=generator)
                tokens = payload(train, idx, 'dynamic', device)
                mask = torch.rand(args.batch_size, generator=generator) < .5
                if mixed:
                    tokens[mask, :-1] = train['frozen_task40'][idx[mask]].to(device)
                visual, states = train['visual'][idx].to(device), train['states'][idx].to(device)
                torch.manual_seed(seed * 100000 + step)
                cond = head(visual, states, tokens)
                loss = head.loss(train['actions'][idx].to(device), cond)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss: {arm}/{seed}/{step}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([v for v in head.parameters() if v.requires_grad], 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
                if (step + 1) % 50 == 0:
                    print(f'TRAIN {arm} seed={seed} step={step+1}/{args.steps} loss={np.mean(losses[-50:]):.6f}', flush=True)
            evaluations = {mode: evaluate(head, dev, mode, device, 20261022, args.batch_size)
                           for mode in ('dynamic', 'freeze40')}
            evaluations['freeze40_no_task'] = evaluate(head, dev, 'freeze40', device, 20261022, args.batch_size, 'no_task')
            evaluations['freeze40_task_swap'] = evaluate(head, dev, 'freeze40', device, 20261022, args.batch_size, 'task_swap', swapped_task)
            for mode in ('dynamic', 'freeze40'):
                evaluations[mode]['paired_vs_original'] = paired_interval(evaluations[mode]['rows'], baseline[mode]['rows'])
            cfg = copy.deepcopy(checkpoint['config'])
            cfg.update(task_context_fusion=fusion, task_context_fusion_dim=args.fusion_dim)
            weights = dict(checkpoint['model_state_dict'])
            weights.update({'intention_head.'+k: v.detach().cpu() for k, v in head.state_dict().items()})
            destination = args.output/f'{arm}_seed{seed}.pt'
            torch.save(dict(model_state_dict=weights, config=cfg, experiment=protocol,
                            adapter_arm=arm, adapter_seed=seed), destination)
            row = dict(arm=arm, fusion=fusion, seed=seed, trainable_parameters=trainable,
                       train_loss_first50=float(np.mean(losses[:50])), train_loss_last50=float(np.mean(losses[-50:])),
                       checkpoint=str(destination.resolve()), evaluations=evaluations)
            outcomes.append(row)
            result = dict(protocol=protocol, original=baseline, outcomes=outcomes,
                          elapsed_seconds=time.monotonic()-started, complete=False)
            (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
            print('EVALUATED', arm, seed, {m: e['aggregate']['position_mse'] for m, e in evaluations.items()}, flush=True)
            del optimizer, head
    for row in outcomes:
        control = next((r for r in outcomes if r['arm'] == 'additive_mixed' and r['seed'] == row['seed']), None)
        if control is not None:
            for mode in ('dynamic', 'freeze40'):
                row['evaluations'][mode]['paired_vs_additive_mixed'] = paired_interval(
                    row['evaluations'][mode]['rows'], control['evaluations'][mode]['rows'])
        for arm in ('freeze40_no_task', 'freeze40_task_swap'):
            row['evaluations'][arm]['paired_vs_full_context'] = paired_interval(
                row['evaluations'][arm]['rows'], row['evaluations']['freeze40']['rows'])
    result['complete'] = True
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    write_summary(result, args.output)


def write_summary(result, output):
    lines = ['# Token utilization adapter experiment', '',
        'All action-decoder, visual/state encoder, and observer weights are frozen. Only task/execution conditioning adapters and new fusion layers train. '
        'Training uses 340 expert episodes; evaluation uses 44 development episodes; confirmation remains closed. '
        'These are cached expert-state imitation results, not task-completion rates.', '',
        '| Approach | Dynamic position MSE | Frozen40 position MSE | Frozen40 without task | Frozen40 task swap | First-action position MSE, frozen40 |',
        '|---|---:|---:|---:|---:|---:|']
    def cells(values):
        return ' | '.join(f'{np.mean(v):.6f}' + (f' ± {np.std(v):.6f}' if len(v)>1 else '') for v in values)
    def metrics(e):
        return [e['dynamic']['aggregate']['position_mse'], e['freeze40']['aggregate']['position_mse'],
                e['freeze40_no_task']['aggregate']['position_mse'], e['freeze40_task_swap']['aggregate']['position_mse'],
                e['freeze40']['aggregate']['first_position_mse']]
    lines.append('| Original checkpoint | '+cells([[v] for v in metrics(result['original'])])+' |')
    for arm in result['protocol']['arms']:
        entries = [r for r in result['outcomes'] if r['arm'] == arm]
        values = np.array([metrics(r['evaluations']) for r in entries]).T
        lines.append('| '+arm+' | '+cells(values)+' |')
    lines += ['', 'Lower MSE is better. Values show mean ± population standard deviation over training seeds. '
              'Every run starts from the same pretrained checkpoint and receives matched train batches, diffusion noise, and development DDIM noise. '
              'New fusion layers begin with exactly the original additive behavior. Each adapter receives the same optimizer-step budget, rather than the same parameter count.', '',
              'The mixed context schedule uses current tokens or the same episode’s 40% snapshot, without task labels. It does not reproduce highest-score selection, simulator observations, or arbitrary handover times. '
              'The no-task control removes the entire task contribution, including its interaction layers; execution stays active.', '',
              'A positive no-task MSE change indicates useful task conditioning on these observations. Improvements require confirmation through full policy training and repeated simulator rollouts. '
              'Development results are exploratory and should not be reused as an untouched final test set.', '',
              '[Detailed per-episode errors and bootstrap intervals](result.json). [Exact protocol](protocol.json).']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
