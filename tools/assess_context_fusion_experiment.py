"""Combine adapter screens with matched extra DDIM draws; confirmation stays closed."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from data.task_context_dataset import file_hash
from tools.experiment_context_fusion import evaluate, make_head

FIELDS = ('action_mse', 'position_mse', 'first_position_mse', 'rotation_mse', 'gripper_error')
MODES = {'dynamic': ('dynamic', 'normal'), 'freeze40': ('freeze40', 'normal'),
         'freeze40_no_task': ('freeze40', 'no_task'), 'freeze40_task_swap': ('freeze40', 'task_swap')}


def _coordinates(rows):
    return [(r['episode'], r['anchor']) for r in rows]


def mean_rows(evaluations):
    """Average draws/seeds at matched anchors before resampling episodes."""
    rows = evaluations[0]['rows']
    if any(_coordinates(e['rows']) != _coordinates(rows) for e in evaluations):
        raise ValueError('Evaluation anchor order differs; diffusion draws cannot be paired')
    return [dict(episode=r['episode'], task=r['task'], anchor=r['anchor'],
        **{field: float(np.mean([e['rows'][i][field] for e in evaluations])) for field in FIELDS})
        for i, r in enumerate(rows)]


def paired(candidate, baseline):
    if _coordinates(candidate) != _coordinates(baseline):
        raise ValueError('Paired comparisons require identical episode/anchor order')
    episodes = sorted({r['episode'] for r in baseline})
    rng = np.random.default_rng(20261022)
    samples = rng.integers(len(episodes), size=(3000, len(episodes)))
    result = {}
    for field in FIELDS:
        values = np.asarray([np.mean([a[field] - b[field] for a, b in zip(candidate, baseline)
            if a['episode'] == ep]) for ep in episodes])
        result[field] = dict(mean=float(values.mean()),
            episode_bootstrap_95pct=np.quantile(values[samples].mean(1), [.025, .975]).tolist())
    return result


def aggregate(entries, noise_seeds):
    result = {}
    for mode in MODES:
        per_seed = [mean_rows([e['evaluations'][str(seed)][mode] for seed in noise_seeds]) for e in entries]
        rows = mean_rows([dict(rows=r) for r in per_seed])
        result[mode] = dict(rows=rows, mean={f: float(np.mean([r[f] for r in rows])) for f in FIELDS},
            training_seed_std={f: float(np.std([np.mean([r[f] for r in seed_rows]) for seed_rows in per_seed]))
                for f in FIELDS})
    return result


def render(result):
    lines = ['# Token fusion: multiple training seeds and diffusion draws', '',
        'Frozen expert-state continuation screen; action decoder and observer stay frozen. '
        'These errors do not measure task completion. Confirmation stays closed.', '',
        '| Budget / approach | Seeds | Dynamic pos. MSE | Frozen40 pos. MSE | First pos. MSE | Rotation MSE | Gripper error | No-task Δpos. | Swap Δpos. | Extra weights |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in [result['original']] + result['groups']:
        metrics = group['aggregate']; frozen = metrics['freeze40']['mean']
        value = lambda mode: metrics[mode]['mean']['position_mse']
        name = 'Original' if group['arm'] == 'original' else f"{group['steps']} / {group['arm']}"
        lines.append(f"| {name} | {len(group['train_seeds'])} | {value('dynamic'):.6f} | "
            f"{value('freeze40'):.6f} ± {metrics['freeze40']['training_seed_std']['position_mse']:.6f} | "
            f"{frozen['first_position_mse']:.6f} | {frozen['rotation_mse']:.6f} | {frozen['gripper_error']:.1%} | "
            f"{value('freeze40_no_task')-value('freeze40'):+.6f} | "
            f"{value('freeze40_task_swap')-value('freeze40'):+.6f} | {group['extra_parameters']:,} |")
    lines += ['', 'Frozen40 position MSE shows mean ± population standard deviation across training seeds after averaging diffusion draws. '
        'Positive removal/swap deltas mean the correct task context improves imitation relative to that diagnostic control. '
        'Removal and donor swaps can be outside the training distribution.', '',
        '| Budget / approach | Δfirst pos. vs original [95% interval] | Δfirst pos. vs additive_mixed [95% interval] | Δchunk pos. vs original [95% interval] | Δchunk pos. vs additive_mixed [95% interval] |',
        '|---|---:|---:|---:|---:|']
    def interval(value, field='position_mse'):
        v = value[field]; lo, hi = v['episode_bootstrap_95pct']
        return f"{v['mean']:+.6f} [{lo:+.6f}, {hi:+.6f}]"
    for group in result['groups']:
        control = group['comparisons'].get('vs_additive_mixed')
        lines.append(f"| {group['steps']} / {group['arm']} | {interval(group['comparisons']['vs_original'], 'first_position_mse')} | "
            f"{interval(control, 'first_position_mse') if control else 'Unavailable'} | "
            f"{interval(group['comparisons']['vs_original'])} | {interval(control) if control else 'Unavailable'} |")
    lines += ['', f"Diffusion seeds: {result['protocol']['noise_seeds']}. Every candidate uses the same batch order and draws. "
        'Bootstrap intervals resample development episodes after averaging anchors, training seeds, and diffusion draws; '
        'they are descriptive intervals conditional on these trained adapters and selected checkpoints. Training-seed variability is reported separately. '
        'Different optimizer-step budgets remain separate groups. First-action position error is especially relevant to action_horizon=1, '
        'which executes only the first action before replanning. No development labels train adapters.', '',
        '[Full metrics, paired controls, per-anchor rows, and provenance](result.json).']
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--noise-seeds', type=int, nargs='+', default=[20261022, 20261023, 20261024])
    p.add_argument('--device', default='cuda')
    p.add_argument('--gpu-memory-fraction', type=float, default=.20)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(set(args.noise_seeds)) != len(args.noise_seeds):
        raise ValueError('Distinct noise seeds required')
    started = time.monotonic(); torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.enabled = False
    packed = torch.load(args.features, map_location='cpu', weights_only=True)
    features_hash = file_hash(args.features)
    runs = [json.loads((path/'result.json').read_text()) for path in args.runs]
    reference = runs[0]['protocol']; checkpoint_path = Path(reference['checkpoint'])
    checkpoint_hash = file_hash(checkpoint_path)
    for run in runs:
        protocol = run['protocol']
        if not run.get('complete') or protocol.get('confirmation_opened'):
            raise ValueError('Only complete development-only experiments can be assessed')
        if (protocol['checkpoint_sha256'] != checkpoint_hash or protocol['features_sha256'] != features_hash
                or any(protocol[k] != reference[k] for k in ('batch_size', 'lr', 'fusion_dim', 'evaluation_noise_seed'))):
            raise ValueError('Runs must share checkpoint, cache, LR, batch order, and fusion width')
    if packed['metadata']['checkpoint_sha256'] != checkpoint_hash or packed['metadata']['confirmation_opened']:
        raise ValueError('Feature cache is not bound to the original checkpoint')
    train, dev = packed['train'], packed['development']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    protected = checkpoint['config']['task_context_split']
    if (set(train['episodes']) != set(protected['train_episodes']) or set(dev['episodes']) != set(protected['development_episodes'])
            or (set(train['episodes']) | set(dev['episodes'])) & set(protected['confirmation_episodes'])):
        raise ValueError('Feature split differs from protected training/development episodes')
    donor_rng = np.random.default_rng(20261022)
    donors = [int(donor_rng.choice([i for i, (name, f) in enumerate(zip(train['tasks'], train['fractions']))
        if name != task and f == fraction])) for task, fraction in zip(dev['tasks'], dev['fractions'])]
    swaps = train['frozen_task40'][donors]
    original_stored = {}
    entries = []; seen = set()
    for directory, run in zip(args.runs, runs):
        for mode, value in run['original'].items():
            if mode in original_stored and original_stored[mode]['rows'] != value['rows']:
                raise ValueError('Stored original-checkpoint draws differ across runs')
            original_stored[mode] = value
        expected = {(arm, seed) for arm in run['protocol']['arms'] for seed in run['protocol']['train_seeds']}
        if {(r['arm'], r['seed']) for r in run['outcomes']} != expected:
            raise ValueError('Completed run does not contain every declared arm/seed')
        for row in run['outcomes']:
            identity = (run['protocol']['steps'], row['arm'], row['seed'])
            if identity in seen:
                raise ValueError(f'Duplicate budget/arm/seed: {identity}')
            seen.add(identity)
            entries.append(dict(arm=row['arm'], fusion=row['fusion'], seed=row['seed'], steps=identity[0],
                checkpoint=row['checkpoint'], checkpoint_sha256=file_hash(row['checkpoint']),
                trainable_parameters=row['trainable_parameters'], run=str(directory.resolve()),
                stored_noise=run['protocol']['evaluation_noise_seed'], stored=row['evaluations']))
    base_weights = sum(v.numel() for k, v in checkpoint['model_state_dict'].items() if k.startswith('intention_head.'))
    original = dict(arm='original', fusion='additive', seed=None, steps=0, checkpoint=str(checkpoint_path),
        checkpoint_sha256=checkpoint_hash, stored_noise=reference['evaluation_noise_seed'], stored=original_stored)
    args.output.mkdir(parents=True)
    for completed, entry in enumerate([original] + entries, 1):
        saved = checkpoint if entry['arm'] == 'original' else torch.load(entry['checkpoint'], map_location='cpu', weights_only=True)
        if saved.get('config', {}).get('task_context_split') != protected:
            raise ValueError('Adapted checkpoint protected split changed')
        entry['extra_parameters'] = sum(v.numel() for k, v in saved['model_state_dict'].items()
            if k.startswith('intention_head.')) - base_weights
        head = None; entry['evaluations'] = {}
        for seed in args.noise_seeds:
            values = dict(entry['stored']) if seed == entry['stored_noise'] else {}
            for name, (mode, intervention) in MODES.items():
                if name not in values:
                    if head is None:
                        head = make_head(saved, entry['fusion'], reference['fusion_dim'], device)
                    values[name] = evaluate(head, dev, mode, device, seed, reference['batch_size'], intervention, swaps)
            entry['evaluations'][str(seed)] = {name: values[name] for name in MODES}
        entry.pop('stored'); entry.pop('stored_noise')
        del saved, head
        print(f"ASSESSED {entry['steps']} {entry['arm']} seed={entry['seed']} elapsed={time.monotonic()-started:.1f}s", flush=True)
        (args.output/'progress.json').write_text(json.dumps(dict(completed=completed, current=entry['checkpoint']))+'\n')
    original.update(aggregate=aggregate([original], args.noise_seeds), train_seeds=[], extra_parameters=0)
    groups = []
    for steps, arm in sorted({(e['steps'], e['arm']) for e in entries}):
        rows = [e for e in entries if (e['steps'], e['arm']) == (steps, arm)]
        groups.append(dict(steps=steps, arm=arm, train_seeds=sorted(e['seed'] for e in rows),
            trainable_parameters=rows[0]['trainable_parameters'], extra_parameters=rows[0]['extra_parameters'],
            aggregate=aggregate(rows, args.noise_seeds)))
    for group in groups:
        full = group['aggregate']['freeze40']['rows']
        group['comparisons'] = dict(vs_original=paired(full, original['aggregate']['freeze40']['rows']),
            no_task_vs_full=paired(group['aggregate']['freeze40_no_task']['rows'], full),
            task_swap_vs_full=paired(group['aggregate']['freeze40_task_swap']['rows'], full))
        control = next((g for g in groups if g['steps'] == group['steps'] and g['arm'] == 'additive_mixed'
            and g['train_seeds'] == group['train_seeds']), None)
        if control:
            group['comparisons']['vs_additive_mixed'] = paired(full, control['aggregate']['freeze40']['rows'])
        group['dynamic_comparisons'] = dict(vs_original=paired(group['aggregate']['dynamic']['rows'],
            original['aggregate']['dynamic']['rows']))
        if control:
            group['dynamic_comparisons']['vs_additive_mixed'] = paired(group['aggregate']['dynamic']['rows'],
                control['aggregate']['dynamic']['rows'])
    result = dict(protocol=dict(schema='align-context-fusion-multidraw-assessment-v1', noise_seeds=args.noise_seeds,
        checkpoint_sha256=checkpoint_hash, features_sha256=features_hash, confirmation_opened=False,
        donor_seed=20261022, donor_indices=donors, runtime_seconds=time.monotonic()-started,
        source_sha256={name: file_hash(ROOT/name) for name in ('tools/assess_context_fusion_experiment.py',
            'tools/experiment_context_fusion.py', 'models/task_context_policy.py')},
        input_runs=[dict(path=str(d.resolve()), result_sha256=file_hash(d/'result.json'), protocol=r['protocol'])
            for d, r in zip(args.runs, runs)]), original=original, groups=groups, entries=entries, complete=True)
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (args.output/'summary.md').write_text(render(result))
    archive = args.output/'sources'
    archive.mkdir()
    for source in result['protocol']['source_sha256']:
        destination = archive/source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/source, destination)


if __name__ == '__main__':
    main()
