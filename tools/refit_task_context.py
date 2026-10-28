"""Refit teacher/student on 384 episodes, optionally matching a policy split.

Uses fixed, predeclared budgets and final checkpoints. No development or
confirmation metrics select weights. Existing gated exports remain unchanged;
this export is explicitly a fixed-budget refit, not a new passing gate claim.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data.task_context_dataset import TaskContextEpisodes, file_hash, json_hash
from data.training_split import PIPER_SPLIT
from data.task_context_prefix_dataset import TaskContextPrefixes, build_prefix_cache
from training.train_task_context_teacher import load_checkpoint as load_teacher, make_loader
from training.train_task_context_teacher import to_device, evaluate as evaluate_teacher
from training.train_task_context_encoder import load_checkpoint as load_student
from training.train_task_context_encoder import evaluate as evaluate_student, make_loader as prefix_loader


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_policy_split(path, episode_keys, cameras=None):
    """Validate an explicit policy split before materializing any refit data."""
    path = Path(path).resolve()
    reference = json.loads(path.read_text())
    train, heldout = reference.get('train_episodes'), reference.get('val_episodes')
    if not isinstance(train, list) or not isinstance(heldout, list) or len(train) != 384 or len(heldout) != 44:
        raise ValueError('Policy split must provide 384 train_episodes and 44 val_episodes')
    if not all(isinstance(key, str) for key in train + heldout):
        raise ValueError('Policy split keys must be strings')
    if len(set(train)) != 384 or len(set(heldout)) != 44 or set(train) & set(heldout):
        raise ValueError('Policy split contains duplicate or overlapping episodes')
    if set(train + heldout) != set(episode_keys):
        raise ValueError('Policy split does not cover the exact source episode set')
    if cameras is not None and reference.get('cameras') != list(cameras):
        raise ValueError('Policy split camera order differs from source targets')
    return dict(train=list(train), development=[], confirmation=list(heldout)), dict(
        path=str(path), sha256=file_hash(path),
        rule='Exact policy train_episodes; policy val_episodes held out as confirmation')


def prepare_targets(source, output, split_manifest=PIPER_SPLIT):
    """Reuse immutable raw episode targets; fit normalization on declared train only."""
    source, output = Path(source).resolve(), Path(output)
    data = TaskContextEpisodes(source, 'train')
    original = data.manifest
    if {k: len(v) for k, v in original['split'].items()} != dict(train=340, development=44, confirmation=44):
        raise ValueError('Expected the original 340/44/44 episode split')
    if output.exists():
        raise FileExistsError(output)
    manifest = copy.deepcopy(original)
    if split_manifest:
        manifest['split'], split_reference = read_policy_split(split_manifest, original['episodes'], original.get('cameras'))
    else:
        manifest['split'] = dict(train=sorted(original['split']['train'] + original['split']['development']),
            development=[], confirmation=list(original['split']['confirmation']))
        split_reference = dict(rule='Original train + development; original confirmation keys unchanged')
    assert not set(manifest['split']['train']) & set(manifest['split']['confirmation'])
    output.mkdir(parents=True)
    for key in original['episodes']:
        for suffix in ('npy', 'npz'):
            (output / f'{key}.{suffix}').symlink_to(source / f'{key}.{suffix}')
    states, terminals, procedures = [], [], []
    for key in manifest['split']['train']:
        with np.load(source / f'{key}.npz', allow_pickle=False) as values:
            states.append(values['states'])
            terminals.append(values['terminal_delta'])
            procedures.append(values['procedure_delta'])
    states, terminals, procedures = np.concatenate(states), np.stack(terminals), np.stack(procedures)
    np.savez(output / 'normalization.npz', state_mean=states.mean(0),
        state_std=np.maximum(states.std(0), .05),
        visual_scale=np.maximum(terminals.std(axis=(0, 3)), .05)[..., None],
        terminal_mean=terminals.mean(0), procedure_mean=procedures.mean(0))
    manifest['normalization_sha256'] = file_hash(output / 'normalization.npz')
    manifest['refit'] = dict(parent_manifest=str(source / 'manifest.json'),
        parent_manifest_sha256=original['manifest_sha256'],
        **split_reference,
        checkpoint_selection='fixed_budget_final', normalization_episodes=384,
        heldout_previously_assessed=True)
    manifest.pop('manifest_sha256')
    manifest['manifest_sha256'] = json_hash(manifest)
    write_json(output / 'manifest.json', manifest)
    TaskContextEpisodes(output, 'train')
    return manifest


@torch.no_grad()
def export_latents(targets, teacher_path, output, device, protocol_path):
    """Materialize frozen labels without fitting anything to the held-out split."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    teacher, checkpoint = load_teacher(teacher_path, device)
    teacher.requires_grad_(False)
    if checkpoint['config'].get('checkpoint_selection') != 'fixed_budget_final':
        raise ValueError('Expected a fixed-budget teacher refit')
    output.mkdir(parents=True)
    files = {}
    for split in ('train', 'development', 'confirmation'):
        ds = TaskContextEpisodes(targets, split, anchors=1, verify=True)
        if checkpoint['config']['manifest_sha256'] != ds.manifest['manifest_sha256']:
            raise ValueError('Teacher/target manifest mismatch')
        keys, values = [], []
        for batch in make_loader(ds, 8):
            keys.extend(batch['keys'])
            batch = to_device(batch, device)
            tokens, _ = teacher.encode(batch['features'], batch['states'], batch['times'],
                batch['lengths'], batch['anchors'], 'full')
            values.append(tokens[:, 0].cpu().numpy())
        shape = [teacher.config['tokens'], teacher.config['token_dim']]
        array = np.concatenate(values) if values else np.empty((0, *shape), np.float32)
        path = output / f'{split}.npz'
        np.savez(path, keys=np.asarray(keys, dtype=str), task_tokens=array)
        files[split] = dict(episodes=len(keys), sha256=file_hash(path))
    write_json(output / 'manifest.json', dict(schema='align-privileged-task-tokens-v1',
        teacher_sha256=file_hash(teacher_path), target_manifest_sha256=ds.manifest['manifest_sha256'],
        teacher_epoch=checkpoint['epoch'], shape=shape, splits=files,
        refit_protocol_sha256=file_hash(protocol_path), checkpoint_selection='fixed_budget_final',
        contract='Frozen whole-episode labels; no held-out optimization or checkpoint selection. Not a confirmation-gated export.'))


def export_observer(teacher_path, student_path, targets, cache, output, protocol_path):
    teacher, tc = load_teacher(teacher_path)
    student, sc = load_student(student_path)
    source, prefixes = TaskContextEpisodes(targets, 'train'), TaskContextPrefixes(cache, 'train')
    if any(c['config'].get('checkpoint_selection') != 'fixed_budget_final' for c in (tc, sc)):
        raise ValueError('Expected fixed-budget final teacher and student')
    if sc['config']['teacher_sha256'] != file_hash(teacher_path) or student.config['history'] != 'full':
        raise ValueError('Student does not match the refitted teacher/full-history contract')
    if (tc['config']['manifest_sha256'] != source.manifest['manifest_sha256'] or
        sc['config']['manifest_sha256'] != prefixes.manifest['manifest_sha256'] or
        prefixes.manifest['target_manifest_sha256'] != source.manifest['manifest_sha256']):
        raise ValueError('Refit normalization/target/prefix binding mismatch')
    metadata = dict(teacher_sha256=file_hash(teacher_path), student_sha256=file_hash(student_path),
        target_manifest_sha256=source.manifest['manifest_sha256'],
        prefix_manifest_sha256=prefixes.manifest['manifest_sha256'],
        refit_protocol_sha256=file_hash(protocol_path), training_episodes=384, heldout_episodes=44,
        split_reference=source.manifest['refit'],
        checkpoint_selection='fixed_budget_final', cameras=source.manifest['cameras'], fps=source.manifest['fps'],
        limitations=source.manifest['limitations'] + [
            'Fixed-budget 384-episode refit; no new controlled information-gate claim.',
            'Held-out episodes have appeared in earlier experiments; this refit initializes from scratch.',
            'New latent space requires new downstream action-policy training and task-label estimators.'])
    bundle = dict(schema='align-causal-task-observer-v1',
        summary_config={k: teacher.config[k] for k in ('cameras', 'raw_dim', 'width')},
        encoder_config=student.config, summary=teacher.summary.state_dict(), encoder=student.state_dict(),
        state_mean=torch.from_numpy(source.norm['state_mean'].copy()),
        state_std=torch.from_numpy(source.norm['state_std'].copy()), metadata=metadata)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.save(bundle, output)
    return dict(path=str(output), sha256=file_hash(output), **metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-targets', default='reports/task_context_teacher_20261020/targets')
    parser.add_argument('--split-manifest', default=str(PIPER_SPLIT), help='Policy manifest specifying exact 384 train_episodes and 44 val_episodes (default: canonical Piper split)')
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--num-task-tokens','--task-context-tokens',type=int,default=4)
    parser.add_argument('--teacher-epochs', type=int, default=80)
    parser.add_argument('--student-epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    a = parser.parse_args()
    if min(a.teacher_epochs, a.student_epochs, a.num_task_tokens) < 1:
        parser.error('Epoch budgets and task-token count must be positive')
    report, checkpoints = Path(a.report_dir).resolve(), Path(a.checkpoint_dir).resolve()
    source_manifest = json.loads((Path(a.source_targets) / 'manifest.json').read_text())
    split_reference = None
    if a.split_manifest:
        _, split_reference = read_policy_split(a.split_manifest, source_manifest['episodes'], source_manifest['cameras'])
    if report.exists() or checkpoints.exists():
        raise FileExistsError('Use new report and checkpoint directories')
    report.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    torch.set_num_threads(2)
    device = torch.device(a.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.24)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.enable_cudnn_sdp(False)
    start = time.perf_counter()
    sources = ['models/vision_config.py','tools/refit_task_context.py', 'training/train_task_context_teacher.py',
        'data/training_split.py', 'configs/splits/libero_goal_piper_384_44.json',
        'training/train_task_context_encoder.py', 'data/task_context_dataset.py',
        'data/task_context_prefix_dataset.py', 'models/task_context_teacher.py',
        'models/task_context_encoder.py', 'models/task_context_observer.py']
    snapshot = report / 'source_snapshot'
    snapshot.mkdir()
    for name in sources:
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    protocol_path = report / 'protocol.json'
    if split_reference:
        shutil.copyfile(a.split_manifest, report / 'policy_split_reference.json')
    write_json(protocol_path, dict(schema='align-task-context-fixed-refit-v1', seed=a.seed,
        teacher_epochs=a.teacher_epochs, student_epochs=a.student_epochs, num_task_tokens=a.num_task_tokens,
        train_episodes=384, development_episodes=0, heldout_episodes=44,
        teacher_updates=a.teacher_epochs * 48, student_updates=a.student_epochs * 48,
        checkpoint_selection='fixed_budget_final', initialize='from_scratch',
        split_reference=split_reference,
        heldout_rule=('Exact policy val_episodes' if split_reference else 'Original confirmation keys')
            + '; excluded from normalization fitting, optimization, and selection.',
        assessment='After both final checkpoints are fixed; no metric-driven retries.',
        source_hashes={name: file_hash(ROOT / name) for name in sources}))
    def status(stage, **extra):
        write_json(report / 'status.json', dict(stage=stage, elapsed_s=time.perf_counter()-start, **extra))
        print(json.dumps(dict(stage=stage, **extra)), flush=True)
    def run(name, arguments):
        status(name)
        with (report / f'{name}.log').open('w') as log:
            subprocess.run([sys.executable, *arguments], cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT, check=True)
    try:
        status('prepare_targets')
        targets = report / 'targets'
        manifest = prepare_targets(a.source_targets, targets, a.split_manifest)
        write_json(report / 'split.json', dict(split=manifest['split'],
            manifest_sha256=manifest['manifest_sha256'], counts={k:len(v) for k,v in manifest['split'].items()}))
        teacher_run, student_run = checkpoints / 'teacher', checkpoints / 'student'
        run('train_teacher', ['training/train_task_context_teacher.py', '--targets', str(targets),
            '--output', str(teacher_run), '--epochs', str(a.teacher_epochs), '--seed', str(a.seed),
            '--num-task-tokens', str(a.num_task_tokens), '--mode', 'full', '--pooling', 'quarters', '--decoder-style', 'temporal',
            '--final-refit', '--device', a.device])
        teacher = checkpoints / 'teacher_frozen.pt'
        shutil.copyfile(teacher_run / 'teacher_final.pt', teacher)
        status('export_teacher_labels')
        latents = report / 'frozen_latents'
        export_latents(targets, teacher, latents, device, protocol_path)
        status('build_prefix_cache')
        cache = report / 'prefix_cache'
        build_prefix_cache(targets, latents, teacher, cache, a.device)
        run('train_student', ['training/train_task_context_encoder.py', '--cache', str(cache),
            '--teacher', str(teacher), '--output', str(student_run), '--epochs', str(a.student_epochs),
            '--seed', str(a.seed), '--history', 'full', '--final-refit', '--device', a.device])
        student = checkpoints / 'student_frozen.pt'
        shutil.copyfile(student_run / 'student_final.pt', student)
        status('export_observer')
        observer = checkpoints / 'observer_frozen.pt'
        exported = export_observer(teacher, student, targets, cache, observer, protocol_path)
        write_json(report / 'observer_export.json', exported)
        status('heldout_assessment')
        teacher_model, _ = load_teacher(teacher, device)
        student_model, _ = load_student(student, device)
        teacher_ds = TaskContextEpisodes(targets, 'confirmation', anchors=3, verify=True)
        student_ds = TaskContextPrefixes(cache, 'confirmation', verify=True)
        teacher_metrics, teacher_rows = evaluate_teacher(teacher_model, make_loader(teacher_ds, 8), device, 'full')
        student_metrics, student_rows = evaluate_student(student_model, teacher_model, prefix_loader(student_ds), device)
        write_json(report / 'heldout.json', dict(split='confirmation', episodes=44,
            checkpoints=dict(teacher=file_hash(teacher), student=file_hash(student)),
            used_for_selection=False, teacher=teacher_metrics, student=student_metrics,
            teacher_rows=teacher_rows, student_rows=student_rows))
        del teacher_model, student_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        run('verify_observer', ['tools/verify_task_context_observer.py', '--bundle', str(observer),
            '--student', str(student), '--targets', str(targets), '--cache', str(cache),
            '--output', str(report / 'observer_verification.json'), '--device', a.device])
        status('complete', teacher=str(teacher), student=str(student), observer=str(observer))
    except Exception as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
