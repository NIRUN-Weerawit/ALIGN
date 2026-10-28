"""Run immutable Piper best checkpoints through the requested evaluation matrix.

Context models: dynamic/freeze/handover at 40%; text models: autonomous at 0%.
The highest-score selector is fitted on training prefixes only. Existing reports
are never reused as new runs. Source, weights, commands, and configs are saved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/whinnoy/.venvs/align-ablation/bin/python'
LIBERO = '/home/whinnoy/.local/share/align/LIBERO'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def fit_decoder(out, models):
    import numpy as np
    import torch
    import h5py
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    from data.task_context_dataset import json_hash
    from eval.task_estimation import TaskLabelEstimator, context_fingerprint, SCHEMA
    from models.task_context_policy import TaskContextPolicyBranch
    from tools.probe_task_context_tokens import extract, QUERIES

    decoder_dir = out / 'decoder'
    decoder_dir.mkdir()
    cache = ROOT / 'reports/task_context_piper_384_20261023/prefix_cache'
    manifest = json.loads((cache / 'manifest.json').read_text())
    assert json_hash({k:v for k,v in manifest.items() if k != 'manifest_sha256'}) == manifest['manifest_sha256']
    first = next(m for m in models if not m['use_text'])
    checkpoint = torch.load(first['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    cfg = checkpoint['config']
    branch = TaskContextPolicyBranch(config=cfg['task_context_config']).requires_grad_(False).eval()
    branch.load_state_dict({k.removeprefix('task_context.'):v for k,v in checkpoint['model_state_dict'].items()
                            if k.startswith('task_context.')}, strict=True)
    fingerprint = context_fingerprint(branch)
    expected = cfg['task_context_split']['train_episodes']
    assert expected == manifest['split']['train']
    # Extract only training observations. No held-out features/labels are read.
    with threadpool_limits(limits=2):
        features, _ = extract(branch, cache, manifest, 'train', decoder_dir)
    assert features['keys'].tolist() == expected
    assert not set(expected) & set(cfg['val_episodes'])
    names = {}
    with h5py.File(ROOT/'data/libero_goal.h5', 'r') as data:
        for key, group in zip(features['keys'], features['groups']):
            text = json.loads(data[str(key)]['texts'][()])[0]
            if int(group) in names:
                assert names[int(group)] == text
            names[int(group)] = text
    assert sorted(names) == list(range(10))
    x = features['tokens'].reshape(-1, 512).astype(np.float64)
    labels = np.repeat(features['groups'], features['tokens'].shape[1])
    classifier = make_pipeline(StandardScaler(), LogisticRegression(
        C=.1, class_weight='balanced', max_iter=3000, random_state=42))
    with threadpool_limits(limits=2):
        classifier.fit(x, labels)
    scale, linear = classifier[0], classifier[1]
    metadata = dict(schema=SCHEMA, training_only=True, confirmation_opened=False,
        neural_weights_updated=False, context_fingerprint=fingerprint,
        fitting_episodes=expected, fitting_observations=len(x), fitting_queries=QUERIES,
        classes=linear.classes_.tolist(), probabilities_calibrated=False,
        fitting='Training prefixes only; StandardScaler + balanced LogisticRegression',
        C=.1, random_state=42, max_iter=3000,
        prefix_manifest_sha256=manifest['manifest_sha256'],
        train_features_sha256=digest(decoder_dir/'train_features.npz'),
        source_sha256=digest(__file__))
    path = decoder_dir/'task_estimator.npz'
    np.savez_compressed(path, metadata=np.asarray(json.dumps(metadata)),
        task_names=np.asarray([names[i] for i in range(10)]),
        latent_mean=branch.observer.encoder.latent_mean.numpy(),
        latent_std=branch.observer.encoder.latent_std.numpy(),
        feature_mean=scale.mean_, feature_scale=scale.scale_, weight=linear.coef_, bias=linear.intercept_)
    runtime = TaskLabelEstimator(path)
    runtime.validate_branch(branch)
    native = features['tokens'][:16,3]*runtime.latent_std+runtime.latent_mean
    with threadpool_limits(limits=2):
        error = float(np.max(np.abs(runtime.probabilities(native) - classifier.predict_proba(features['tokens'][:16,3].reshape(16,-1)))))
    assert error < 1e-8, error
    # Every policy must expose exactly this frozen observer coordinate system.
    for model in models:
        if model['use_text']:
            continue
        checkpoint = torch.load(model['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
        cfg = checkpoint['config']
        candidate = TaskContextPolicyBranch(config=cfg['task_context_config']).requires_grad_(False).eval()
        candidate.load_state_dict({k.removeprefix('task_context.'):v for k,v in checkpoint['model_state_dict'].items()
                                  if k.startswith('task_context.')}, strict=True)
        runtime.validate_branch(candidate)
        assert cfg['train_episodes'] == expected
        del candidate, checkpoint
    save(decoder_dir/'metadata.json', dict(metadata=metadata, runtime_probability_max_error=error,
        estimator_sha256=digest(path), all_context_checkpoints_match=True))
    print('DECODER_READY', len(expected), 'training episodes;', len(x), 'training prefix queries', flush=True)
    return str(path)


def prepare(out):
    import torch
    torch.set_num_threads(2)
    paths = sorted((ROOT/'checkpoints').glob('piper_context_*/*/intention_best.pt'))
    if not paths:
        raise ValueError('No Piper best checkpoints')
    out.mkdir(parents=True, exist_ok=False)
    source = out/'source_snapshot'
    for folder in ('models','data','eval','tools','utils','training','scripts'):
        for p in (ROOT/folder).rglob('*.py'):
            target=source/p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p,target)
    save(out/'source_manifest.json', {str(p.relative_to(source)):digest(p) for p in sorted(source.rglob('*.py'))})
    models=[]
    episodes=None
    for path in paths:
        relative=path.relative_to(ROOT/'checkpoints')
        target=out/'checkpoints'/relative
        target.parent.mkdir(parents=True)
        before=digest(path)
        shutil.copy2(path,target)
        if digest(target) != before or digest(path) != before:
            raise RuntimeError('Checkpoint changed while snapshotting: '+str(path))
        ck=torch.load(target,map_location='cpu',weights_only=True,mmap=True)
        cfg=ck['config']
        save(target.parent/'config.json',cfg)
        ids=sorted(cfg['val_episodes'])
        if episodes is None:
            episodes=ids
        assert ids==episodes and len(ids)==44
        assert not set(ids)&set(cfg['train_episodes'])
        text=bool(cfg.get('use_text',False))
        assert text or cfg.get('task_context_config')
        models.append(dict(model=path.parent.parent.name,variant=path.parent.name,
            source_checkpoint=str(path),checkpoint=str(target),checkpoint_sha256=before,
            use_text=text,epoch=ck.get('epoch'),
            config={k:cfg.get(k) for k in ['use_text','use_task_text','use_intent_tokens','use_memory_bank',
                'compressed_dim','head_d_model','history_size','head_history_mode','chunk_size','lr','epochs',
                'task_context_trainable','task_context_checkpoint','selection_metric','memory_bank_len']},
            train_episodes=cfg['train_episodes'],val_episodes=cfg['val_episodes']))
        del ck
    save(out/'checkpoints.json',models)
    # Fit using the snapshotted implementation too.
    sys.path.insert(0,str(source))
    estimator=fit_decoder(out,models)
    wrapper=out/'run_one.py'
    wrapper.write_text('import os,runpy,sys,torch\n'
        'torch.set_num_threads(2)\n'
        'torch.cuda.set_per_process_memory_fraction(0.30)\n'
        f'sys.path.insert(0,{str(source)!r})\n'
        f'sys.argv=[{str(source/"eval/eval_libero_v4_trajectory.py")!r}]+sys.argv[1:]\n'
        f'runpy.run_path({str(source/"eval/eval_libero_v4_trajectory.py")!r},run_name="__main__")\n')
    jobs=[]
    # Text evaluations begin without human assistance and have no mode sweep.
    for model in sorted(models,key=lambda m:(not m['use_text'],m['model'],m['variant'])):
        for mode in (['text'] if model['use_text'] else ['dynamic','freeze','handover']):
            switch=0.0 if model['use_text'] else .4
            directory=out/'runs'/model['model']/model['variant']/(mode+'_00' if model['use_text'] else mode+'_04')
            directory.mkdir(parents=True)
            cmd=[PYTHON,str(wrapper),'--data','data/libero_goal.h5','--checkpoint',model['checkpoint'],
                '--n-episodes','44','--switch-at',str(switch),'--max-steps','300','--seed','42',
                '--device','cuda','--out-dir',str(directory),'--no-video','--no-plot']
            if not model['use_text']:
                cmd+=['--task-context-mode',mode,'--task-estimator',estimator]
            job=dict(id=f"{model['model']}/{model['variant']}/{mode}",model=model['model'],variant=model['variant'],
                use_text=model['use_text'],mode=mode,switch_at=switch,out_dir=str(directory),command=cmd,
                checkpoint_sha256=model['checkpoint_sha256'],status='pending',attempts=0)
            save(directory/'command.json',cmd)
            jobs.append(job)
    save(out/'batch_manifest.json',dict(models=models,jobs=jobs,episodes=episodes,
        context_modes=['dynamic','freeze','handover'],context_switch=.4,text_switch=0.,
        estimator=estimator,source_snapshot=str(source),python=PYTHON,
        data_sha256=digest(ROOT/'data/libero_goal.h5'),created=time.time()))
    print('PREPARED',len(models),'models;',len(jobs),'jobs',flush=True)
    return jobs


def gpu_free():
    p=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return int(p.stdout.splitlines()[0])


def write_status(out,jobs):
    rows=[]
    for job in jobs:
        row={k:v for k,v in job.items() if k!='command'}
        progress=Path(job['out_dir'])/'episode_results.json'
        if progress.exists():
            try:
                e=json.loads(progress.read_text())['episodes']
                row.update(processed=len(e),successes=sum(bool(r.get('success')) for r in e))
            except (json.JSONDecodeError,KeyError):
                pass
        rows.append(row)
    save(out/'batch_status.json',dict(updated=time.time(),jobs=rows,
        completed=sum(r['status']=='complete' for r in rows),total=len(rows)))
    lines=['# Piper checkpoint evaluation batch','',
        'Context models: 40% expert prefix; dynamic / highest-score freeze / handover. '
        'Text models: autonomous from 0%, one run per checkpoint. '
        'All runs request 44 episodes, seed 42, 300 steps, action horizon 1, no video or plots.','',
        f"Completed {sum(r['status']=='complete' for r in rows)}/{len(rows)} runs.",'',
        '| Model | Variant | Mode | Expert prefix | Status | Completion / processed |',
        '|---|---|---|---:|---|---:|']
    for row in rows:
        n=row.get('processed',0);s=row.get('successes',0)
        result=f'{s}/{n} ({100*s/n:.1f}%)' if n else '—'
        lines.append(f"| {row['model']} | {row['variant']} | {row['mode']} | {100*row['switch_at']:.0f}% | {row['status']} | {result} |")
    lines+=['','Text and context runs have different assistance levels. Compare architectures within the same prefix setting. '
        'Checkpoint configs and immutable weight identities are saved in [checkpoints.json](checkpoints.json). '
        'Live progress is in [batch_status.json](batch_status.json).']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    return rows


def run(out,jobs,max_parallel):
    running={}
    env=dict(os.environ,MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',PYTHONPATH=LIBERO,PYTHONUNBUFFERED='1')
    while any(j['status'] in ('pending','running') for j in jobs):
        for pid,(p,job,log) in list(running.items()):
            code=p.poll()
            if code is None:
                continue
            log.close();del running[pid]
            job.update(returncode=code,ended=time.time())
            result=Path(job['out_dir'])/'result.json'
            valid=False
            if code==0 and result.exists():
                d=json.loads(result.read_text())
                valid=bool(d.get('evaluation_complete')) and len(d.get('episodes',[]))==44
            job['status']='complete' if valid else 'failed'
            print('FINISHED',job['id'],job['status'],'returncode',code,flush=True)
            if not valid:
                tail=(Path(job['out_dir'])/'run.log').read_text(errors='replace')[-5000:]
                print('FAILURE_LOG',tail,flush=True)
        pending=next((j for j in jobs if j['status']=='pending'),None)
        if pending is not None and len(running)<max_parallel:
            # Conservative admission, including large memory-bank checkpoints.
            size=Path(next(m['checkpoint'] for m in json.loads((out/'checkpoints.json').read_text())
                           if m['model']==pending['model'] and m['variant']==pending['variant'])).stat().st_size/(1024**2)
            required=max(2800,int(size+2200))
            if gpu_free()>=required:
                log=(Path(pending['out_dir'])/'run.log').open('w')
                p=subprocess.Popen(pending['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                pending.update(status='running',pid=p.pid,started=time.time(),attempts=pending['attempts']+1)
                running[p.pid]=(p,pending,log)
                print('STARTED',pending['id'],'pid',p.pid,'GPU free MiB',gpu_free(),flush=True)
                # Allow model initialization to reserve memory before admission.
                time.sleep(10)
        rows=write_status(out,jobs)
        print('PROGRESS',[(r['id'],r['status'],r.get('processed',0),r.get('successes',0)) for r in rows if r['status']=='running'],flush=True)
        time.sleep(20)
    write_status(out,jobs)
    save(out/'finished.json',dict(completed=sum(j['status']=='complete' for j in jobs),
        failed=[j['id'] for j in jobs if j['status']=='failed'],total=len(jobs),finished=time.time()))
    return 1 if any(j['status']=='failed' for j in jobs) else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-parallel',type=int,default=4)
    args=p.parse_args()
    if not 1<=args.max_parallel<=6:
        raise ValueError('Use one to six concurrent evaluations')
    out=args.output.resolve()
    jobs=prepare(out)
    sys.exit(run(out,jobs,args.max_parallel))


if __name__=='__main__':main()
