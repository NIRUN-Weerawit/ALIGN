"""Run a frozen-policy, paired task-token counterfactual study on all 44 episodes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/home/whinnoy/.venvs/align-ablation/bin/python'
LIBERO='/home/whinnoy/.local/share/align/LIBERO'
ARMS=('normal','repeat','same_task','wrong_task','no_task')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)


def prepare(root,reference,seeds):
    import h5py
    import numpy as np
    import torch
    torch.set_num_threads(2)
    root.mkdir(parents=True,exist_ok=False)
    snapshot=root/'source_snapshot'
    shutil.copytree(reference/'source_snapshot',snapshot,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT/'eval/task_token_counterfactual.py',snapshot/'eval/task_token_counterfactual.py')
    sys.path.insert(0,str(snapshot))
    from eval.task_estimation import TaskLabelEstimator
    from models.task_context_policy import TaskContextPolicyBranch
    original=reference/'checkpoints/piper_context_1/context_no_memory/intention_best.pt'
    ckpath=root/'checkpoint.pt';shutil.copy2(original,ckpath)
    assert sha(original)==sha(ckpath)
    ck=torch.load(ckpath,map_location='cpu',weights_only=True,mmap=True);cfg=ck['config']
    save(root/'config.json',cfg)
    assert not cfg['use_memory_bank'] and not cfg['use_text'] and cfg['history_size']==1
    estimator_path=root/'task_estimator.npz';shutil.copy2(reference/'decoder/task_estimator.npz',estimator_path)
    estimator=TaskLabelEstimator(estimator_path)
    branch=TaskContextPolicyBranch(config=cfg['task_context_config']).requires_grad_(False).eval()
    branch.load_state_dict({k.removeprefix('task_context.'):v for k,v in ck['model_state_dict'].items()
                            if k.startswith('task_context.')},strict=True)
    estimator.validate_branch(branch)
    features_path=reference/'decoder/train_features.npz'
    assert sha(features_path)==estimator.metadata['train_features_sha256']
    with np.load(features_path,allow_pickle=False) as f:
        keys=f['keys'].copy();groups=f['groups'].copy()
        assert keys.tolist()==cfg['train_episodes']
        assert estimator.metadata['fitting_queries'][3]['fraction']==.4
        native=(f['tokens'][:,3]*estimator.latent_std+estimator.latent_mean).astype(np.float32)
    predicted=estimator.probabilities(native).argmax(-1)
    good=predicted==groups
    assert all(((groups==i)&good).any() for i in range(10))
    np.savez_compressed(root/'donor_bank.npz',keys=keys,groups=groups,native_tokens=native,
        correctly_decoded=good,task_names=np.asarray(estimator.task_names))
    episodes=sorted(cfg['val_episodes'])
    assert len(episodes)==44 and not set(episodes)&set(keys)
    names=estimator.task_names
    bowl=['put the bowl on the plate','put the bowl on the stove','put the bowl on top of the cabinet']
    bottle=['put the wine bottle on the rack','put the wine bottle on top of the cabinet']
    drawer=['open the middle drawer of the cabinet','open the top drawer and put the bowl inside']
    neighbors={task:[other for other in family if other!=task] for family in [bowl,bottle,drawer] for task in family}
    neighbors.update({'push the plate to the front of the stove':['put the bowl on the stove'],
        'put the cream cheese in the bowl':['put the bowl on the plate'],
        'turn on the stove':['push the plate to the front of the stove']})
    plan={}
    with h5py.File(ROOT/'data/libero_goal.h5','r') as data:
        actual={ep:json.loads(data[ep]['texts'][()])[0] for ep in episodes}
    for seed in seeds:
        plan[str(seed)]={}
        for ep in episodes:
            task=actual[ep];group=names.index(task)
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'{seed}/{ep}'.encode()).digest()[:8],'little'))
            wrong_task=str(rng.choice(neighbors[task]));wrong_group=names.index(wrong_task)
            same=int(rng.choice(np.flatnonzero((groups==group)&good)))
            wrong=int(rng.choice(np.flatnonzero((groups==wrong_group)&good)))
            def donor(i):return dict(index=i,episode=str(keys[i]),task=names[int(groups[i])],fraction=.4,
                correctly_decoded=True,norm=float(np.linalg.norm(native[i])))
            plan[str(seed)][ep]=dict(actual_task=task,same_task=donor(same),wrong_task=donor(wrong))
    save(root/'donor_plan.json',plan)
    save(root/'source_manifest.json',{str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*.py')})
    wrapper=root/'run_one.py'
    wrapper.write_text('import sys,torch\n'
        'torch.set_num_threads(2)\n'
        'torch.cuda.set_per_process_memory_fraction(.24)\n'
        'torch.use_deterministic_algorithms(True)\n'
        f'sys.path.insert(0,{str(snapshot)!r})\n'
        'from eval.task_token_counterfactual import run_cli\n'
        'run_cli(sys.argv[1],sys.argv[2:])\n')
    manifest=dict(checkpoint=str(ckpath),checkpoint_sha256=sha(ckpath),checkpoint_epoch=ck.get('epoch'),
        original_snapshot=str(original),config=cfg,episodes=episodes,seeds=seeds,arms=ARMS,
        plan=str(root/'donor_plan.json'),bank=str(root/'donor_bank.npz'),estimator=str(estimator_path),
        reference_batch=str(reference),donor_source_sha256=sha(features_path),
        donor_selection='Training-only native student vectors at 40%, correctly decoded; predefined related wrong-task goals; no policy outcome selection',
        conditions=dict(switch_at=.4,max_steps=300,action_horizon=1,stop_on_success=True,video=False,plots=False),
        matching='Explicit baseline initial states per seed; unchanged expert prefix; per-observation DDIM noise; full repeated normal controls',
        weights_updated=False,created=time.time(),source_sha256=sha(__file__))
    save(root/'protocol.json',manifest)
    return manifest


def make_jobs(root,manifest,smoke=False):
    episodes=manifest['episodes'][:1] if smoke else manifest['episodes']
    seeds=manifest['seeds'][:1] if smoke else manifest['seeds']
    jobs=[]
    for seed in seeds:
        normal_dir=root/('smoke' if smoke else 'runs')/f'seed_{seed}'/'normal'
        for arm in ARMS:
            directory=normal_dir.parent/arm;directory.mkdir(parents=True,exist_ok=True)
            job=dict(seed=seed,arm=arm,episodes=episodes,out_dir=str(directory),
                bank=manifest['bank'],plan=manifest['plan'],estimator=manifest['estimator'],
                checkpoint_sha256=manifest['checkpoint_sha256'],status='pending',
                normal_dir=None if arm=='normal' else str(normal_dir))
            args=['--data','data/libero_goal.h5','--checkpoint',manifest['checkpoint'],
                '--n-episodes',str(len(episodes)),'--switch-at','0.4','--task-context-mode','handover',
                '--max-steps','300','--seed',str(seed),'--device','cuda','--no-video','--no-plot',
                '--out-dir',str(directory)]
            if smoke:
                epfile=directory/'episodes.txt';epfile.write_text('\n'.join(episodes)+'\n');args+=['--val-episodes',str(epfile)]
            if arm!='normal':args+=['--initial-states',str(normal_dir/'initial_states.npz')]
            save(directory/'job.json',job)
            job['command']=[PYTHON,str(root/'run_one.py'),str(directory/'job.json')]+args
            save(directory/'command.json',job['command']);jobs.append(job)
    return jobs


def sidecar(job):
    import numpy as np
    directory=Path(job['out_dir']);states={}
    for ep in job['episodes']:
        with np.load(directory/(ep+'_rollout.npz'),allow_pickle=False) as d:states[ep]=d['initial_sim_state'].copy()
    np.savez_compressed(directory/'initial_states.npz',**states)


def run_jobs(root,jobs,parallel,smoke=False):
    running={};normal={(j['seed']):j for j in jobs if j['arm']=='normal'}
    env=dict(os.environ,MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONPATH=LIBERO,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    while any(j['status'] in ('pending','running') for j in jobs):
        for pid,(process,job,log) in list(running.items()):
            code=process.poll()
            if code is None:continue
            log.close();del running[pid]
            path=Path(job['out_dir'])/'result.json'
            valid=code==0 and path.exists() and json.loads(path.read_text())['evaluation_complete']
            job.update(status='complete' if valid else 'failed',returncode=code)
            if valid and job['arm']=='normal':sidecar(job)
            if valid:
                d=json.loads(path.read_text());d['evaluation_kind']='controlled task-token counterfactual'
                d['counterfactual_protocol']={k:v for k,v in job.items() if k!='command'}
                save(path,d)
                summary=Path(job['out_dir'])/'summary.md'
                summary.write_text(f"**Task-token counterfactual: {job['arm']}; seed {job['seed']}.** "
                    'Actual actor payloads, task-removal masks, and passive environment audits are in `counterfactual/`.\n\n'+summary.read_text())
            print('FINISHED',job['seed'],job['arm'],job['status'],flush=True)
            if not valid:print((Path(job['out_dir'])/'run.log').read_text(errors='replace')[-5000:],flush=True)
        if any(j['status']=='failed' for j in jobs):raise RuntimeError('A counterfactual evaluation failed; see its run.log')
        ready=next((j for j in jobs if j['status']=='pending' and (j['arm']=='normal' or normal[j['seed']]['status']=='complete')),None)
        if ready and len(running)<parallel:
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            if free>=2800:
                log=(Path(ready['out_dir'])/'run.log').open('w')
                process=subprocess.Popen(ready['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                ready.update(status='running',pid=process.pid,started=time.time())
                running[process.pid]=(process,ready,log)
                print('STARTED',ready['seed'],ready['arm'],process.pid,flush=True)
                time.sleep(8)
        for j in jobs:
            progress=Path(j['out_dir'])/'episode_results.json'
            if progress.exists():
                try:
                    rows=json.loads(progress.read_text())['episodes'];j.update(processed=len(rows),successes=sum(r['success'] for r in rows))
                except json.JSONDecodeError:pass
        save(root/('smoke_status.json' if smoke else 'status.json'),dict(jobs=jobs,updated=time.time()))
        print('PROGRESS',[(j['seed'],j['arm'],j.get('processed',0),j.get('successes',0)) for j in jobs if j['status']=='running'],flush=True)
        time.sleep(15)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-batch',type=Path,default=ROOT/'reports/piper_context_evaluations_20261026')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--max-parallel',type=int,default=4)
    p.add_argument('--resume-after-smoke-fix',action='store_true')
    a=p.parse_args();root=a.output.resolve()
    if a.resume_after_smoke_fix:
        if (root/'runs').exists():raise ValueError('Only resume before full evaluations start')
        manifest=json.loads((root/'protocol.json').read_text())
        if (root/'smoke').exists():
            archive=root/f'failed_smoke_{int(time.time())}'
            (root/'smoke').rename(archive)
            shutil.copy2(root/'source_snapshot/eval/task_token_counterfactual.py',archive/'intervention_source.py')
        shutil.copy2(ROOT/'eval/task_token_counterfactual.py',root/'source_snapshot/eval/task_token_counterfactual.py')
        save(root/'source_manifest.json',{str(p.relative_to(root/'source_snapshot')):sha(p) for p in (root/'source_snapshot').rglob('*.py')})
        manifest.update(runner_source_sha256=sha(__file__),intervention_source_sha256=sha(ROOT/'eval/task_token_counterfactual.py'))
        save(root/'protocol.json',manifest)
    else:
        manifest=prepare(root,a.reference_batch.resolve(),a.seeds)
    smoke=make_jobs(root,manifest,True);run_jobs(root,smoke,a.max_parallel,True)
    jobs=make_jobs(root,manifest);save(root/'jobs.json',jobs)
    run_jobs(root,jobs,a.max_parallel)
    save(root/'finished.json',dict(completed=len(jobs),failed=[],finished=time.time()))
    print('ALL_COUNTERFACTUALS_COMPLETE',flush=True)


if __name__=='__main__':main()
