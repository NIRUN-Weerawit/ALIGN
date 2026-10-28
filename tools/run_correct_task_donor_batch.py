"""Evaluate GT-selected native task donors across the eight non-text Piper policies.

Reuse the completed cp4/no-memory paired runs explicitly. Other policies receive
fresh normal and same-task runs with the original seeds, donors, and scene states.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_task_token_counterfactuals import PYTHON, LIBERO, sha, save


def prepare(root, reference, batch):
    import torch
    from eval.task_estimation import TaskLabelEstimator
    from models.task_context_policy import TaskContextPolicyBranch
    torch.set_num_threads(2)
    original=json.loads((reference/'protocol.json').read_text())
    candidates=[m for m in json.loads((batch/'checkpoints.json').read_text()) if not m['use_text']]
    assert len(candidates)==8
    root.mkdir(parents=True,exist_ok=False)
    snapshot=root/'source_snapshot'
    shutil.copytree(reference/'source_snapshot',snapshot,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT/'eval/task_token_counterfactual.py',snapshot/'eval/task_token_counterfactual.py')
    for name in ('donor_bank.npz','donor_plan.json','task_estimator.npz'):
        shutil.copy2(reference/name,root/name)
        assert sha(root/name)==sha(reference/name)
    shutil.copy2(__file__,root/'runner_source.py')
    shutil.copy2(ROOT/'tools/summarize_correct_task_donor_batch.py',root/'analysis_source.py')
    estimator=TaskLabelEstimator(root/'task_estimator.npz')
    models=[]
    for item in candidates:
        source=Path(item['checkpoint'])
        assert sha(source)==item['checkpoint_sha256']
        target=root/'checkpoints'/item['model']/item['variant']/'intention_best.pt'
        target.parent.mkdir(parents=True)
        shutil.copy2(source,target)
        assert sha(target)==item['checkpoint_sha256']
        ck=torch.load(target,map_location='cpu',weights_only=True,mmap=True)
        cfg=ck['config']
        assert sorted(cfg['val_episodes'])==original['episodes']
        assert cfg['train_episodes']==original['config']['train_episodes']
        assert not cfg['use_text'] and not cfg.get('task_context_trainable',False)
        branch=TaskContextPolicyBranch(config=cfg['task_context_config']).requires_grad_(False).eval()
        branch.load_state_dict({k.removeprefix('task_context.'):v for k,v in ck['model_state_dict'].items()
                                if k.startswith('task_context.')},strict=True)
        estimator.validate_branch(branch)
        save(target.parent/'config.json',cfg)
        spec={k:cfg[k] for k in ('use_memory_bank','history_size','compressed_dim')}
        models.append(dict(model=item['model'],variant=item['variant'],checkpoint=str(target),
            checkpoint_sha256=item['checkpoint_sha256'],epoch=ck.get('epoch'),policy_spec=spec,
            head_d_model=cfg['head_d_model'],head_history_mode=cfg.get('head_history_mode','mean'),
            reused=sha(target)==original['checkpoint_sha256']))
    assert sum(m['reused'] for m in models)==1
    wrapper=root/'run_one.py'
    wrapper.write_text('import sys,torch\n'
        'torch.set_num_threads(2)\n'
        'torch.cuda.set_per_process_memory_fraction(.24)\n'
        'torch.use_deterministic_algorithms(True)\n'
        f'sys.path.insert(0,{str(snapshot)!r})\n'
        'from eval.task_token_counterfactual import run_cli\n'
        'run_cli(sys.argv[1],sys.argv[2:])\n')
    save(root/'source_manifest.json',{str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*.py')})
    protocol=dict(models=models,episodes=original['episodes'],seeds=original['seeds'],arms=['normal','same_task'],
        bank=str(root/'donor_bank.npz'),plan=str(root/'donor_plan.json'),estimator=str(root/'task_estimator.npz'),
        reference_study=str(reference),reference_batch=str(batch),conditions=original['conditions'],
        donor_selection=original['donor_selection'],weights_updated=False,
        initial_states='Per-seed normal states from reference study, used by every model and arm',
        scope='Eight non-text Piper checkpoints; known tasks; policy-validation episodes; GT is privileged',
        reused_runs='cp4/context_no_memory: original normal and same_task runs, not new evaluations',
        runner_sha256=sha(__file__),analysis_sha256=sha(root/'analysis_source.py'),created=time.time())
    save(root/'protocol.json',protocol)
    return protocol


def make_jobs(root, protocol, smoke=False):
    jobs=[]
    reference=Path(protocol['reference_study'])
    for model in protocol['models']:
        if model['reused']:continue
        for seed in (protocol['seeds'][:1] if smoke else protocol['seeds']):
            for arm in protocol['arms']:
                episodes=['ep_000069'] if smoke else protocol['episodes']
                directory=root/('smoke' if smoke else 'runs')/model['model']/model['variant']/f'seed_{seed}'/arm
                directory.mkdir(parents=True,exist_ok=True)
                normal_dir=directory.parent/'normal'
                job=dict(model=model['model'],variant=model['variant'],seed=seed,arm=arm,episodes=episodes,
                    out_dir=str(directory),bank=protocol['bank'],plan=protocol['plan'],estimator=protocol['estimator'],
                    checkpoint_sha256=model['checkpoint_sha256'],policy_spec=model['policy_spec'],
                    normal_dir=None if arm=='normal' else str(normal_dir),status='pending')
                args=['--data','data/libero_goal.h5','--checkpoint',model['checkpoint'],
                    '--n-episodes',str(len(episodes)),'--switch-at','0.4','--task-context-mode','handover',
                    '--max-steps','300','--seed',str(seed),'--device','cuda','--no-video','--no-plot',
                    '--out-dir',str(directory),'--initial-states',str(reference/'runs'/f'seed_{seed}'/'normal'/'initial_states.npz')]
                if smoke:
                    epfile=directory/'episodes.txt';epfile.write_text('\n'.join(episodes)+'\n')
                    args+=['--val-episodes',str(epfile)]
                save(directory/'job.json',job)
                job['command']=[PYTHON,str(root/'run_one.py'),str(directory/'job.json')]+args
                save(directory/'command.json',job['command'])
                jobs.append(job)
    return jobs


def validate_result(job):
    path=Path(job['out_dir'])/'result.json'
    if not path.exists():return False
    data=json.loads(path.read_text())
    return (data.get('evaluation_complete',False) and
            data['evaluation_protocol']['checkpoint_sha256']==job['checkpoint_sha256'] and
            sorted(r['episode'] for r in data['episodes'])==sorted(job['episodes']) and
            all(r.get('task_token_intervention',{}).get('strategy')=='counterfactual_'+job['arm']
                for r in data['episodes']))


def run_jobs(root,jobs,parallel,smoke=False,resume=False):
    running={}
    env=dict(os.environ,MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONPATH=LIBERO,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    if resume:
        for j in jobs:
            if validate_result(j):j['status']='complete'
            elif any(Path(j['out_dir']).iterdir()):
                # Preserve partial artifacts; never mix retries into a partial run.
                directory=Path(j['out_dir'])
                if (directory/'run.log').exists():
                    archive=directory.parent/(directory.name+f'_partial_{int(time.time())}')
                    directory.rename(archive);directory.mkdir()
                    for name in ('job.json','command.json','episodes.txt'):
                        if (archive/name).exists():shutil.copy2(archive/name,directory/name)
    try:
        while any(j['status'] in ('pending','running') for j in jobs):
            for pid,(process,job,log) in list(running.items()):
                code=process.poll()
                if code is None:continue
                log.close();del running[pid]
                valid=code==0 and validate_result(job)
                job.update(status='complete' if valid else 'failed',returncode=code,finished=time.time())
                print('FINISHED',job['model'],job['variant'],job['seed'],job['arm'],job['status'],flush=True)
                if not valid:print((Path(job['out_dir'])/'run.log').read_text(errors='replace')[-5000:],flush=True)
            if any(j['status']=='failed' for j in jobs):raise RuntimeError('Evaluation failed; inspect run.log')
            ready=next((j for j in jobs if j['status']=='pending'),None)
            if ready and len(running)<parallel:
                free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
                if free>=2800:
                    log=(Path(ready['out_dir'])/'run.log').open('w')
                    process=subprocess.Popen(ready['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    ready.update(status='running',pid=process.pid,started=time.time())
                    running[process.pid]=(process,ready,log)
                    print('STARTED',ready['model'],ready['variant'],ready['seed'],ready['arm'],process.pid,flush=True)
                    time.sleep(5)
            for job in jobs:
                progress=Path(job['out_dir'])/'episode_results.json'
                if progress.exists():
                    try:
                        rows=json.loads(progress.read_text())['episodes']
                        job.update(processed=len(rows),successes=sum(r['success'] for r in rows))
                    except json.JSONDecodeError:pass
            save(root/('smoke_status.json' if smoke else 'status.json'),dict(jobs=jobs,updated=time.time()))
            time.sleep(10)
    finally:
        for process,job,log in running.values():
            process.terminate();process.wait();log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-study',type=Path,default=ROOT/'reports/piper_task_token_counterfactual_20261026')
    parser.add_argument('--reference-batch',type=Path,default=ROOT/'reports/piper_context_evaluations_20261026')
    parser.add_argument('--max-parallel',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    protocol=(json.loads((root/'protocol.json').read_text()) if args.resume else
              prepare(root,args.reference_study.resolve(),args.reference_batch.resolve()))
    smoke=make_jobs(root,protocol,True)
    run_jobs(root,smoke,args.max_parallel,True,args.resume)
    jobs=make_jobs(root,protocol);save(root/'jobs.json',jobs)
    run_jobs(root,jobs,args.max_parallel,resume=args.resume)
    subprocess.run([PYTHON,str(root/'analysis_source.py'),'--root',str(root)],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)),check=True)
    save(root/'finished.json',dict(new_full_runs=len(jobs),reused_full_runs=6,failed=[],finished=time.time()))
    print('ALL_CORRECT_TASK_DONOR_EVALUATIONS_COMPLETE',flush=True)


if __name__=='__main__':main()
