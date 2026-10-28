"""Repeat the fixed normal/GT-donor protocol only for changed best checkpoints."""
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
from tools.run_correct_task_donor_batch import make_jobs,run_jobs
from tools.run_task_token_counterfactuals import PYTHON,sha,save


def prepare_updates(root,previous):
    import torch
    from eval.task_estimation import TaskLabelEstimator
    from models.task_context_policy import TaskContextPolicyBranch
    torch.set_num_threads(2)
    protocol=json.loads((previous/'protocol.json').read_text())
    assert json.loads((previous/'result.json').read_text())['complete']
    changed=[];checked=[]
    for old in protocol['models']:
        source=ROOT/'checkpoints'/old['model']/old['variant']/'intention_best.pt'
        digest=sha(source)
        checked.append(dict(model=old['model'],variant=old['variant'],path=str(source),
                            previous_sha256=old['checkpoint_sha256'],current_sha256=digest))
        if digest!=old['checkpoint_sha256']:changed.append((old,source,digest))
    if not changed:
        print('NO_UPDATED_BEST_CHECKPOINTS',flush=True)
        return None
    root.mkdir(parents=True,exist_ok=False)
    snapshot=root/'source_snapshot'
    shutil.copytree(previous/'source_snapshot',snapshot,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('donor_bank.npz','donor_plan.json','task_estimator.npz'):
        shutil.copy2(previous/name,root/name)
        assert sha(previous/name)==sha(root/name)
    shutil.copy2(__file__,root/'runner_source.py')
    shutil.copy2(ROOT/'tools/run_correct_task_donor_batch.py',root/'batch_runner_source.py')
    shutil.copy2(ROOT/'tools/summarize_correct_task_donor_batch.py',root/'analysis_source.py')
    estimator=TaskLabelEstimator(root/'task_estimator.npz')
    models=[]
    for old,source,digest in changed:
        target=root/'checkpoints'/old['model']/old['variant']/'intention_best.pt'
        target.parent.mkdir(parents=True)
        shutil.copy2(source,target)
        assert sha(target)==digest and sha(source)==digest, 'Source changed while taking snapshot'
        ck=torch.load(target,map_location='cpu',weights_only=True,mmap=True);cfg=ck['config']
        previous_cfg=json.loads((Path(old['checkpoint']).parent/'config.json').read_text())
        assert cfg['train_episodes']==previous_cfg['train_episodes']
        assert sorted(cfg['val_episodes'])==protocol['episodes']
        assert not cfg['use_text'] and not cfg.get('task_context_trainable',False)
        for key in ('compressed_dim','head_d_model','history_size','use_memory_bank','head_history_mode'):
            assert cfg[key]==previous_cfg[key],f'Architecture changed: {key}'
        branch=TaskContextPolicyBranch(config=cfg['task_context_config']).requires_grad_(False).eval()
        branch.load_state_dict({k.removeprefix('task_context.'):v for k,v in ck['model_state_dict'].items()
                                if k.startswith('task_context.')},strict=True)
        estimator.validate_branch(branch)
        save(target.parent/'config.json',cfg)
        models.append(dict(old,checkpoint=str(target),checkpoint_sha256=digest,epoch=ck['epoch'],
                           reused=False,previous_epoch=old['epoch'],previous_sha256=old['checkpoint_sha256']))
        print('UPDATED',old['model'],old['variant'],'epoch',old['epoch'],'→',ck['epoch'],flush=True)
    (root/'run_one.py').write_text('import sys,torch\n'
        'torch.set_num_threads(2)\n'
        'torch.cuda.set_per_process_memory_fraction(.24)\n'
        'torch.use_deterministic_algorithms(True)\n'
        f'sys.path.insert(0,{str(snapshot)!r})\n'
        'from eval.task_token_counterfactual import run_cli\n'
        'run_cli(sys.argv[1],sys.argv[2:])\n')
    save(root/'source_manifest.json',{str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*.py')})
    protocol.update(models=models,previous_batch=str(previous),bank=str(root/'donor_bank.npz'),
        plan=str(root/'donor_plan.json'),estimator=str(root/'task_estimator.npz'),
        scope='Changed best checkpoints from the eight-model Piper batch; same fixed evaluation scheme',
        reused_runs='None; unchanged checkpoints retain their prior evaluation',
        checked_current_checkpoints=checked,runner_sha256=sha(__file__),analysis_sha256=sha(root/'analysis_source.py'),
        created=time.time())
    save(root/'protocol.json',protocol)
    (root/'summary.md').write_text('# Updated checkpoint evaluation — running\n\n'
        'New normal-context and GT correct-task donor evaluations use the same 44 episodes, '
        'seeds 42/43/44, 40% handover, live execution context, 300-step cap, initial states and donor plan. '
        'The complete comparison with previous checkpoints will be generated automatically.\n\n'
        '[Protocol](protocol.json) · [Live status](status.json)\n')
    return protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous-batch',type=Path,default=ROOT/'reports/piper_correct_task_donor_all_20261027')
    parser.add_argument('--max-parallel',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    protocol=(json.loads((root/'protocol.json').read_text()) if args.resume else
              prepare_updates(root,args.previous_batch.resolve()))
    if protocol is None:return
    smoke=make_jobs(root,protocol,True)
    run_jobs(root,smoke,args.max_parallel,True,args.resume)
    jobs=make_jobs(root,protocol);save(root/'jobs.json',jobs)
    run_jobs(root,jobs,args.max_parallel,resume=args.resume)
    subprocess.run([PYTHON,str(root/'analysis_source.py'),'--root',str(root)],cwd=ROOT,
                   env=dict(os.environ,PYTHONPATH=str(ROOT)),check=True)
    save(root/'finished.json',dict(new_full_runs=len(jobs),reused_full_runs=0,failed=[],finished=time.time()))
    print('ALL_UPDATED_CORRECT_TASK_DONOR_EVALUATIONS_COMPLETE',flush=True)


if __name__=='__main__':main()
