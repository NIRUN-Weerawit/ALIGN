#!/usr/bin/env python3
"""Evaluate checkpoint candidates on shared LIBERO tasks/seeds and rank by success.

Use validation tasks for selection and independent episodes/seeds for final
reporting. Prediction loss does not select the deployed policy here.
"""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_intention_ablation import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,nargs='+',required=True)
    p.add_argument('--data',type=Path,default=ROOT/'data/libero_goal.h5')
    p.add_argument('--episodes',type=Path,required=True,help='Shared episode keys, one per task')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42])
    p.add_argument('--max-steps',type=int,default=300)
    p.add_argument('--switch-at',type=float,default=0.,help='0 means model controls from reset')
    p.add_argument('--interventions',nargs='+',choices=['normal','bypass','empty'],default=['normal'])
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    n=len([x for x in a.episodes.read_text().splitlines() if x.strip()])
    if n < 1 or a.max_steps < 1 or not a.seeds:
        p.error('At least one episode, seed and simulator step are required')
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    protocol=dict(checkpoint_sha256={str(x.resolve()):digest(x) for x in a.candidates},candidates=[str(x.resolve()) for x in a.candidates],data=str(a.data.resolve()),episodes=a.episodes.read_text().splitlines(),
                  seeds=a.seeds,max_steps=a.max_steps,switch_at=a.switch_at,interventions=a.interventions,
                  selection='normal-memory success rate; deterministic candidate order breaks ties; interventions do not select',
                  caveat='One trial/task/seed is provisional. Selected-on validation episodes cannot serve as an independent final test.')
    if (a.output/'protocol.json').exists() and json.loads((a.output/'protocol.json').read_text())!=protocol:
        raise ValueError('Existing evaluation protocol differs; use a new output directory')
    atomic_json(a.output/'protocol.json',protocol)
    result={}
    env=dict(os.environ,MUJOCO_GL='egl',LIBERO_CONFIG_PATH='/home/whinnoy/.local/share/align/libero-config')
    env['PYTHONPATH']='/home/whinnoy/.local/share/align/LIBERO'+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    for index,checkpoint in enumerate(a.candidates):
        key=f'{index:02d}_{checkpoint.parent.name}_{checkpoint.stem}'
        result[key]={}
        for intervention in a.interventions:
            episodes=[]
            for seed in a.seeds:
                out=a.output/key/intervention/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
                if not (a.resume and (out/'result.json').exists()):
                    cmd=[sys.executable,str(ROOT/'eval/eval_libero_v4_trajectory.py'),'--checkpoint',str(checkpoint.resolve()),
                         '--data',str(a.data.resolve()),'--val-episodes',str(a.episodes.resolve()),'--n-episodes',str(n),
                         '--out-dir',str(out.resolve()),'--seed',str(seed),'--max-steps',str(a.max_steps),
                         '--switch-at',str(a.switch_at),'--memory-intervention',intervention,'--noise-std','0','--no-video','--no-plot']
                    with (out/'eval.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
                records=json.loads((out/'result.json').read_text())['episodes']
                episodes.extend(dict(record,seed=seed) for record in records)
            result[key][intervention]=dict(success_rate=sum(bool(x['success']) for x in episodes)/len(episodes),episodes=episodes,
                                          checkpoint=str(checkpoint.resolve()))
            atomic_json(a.output/'summary.json',result)
            print(key,intervention,result[key][intervention]['success_rate'],flush=True)
    eligible=[(key,v['normal']) for key,v in result.items() if 'normal' in v]
    if eligible:
        key,winner=max(eligible,key=lambda x:x[1]['success_rate'])
        atomic_json(a.output/'selected_policy.json',dict(candidate=key,checkpoint=winner['checkpoint'],success_rate=winner['success_rate'],protocol=protocol))
    (a.output/'COMPLETE').write_text('Closed-loop candidate comparison complete.\n')

if __name__=='__main__':main()
