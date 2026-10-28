"""Summarize named task estimates on failed expert-replay episodes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

from data.task_context_dataset import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reconstruction',type=Path,required=True)
    parser.add_argument('--handover',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output directory')
    start=json.loads(args.reconstruction.read_text())
    handover=json.loads(args.handover.read_text())
    if start['protocol']['estimator_sha256']!=handover['protocol']['estimator_sha256']:
        raise ValueError('The task estimators differ')
    experts={r['episode']:r for r in start['episodes'] if r['source']=='expert'}
    cold={r['episode']:r for r in start['episodes'] if r['source']=='policy'}
    prefix={r['episode']:r for r in handover['episodes']}
    if set(experts)!=set(cold) or set(experts)!=set(prefix):
        raise ValueError('Episode sets differ')
    cases=[]
    for key in experts:
        expert,autonomous,continued=experts[key],cold[key],prefix[key]
        if len({r['task_name'] for r in (expert,autonomous,continued)})!=1:
            raise ValueError('Task labels differ')
        if any(r['replay_check']['valid'] for r in (expert,autonomous,continued)):
            raise ValueError('Expected failed expert replay checks')
        row=dict(episode=key,actual_task=expert['task_name'],expert_task_success=expert['replay_check']['task_success'],
            prefix_policy_success=continued['success'])
        for name,source in [('expert',expert),('autonomous_start',autonomous),('prefix_policy',continued)]:
            row[name]=source['task_estimation']['checkpoints']
        cases.append(row)
    total=len(cases)
    selections={
        'expert_40pct':[r['expert']['prefix_40pct']['inferred']['correct'] for r in cases],
        'expert_end_reference':[r['expert']['last_reference']['inferred']['correct'] for r in cases],
        'expert_final':[r['expert']['final']['inferred']['correct'] for r in cases],
        'autonomous_start_final':[r['autonomous_start']['final']['inferred']['correct'] for r in cases],
        'prefix_policy_handover':[r['prefix_policy']['handover']['inferred']['correct'] for r in cases],
        'prefix_policy_end_reference':[r['prefix_policy']['last_reference']['inferred']['correct'] for r in cases],
        'prefix_policy_final':[r['prefix_policy']['final']['inferred']['correct'] for r in cases]}
    result=dict(schema='align-failed-expert-task-estimates-v1',
        inputs={str(path):file_hash(path) for path in (args.reconstruction,args.handover)},
        analysis_sha256=file_hash(__file__),episodes=total,
        actual_expert_task_failures=sum(not r['expert_task_success'] for r in cases),
        accuracy={k:dict(correct=sum(v),episodes=len(v),accuracy=float(np.mean(v))) for k,v in selections.items()},
        correct_at_handover_but_wrong_at_final=sum(r['prefix_policy']['handover']['inferred']['correct']
            and not r['prefix_policy']['final']['inferred']['correct'] for r in cases),
        failed_policy_despite_correct_handover=sum(not r['prefix_policy_success']
            and r['prefix_policy']['handover']['inferred']['correct'] for r in cases),
        cases=cases)
    args.output.mkdir(parents=True)
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    short={
        'put the bowl on the plate':'Bowl → plate',
        'put the wine bottle on the rack':'Wine → rack',
        'put the wine bottle on top of the cabinet':'Wine → cabinet',
        'open the top drawer and put the bowl inside':'Top drawer + bowl',
        'put the bowl on top of the cabinet':'Bowl → cabinet',
        'put the cream cheese in the bowl':'Cheese → bowl',
        'open the middle drawer of the cabinet':'Middle drawer',
        'turn on the stove':'Turn on stove',
        'put the bowl on the stove':'Bowl → stove',
        'push the plate to the front of the stove':'Push plate'}
    def label(prediction):
        return short.get(prediction['predicted_task'],prediction['predicted_task'])+(' ✓' if prediction['correct'] else ' ✗')
    lines=['| Episode | Recorded task | Estimate at 40% handover | Estimate at step 299 | Policy completes task |',
           '|---|---|---|---|---|']
    for r in cases:
        lines.append(f'| {r["episode"]} | {short.get(r["actual_task"],r["actual_task"])} | '
            f'{label(r["prefix_policy"]["handover"]["inferred"])} | '
            f'{label(r["prefix_policy"]["final"]["inferred"])} | '
            f'{"Yes" if r["prefix_policy_success"] else "No"} |')
    (args.output/'named_predictions.md').write_text('\n'.join(lines)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    fig,axes=plt.subplots(1,3,figsize=(18,8),sharey=True)
    sources=[('Expert replay',experts,args.reconstruction.parent,'expert'),
             ('Autonomous start: no human prefix',cold,args.reconstruction.parent,'policy'),
             ('40% human prefix → autonomous policy',prefix,args.handover.parent,'policy')]
    for ax,(title,rows,directory,suffix) in zip(axes,sources):
        correct=[]
        for i,key in enumerate(experts):
            observations=json.loads((directory/f'{key}_{suffix}_task_estimates.json').read_text())['observations']
            correct.append([r['inferred']['correct'] for r in observations])
            checkpoints=rows[key]['task_estimation']['checkpoints']
            ax.scatter(checkpoints['prefix_40pct']['step'],i,s=18,c='white',edgecolor='black',linewidth=.5,zorder=5)
            ax.scatter(checkpoints['last_reference']['step'],i,s=60,c='black',marker='|',linewidth=1,zorder=5)
        ax.imshow(np.asarray(correct,dtype=int),aspect='auto',interpolation='nearest',
                  cmap=ListedColormap(['#D55E00','#009E73']),vmin=0,vmax=1)
        ax.set(title=title,xlabel='Observation step (20 Hz)',xticks=[0,60,120,180,240,299])
    axes[0].set_yticks(range(total),[f'{r["episode"]}: {short.get(r["actual_task"],r["actual_task"])}' for r in cases],fontsize=9)
    fig.suptitle('Does the task-token label match the intended task on failed expert replay cases?',fontsize=14)
    fig.legend(handles=[Patch(color='#009E73',label='Correct task label'),Patch(color='#D55E00',label='Incorrect task label')],
               loc='lower center',bbox_to_anchor=(.57,.045),ncol=2)
    fig.text(.02,.015,'15 development episodes; one fixed training-only label probe. White dot = 40% demonstration time; '
             'black tick = last demonstration observation. Labels are diagnostic; policy success is separate.',fontsize=9)
    fig.tight_layout(rect=(0,.10,1,.95))
    for extension in ('png','svg'):fig.savefig(args.output/f'task_estimate_correctness.{extension}',dpi=170)
    plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','inputs')},indent=2))


if __name__=='__main__':main()
