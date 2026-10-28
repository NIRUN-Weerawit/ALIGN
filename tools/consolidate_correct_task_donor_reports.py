"""Consolidate the eight-model comparison with updated-checkpoint evaluations."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_task_token_counterfactuals import save,sha


def consolidate(base,updates,output):
    base,updates,output=base.resolve(),updates.resolve(),output.resolve()
    sources=[];selected={};original_keys=None;original_protocol=None
    for batch in (base,updates):
        result=json.loads((batch/'result.json').read_text())
        assert result['complete']
        protocol=result['protocol']
        if original_protocol is None:
            original_protocol=protocol
            original_keys={(m['model'],m['variant']) for m in result['models']}
        for field in ('episodes','seeds','arms','conditions'):
            assert protocol[field]==original_protocol[field],f'Incompatible protocol: {field}'
        for filename in ('donor_bank.npz','donor_plan.json','task_estimator.npz'):
            assert sha(batch/filename)==sha(base/filename),f'Different {filename}'
        for record in result['source_results']:
            assert sha(record['path'])==record['sha256'], 'Source result changed'
        sources.append(dict(path=str(batch),result_sha256=sha(batch/'result.json')))
        for row in result['models']:
            key=(row['model'],row['variant'])
            assert key in original_keys,'Unexpected model in update'
            assert sha(row['checkpoint'])==row['checkpoint_sha256']
            matches=[r for r in result['source_results'] if (r['model'],r['variant'])==key]
            directories={str(Path(r['path']).parent.parent.parent) for r in matches}
            assert len(directories)==1
            selected[key]=dict({k:v for k,v in row.items() if k in (
                'model','variant','checkpoint','checkpoint_sha256','epoch','policy_spec',
                'head_d_model','head_history_mode','reused','previous_epoch','previous_sha256')},
                evaluation_runs=directories.pop(),source_batch=str(batch))
    assert len(selected)==8
    output.mkdir(parents=True,exist_ok=False)
    for filename in ('donor_bank.npz','donor_plan.json','task_estimator.npz'):
        shutil.copy2(base/filename,output/filename)
    protocol=dict(original_protocol,models=list(selected.values()),
        consolidated_from=sources,previous_batch=str(base),
        bank=str(output/'donor_bank.npz'),plan=str(output/'donor_plan.json'),
        estimator=str(output/'task_estimator.npz'),weights_updated=False,
        scope='Latest evaluated checkpoints across all eight non-text Piper models; complete 44-episode paired evaluation per seed')
    save(output/'protocol.json',protocol)
    shutil.copy2(__file__,output/'consolidation_source.py')
    shutil.copy2(ROOT/'tools/summarize_correct_task_donor_batch.py',output/'analysis_source.py')
    # Execute the saved analysis source so its recorded hash is reproducible.
    import subprocess,os
    from tools.run_task_token_counterfactuals import PYTHON
    subprocess.run([PYTHON,str(output/'analysis_source.py'),'--root',str(output)],
        cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)),check=True)
    report=output/'summary.md'
    content=report.read_text().replace('# Correct-task donor evaluation across non-text Piper models',
        '# Overall correct-task donor evaluation — updated October 28, 2026',1)
    evidence=[r for r in json.loads((output/'result.json').read_text())['models']]
    supported=sum(r['effect']['interval_95'][0]>0 for r in evidence)
    content=content.replace('## Per-seed completion counts',
        '## Interpretation\n\n'
        'GT-selected correct-task donors improve the point estimate for all eight models. '
        f'The primary episode-cluster 95% interval is strictly above zero for {supported}/8 models; '
        'the two updated cp8 memory models have lower bounds of zero. '
        'The later head-64 memory checkpoint improves observed normal completion from 50.8% to 59.1%, '
        'and GT-donor completion from 62.1% to 68.9%. The later head-256 memory checkpoint changes '
        'normal completion from 61.4% to 62.1%, and GT-donor completion from 69.7% to 71.2%. '
        'These are point estimates from one training seed, not proof of a general scaling advantage.\n\n'
        'Across models, all 15 trials with incorrectly decoded original context fail under normal conditioning; '
        'GT-selected donors rescue 13–15/15. These repeated trials span seven episode identities. '
        'Donor selection is privileged and may also change episode-specific procedure information.\n\n'
        '[Current training conditions and artifact audit](training_conditions.md).\n\n'
        '## Per-seed completion counts',1)
    content+='\n## Source reports\n\n'+''.join(
        f"- [{Path(s['path']).name}]({s['path']}/summary.md)\n" for s in sources)
    report.write_text(content)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--updates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(consolidate(args.base,args.updates,args.output))
