"""Build GT-selected native student snapshots for a privileged bottleneck test."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import torch
from data.task_context_dataset import file_hash, json_hash
from eval.task_context_oracle import SCHEMA
from eval.task_estimation import TaskLabelEstimator, canonical_task_name
from models.task_context_policy import TaskContextPolicyBranch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--prefix-cache', type=Path, required=True)
    p.add_argument('--token-probe', type=Path, required=True)
    p.add_argument('--task-estimator', type=Path, required=True)
    p.add_argument('--handover-fraction', type=float, default=.4)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new oracle bank path')
    torch.set_num_threads(2)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg = saved['config']
    branch = TaskContextPolicyBranch(config=cfg['task_context_config']).eval()
    branch.load_state_dict({k.removeprefix('task_context.'): v for k, v in saved['model_state_dict'].items()
                            if k.startswith('task_context.')}, strict=True)
    estimator = TaskLabelEstimator(args.task_estimator)
    estimator.validate_branch(branch)
    manifest = json.loads((args.prefix_cache/'manifest.json').read_text())
    if json_hash({k:v for k,v in manifest.items() if k!='manifest_sha256'}) != manifest['manifest_sha256']:
        raise ValueError('Invalid prefix-cache manifest')
    for name in ('train', 'development', 'confirmation'):
        if set(cfg['task_context_split'][name+'_episodes']) != set(manifest['split'][name]):
            raise ValueError('Policy and prefix splits differ')
    probe = json.loads((args.token_probe/'protocol.json').read_text())
    if probe['prefix_manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('Probe/prefix-cache binding differs')
    source_path = args.token_probe/'development_features.npz'
    with np.load(source_path, allow_pickle=False) as source:
        keys = source['keys'].tolist()
        if len(keys)!=len(set(keys)) or set(keys)!=set(manifest['split']['development']):
            raise ValueError('Expected exactly development features')
        native = (source['tokens']*estimator.latent_std+estimator.latent_mean).astype(np.float32)
        probabilities = estimator.probabilities(native.reshape(-1,*estimator.token_shape)).reshape(len(keys),-1,len(estimator.task_names))
        groups = source['groups']
        anchors = source['anchors'].copy()
        query_indices = [i for i,q in enumerate(probe['queries']) if q.get('fraction',-1)>=args.handover_fraction]
        rows, tokens = [], []
        with h5py.File(args.data,'r') as data:
            for i,key in enumerate(keys):
                raw = data[key]['texts'][()]
                actual = json.loads(raw.decode() if isinstance(raw,bytes) else raw)[0]
                group = estimator.name_to_group[canonical_task_name(actual)]
                if int(groups[i]) != group:
                    raise ValueError('Task group mapping differs')
                prefix_path=args.prefix_cache/f'{key}.npz'
                if file_hash(prefix_path)!=manifest['files'][key]:
                    raise ValueError('Prefix file hash differs')
                candidates = [q for q in query_indices if probabilities[i,q].argmax()==group]
                if not candidates:
                    raise ValueError(f'{key}: no correct native snapshot; do not invent one')
                q = min(candidates,key=lambda q:(int(anchors[i,q]),q))
                tokens.append(native[i,q])
                rows.append(dict(episode=key,task=actual,source_step=int(anchors[i,q]),query=probe['queries'][q],
                    decoder_score=float(probabilities[i,q,group]),
                    future_relative_to_handover=bool(anchors[i,q]>int(source['lengths'][i]*args.handover_fraction))))
        # Check the cached coordinate system with a fresh causal readout.
        first=keys[0]
        with np.load(args.prefix_cache/f'{first}.npz',allow_pickle=False) as values:
            summaries=torch.from_numpy(values['summary'].copy())[None]
        with torch.no_grad():
            out=branch.observer.encoder(summaries,torch.tensor([summaries.shape[1]]),torch.from_numpy(anchors[0])[None])
        error=float((out['tokens'][0]-torch.from_numpy(native[0])).abs().max())
        if error>2e-4:
            raise ValueError(f'Cached student tokens differ from checkpoint: {error}')
    metadata = dict(schema=SCHEMA, definition='Earliest correctly decoded same-episode native student snapshot among declared fractions at or after handover',
        checkpoint_sha256=file_hash(args.checkpoint), context_fingerprint=estimator.metadata['context_fingerprint'],
        estimator_sha256=file_hash(args.task_estimator), features_sha256=file_hash(source_path),
        prefix_manifest_sha256=manifest['manifest_sha256'], handover_fraction=args.handover_fraction,
        confirmation_opened=False, ground_truth_used=True, privileged=True, deployable=False,
        weights_updated=False, selection_uses_policy_outcomes=False, causal_readout_max_error=error,
        identity_correct=len(rows), episodes=len(rows), rows=rows,
        limitations=['Correct decoder identity does not guarantee perfect complete context.',
            'Uses GT task labels and potentially future recorded observations.',
            'Recorded demonstration scenes may differ from seeded simulator scenes.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,metadata=np.asarray(json.dumps(metadata)),keys=np.asarray(keys),
        task_names=np.asarray([r['task'] for r in rows]),tokens=np.stack(tokens),
        source_steps=np.asarray([r['source_step'] for r in rows]))
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Oracle identity coverage {len(rows)}/{len(rows)}; later snapshots {sum(r["future_relative_to_handover"] for r in rows)}; causal audit error {error:.3g}')


if __name__=='__main__':
    main()
