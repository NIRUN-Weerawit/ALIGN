"""Fit a task-name diagnostic on training episodes and export a NumPy runtime.

The single all-prefix classifier can run causally at any observation without
knowing total episode length, task name, or phase. Neural weights stay frozen.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from data.task_context_dataset import file_hash
from eval.task_estimation import SCHEMA, TaskLabelEstimator, context_fingerprint
from models.task_context_policy import TaskContextPolicyBranch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new estimator path')
    torch.set_num_threads(2)
    result = json.loads((args.probe/'results.json').read_text())
    protocol = result['protocol']
    if protocol['policy_sha256'] != file_hash(args.policy):
        raise ValueError('Probe/policy identity differs')
    train = dict(np.load(args.probe/'train_features.npz',allow_pickle=False))
    if train['keys'].tolist() != protocol['splits']['train']:
        raise ValueError('Training features differ from recorded protected split')
    if set(train['keys']) & set(protocol['splits']['development']):
        raise ValueError('Training/development overlap')
    checkpoint = torch.load(args.policy,map_location='cpu',weights_only=True,mmap=True)
    cfg = checkpoint['config']
    if cfg['task_context_split']['train_episodes'] != train['keys'].tolist():
        raise ValueError('Policy training partition differs')
    branch = TaskContextPolicyBranch(config=cfg['task_context_config']).requires_grad_(False).eval()
    prefix = 'task_context.'
    branch.load_state_dict({k[len(prefix):]:v for k,v in checkpoint['model_state_dict'].items()
                           if k.startswith(prefix)},strict=True)
    x = train['tokens'].reshape(-1,int(np.prod(train['tokens'].shape[-2:]))).astype(np.float64)
    labels = np.repeat(train['groups'],train['tokens'].shape[1])
    if not np.array_equal(np.unique(labels),np.arange(len(result['task_names']))):
        raise ValueError('Expected all task classes in training features')
    # All predeclared training queries are pooled for an online decoder. No
    # development features, predictions or labels are used to fit/select it.
    classifier = make_pipeline(StandardScaler(),LogisticRegression(
        C=.1,class_weight='balanced',max_iter=3000,random_state=42))
    with threadpool_limits(limits=2):
        classifier.fit(x,labels)
    scale,linear = classifier[0],classifier[1]
    metadata = dict(schema=SCHEMA,training_only=True,confirmation_opened=False,
        neural_weights_updated=False,policy_sha256=protocol['policy_sha256'],
        context_fingerprint=context_fingerprint(branch),
        prefix_manifest_sha256=protocol['prefix_manifest_sha256'],
        train_features_sha256=file_hash(args.probe/'train_features.npz'),
        fitting_episodes=train['keys'].tolist(),fitting_observations=len(x),
        fitting_queries=protocol['queries'],classes=linear.classes_.tolist(),
        fitting='All recorded training prefixes pooled; StandardScaler + balanced multinomial logistic regression',
        C=.1,max_iter=3000,random_state=42,probabilities_calibrated=False,
        estimator_is_policy_input=False,source_sha256=file_hash(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,metadata=np.asarray(json.dumps(metadata)),
        task_names=np.asarray(result['task_names']),latent_mean=branch.observer.encoder.latent_mean.numpy(),
        latent_std=branch.observer.encoder.latent_std.numpy(),feature_mean=scale.mean_,
        feature_scale=scale.scale_,weight=linear.coef_,bias=linear.intercept_)
    # Check that the pickle-free exported runtime matches sklearn probabilities
    # on actual training vectors, including the native latent conversion.
    runtime = TaskLabelEstimator(args.output)
    runtime.validate_branch(branch)
    native = train['tokens'][:16,3]*runtime.latent_std+runtime.latent_mean
    with threadpool_limits(limits=2):
        expected = classifier.predict_proba(train['tokens'][:16,3].reshape(16,-1).astype(np.float64))
        actual = runtime.probabilities(native)
    error = float(np.max(np.abs(actual-expected)))
    if error > 1e-8:
        raise ValueError(f'Exported classifier disagrees: {error}')
    Path(str(args.output)+'.json').write_text(json.dumps(dict(metadata=metadata,
        artifact_sha256=file_hash(args.output),runtime_probability_max_error=error),indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),episodes=len(train['keys']),observations=len(x),
                          runtime_probability_max_error=error),indent=2))


if __name__ == '__main__':
    main()
