"""Training-only task probes, trace diagnostics, and peak-score snapshot selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


SCHEMA = 'align-task-label-probe-v1'


def context_fingerprint(branch):
    """Bind a label probe to the complete observation/context coordinate system."""
    digest = hashlib.sha256()
    digest.update(json.dumps(branch.config, sort_keys=True).encode())
    for key, value in sorted(branch.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(f'{key}:{value.dtype}:{tuple(value.shape)}'.encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def canonical_task_name(name):
    return ' '.join(str(name).casefold().split())


class TaskLabelEstimator:
    """A fixed multinomial linear probe; predict() has no task-label argument.

    Task labels are report-only. Scores optionally select native task snapshots
    in highest-score freeze mode; they are not calibrated takeover confidence.
    """
    def __init__(self, path):
        self.path = str(path)
        with np.load(path, allow_pickle=False) as source:
            self.metadata = json.loads(str(source['metadata']))
            if self.metadata.get('schema') != SCHEMA:
                raise ValueError('Unsupported task-estimator schema')
            self.task_names = source['task_names'].tolist()
            self.latent_mean = source['latent_mean'].astype(np.float64)
            self.latent_std = source['latent_std'].astype(np.float64)
            self.feature_mean = source['feature_mean'].astype(np.float64)
            self.feature_scale = source['feature_scale'].astype(np.float64)
            self.weight = source['weight'].astype(np.float64)
            self.bias = source['bias'].astype(np.float64)
        self.token_shape = tuple(self.latent_mean.shape)
        width = int(np.prod(self.token_shape))
        names = [canonical_task_name(name) for name in self.task_names]
        if len(names) < 2 or len(set(names)) != len(names):
            raise ValueError('Expected unique task names')
        if self.latent_std.shape != self.token_shape or self.feature_mean.shape != (width,) or self.feature_scale.shape != (width,):
            raise ValueError('Invalid task-estimator normalization shape')
        if self.weight.shape != (len(names), width) or self.bias.shape != (len(names),):
            raise ValueError('Invalid task-estimator classifier shape')
        for value in (self.latent_mean, self.latent_std, self.feature_mean, self.feature_scale, self.weight, self.bias):
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite task-estimator parameters')
        if (self.latent_std <= 0).any() or (self.feature_scale <= 0).any():
            raise ValueError('Task-estimator scales must be positive')
        if not self.metadata.get('training_only') or self.metadata.get('confirmation_opened'):
            raise ValueError('Task-estimator must be fitted on training episodes only')
        self.name_to_group = {name:i for i,name in enumerate(names)}

    def validate_branch(self, branch):
        if context_fingerprint(branch) != self.metadata['context_fingerprint']:
            raise ValueError('Task estimator does not match the policy context encoder/normalization')

    def probabilities(self, native_tokens):
        if isinstance(native_tokens, torch.Tensor):
            native_tokens = native_tokens.detach().float().cpu().numpy()
        value = np.asarray(native_tokens, dtype=np.float64)
        if value.ndim != 3 or value.shape[1:] != self.token_shape:
            raise ValueError(f'Expected B,{self.token_shape} native task tokens')
        if not np.isfinite(value).all():
            raise ValueError('Task tokens must be finite')
        normalized = ((value-self.latent_mean)/self.latent_std).reshape(len(value),-1)
        features = (normalized-self.feature_mean)/self.feature_scale
        logits = features @ self.weight.T + self.bias
        probability = np.exp(logits-logits.max(1,keepdims=True))
        return probability / probability.sum(1,keepdims=True)

    def predict(self, native_tokens):
        probabilities = self.probabilities(native_tokens)
        return [dict(predicted_task=self.task_names[int(p.argmax())],
                     predicted_group=int(p.argmax()), top1_probability=float(p.max()))
                for p in probabilities]


class PeakTaskSnapshot:
    """Keep the native tokens with the highest observed top-task decoder score.

    Ground-truth labels are never consulted. Strict improvements replace the
    snapshot; equal scores retain the earlier snapshot. State is episode-local.
    """
    def __init__(self, estimator):
        self.estimator = estimator
        self.tokens = None
        self.best_score = -float('inf')
        self.selected_step = None
        self.updates = []

    def observe(self, step, native_tokens):
        probability = self.estimator.probabilities(native_tokens)
        if probability.shape[0] != 1:
            raise ValueError('Peak task selection requires one episode at a time')
        group = int(probability[0].argmax())
        score = float(probability[0,group])
        updated = score > self.best_score
        if updated:
            self.tokens = (native_tokens.detach().clone() if isinstance(native_tokens,torch.Tensor)
                           else np.asarray(native_tokens).copy())
            self.best_score, self.selected_step = score, int(step)
            self.updates.append(dict(step=int(step),score=score,predicted_group=group,
                                     predicted_task=self.estimator.task_names[group]))
        selection = dict(strategy='highest_decoder_score',candidate_score=score,
            selected_score=self.best_score,selected_step=self.selected_step,updated=updated)
        return self.tokens, selection

    def summary(self):
        return dict(strategy='highest_decoder_score',score='maximum task decoder probability',
            probabilities_calibrated=False,ground_truth_used=False,
            replacement_rule='strictly greater; ties keep previous snapshot',
            selected_step=self.selected_step,selected_score=self.best_score if self.tokens is not None else None,
            updates=list(self.updates))


class TaskEstimationTrace:
    """Compare a fixed probe with GT for reporting, after inference is complete."""
    def __init__(self, estimator, actual_task_name, switch_step, reference_steps, context_intervention=None):
        self.estimator = estimator
        self.actual_task_name = actual_task_name
        self.actual_group = estimator.name_to_group.get(canonical_task_name(actual_task_name))
        self.switch_step = int(switch_step)
        self.reference_steps = int(reference_steps)
        self.records = []
        self.inferred_tokens, self.policy_tokens, self.scores = [], [], []
        self.execution_tokens = []
        self.snapshot_selection = []
        self.context_intervention = context_intervention

    def observe(self, step, inferred_tokens, policy_tokens, execution_token=None, snapshot_selection=None):
        def array(value):
            if isinstance(value, torch.Tensor):
                value = value.detach().float().cpu().numpy()
            value = np.asarray(value,dtype=np.float32)
            if value.shape != (1,*self.estimator.token_shape):
                raise ValueError('Task trace requires one episode at a time')
            return value[0].copy()
        inferred, policy = array(inferred_tokens), array(policy_tokens)
        execution = None
        if execution_token is not None:
            if isinstance(execution_token,torch.Tensor):
                execution_token = execution_token.detach().float().cpu().numpy()
            execution = np.asarray(execution_token,dtype=np.float32)
            if execution.shape != (1,1,self.estimator.token_shape[-1]) or not np.isfinite(execution).all():
                raise ValueError('Expected one finite execution token matching the context width')
            execution = execution[0,0].copy()
        if self.records and (execution is None) != (self.execution_tokens[0] is None):
            raise ValueError('Execution tokens must be recorded for every observation or none')
        # Labels are consulted only after the classifier produces its scores.
        scores = self.estimator.probabilities(np.stack([inferred,policy]))
        row = dict(step=int(step), phase='expert' if step < self.switch_step else 'policy')
        if snapshot_selection is not None:
            row['snapshot_selection'] = dict(snapshot_selection)
        for role, probability in zip(('inferred','policy'),scores):
            group = int(probability.argmax())
            row[role] = dict(predicted_task=self.estimator.task_names[group], predicted_group=group,
                top1_probability=float(probability[group]),
                correct=(group == self.actual_group) if self.actual_group is not None else None)
        self.records.append(row)
        self.inferred_tokens.append(inferred)
        self.policy_tokens.append(policy)
        self.scores.append(scores)
        self.execution_tokens.append(execution)
        self.snapshot_selection.append(dict(snapshot_selection) if snapshot_selection is not None else None)

    def summary(self):
        rows = {row['step']:row for row in self.records}
        checkpoints = dict(initial=rows.get(0), handover=rows.get(self.switch_step),
            prefix_40pct=rows.get(int(self.reference_steps*.4)),
            last_reference=rows.get(self.reference_steps-1),
            final=self.records[-1] if self.records else None)
        accuracy = {}
        for phase in ('all','expert','policy'):
            selected = [r for r in self.records if phase == 'all' or r['phase'] == phase]
            accuracy[phase] = {}
            for role in ('inferred','policy'):
                correct = [r[role]['correct'] for r in selected if r[role]['correct'] is not None]
                accuracy[phase][role] = dict(correct=sum(correct), observations=len(correct),
                    accuracy=float(np.mean(correct)) if correct else None)
        controlled = any(value is not None for value in self.snapshot_selection)
        selection = (dict(task_snapshot_strategy='highest_decoder_score',estimator_is_policy_input=True)
                     if controlled else {})
        if self.context_intervention is not None:
            controlled = True
            selection = dict(task_snapshot_strategy=self.context_intervention['strategy'],
                estimator_is_policy_input=True, oracle_context=dict(self.context_intervention),
                privileged=True, deployable=False)
        return dict(schema='align-task-estimation-trace-v1',actual_task=self.actual_task_name,
            actual_group=self.actual_group, actual_task_known=self.actual_group is not None,
            diagnostic_only=not controlled, probabilities_calibrated=False,
            ground_truth_is_model_input=self.context_intervention is not None,
            estimator_fingerprint=self.estimator.metadata['context_fingerprint'],
            observations=len(self.records),checkpoints=checkpoints,observation_accuracy=accuracy,
            task_names=self.estimator.task_names,**selection)

    def save(self, base):
        """Save named per-observation predictions and raw tokens for later audits."""
        base = Path(base)
        extra = (dict(execution_tokens=np.asarray(self.execution_tokens))
                 if self.execution_tokens and self.execution_tokens[0] is not None else {})
        if self.snapshot_selection and all(value is not None for value in self.snapshot_selection):
            extra.update(snapshot_steps=np.asarray([s['selected_step'] for s in self.snapshot_selection]),
                snapshot_scores=np.asarray([s['selected_score'] for s in self.snapshot_selection]),
                snapshot_updated=np.asarray([s['updated'] for s in self.snapshot_selection]),
                candidate_scores=np.asarray([s['candidate_score'] for s in self.snapshot_selection]))
        np.savez_compressed(str(base)+'.npz',inferred_tokens=np.asarray(self.inferred_tokens),
            policy_tokens=np.asarray(self.policy_tokens),probabilities=np.asarray(self.scores),
            steps=np.asarray([r['step'] for r in self.records]),
            task_names=np.asarray(self.estimator.task_names),actual_task=np.asarray(str(self.actual_task_name or '')),**extra)
        Path(str(base)+'.json').write_text(json.dumps(dict(summary=self.summary(),observations=self.records),indent=2)+'\n')


def summarize_task_estimation(episodes):
    """Episode-weighted checkpoint accuracy, also separated by replay outcome."""
    available = [row for row in episodes if row.get('task_estimation')]
    if not available:
        return None
    selections = dict(all=available,
        expert_replay_pass=[r for r in available if r.get('replay_check',{}).get('valid') is True],
        expert_replay_fail=[r for r in available if r.get('replay_check',{}).get('valid') is False],
        expert_task_failure=[r for r in available if r.get('replay_check',{}).get('task_success') is False],
        policy_success=[r for r in available if r.get('success') is True],
        policy_failure=[r for r in available if r.get('success') is False])
    controlled = any(not row['task_estimation'].get('diagnostic_only',True) for row in available)
    result = dict(episodes=len(available),diagnostic_only=not controlled,
        estimator_is_policy_input=controlled,probabilities_calibrated=False,groups={})
    for name, selected in selections.items():
        checkpoints = {}
        for point in ('initial','handover','prefix_40pct','last_reference','final'):
            checkpoints[point] = {}
            for role in ('inferred','policy'):
                rows = [r['task_estimation'] for r in selected
                    if r['task_estimation']['checkpoints'].get(point) is not None
                    and r['task_estimation']['actual_task_known']]
                correct = [r['checkpoints'][point][role]['correct'] for r in rows]
                groups = sorted({r['actual_group'] for r in rows})
                recalls = [np.mean([r['checkpoints'][point][role]['correct'] for r in rows
                                    if r['actual_group'] == group]) for group in groups]
                checkpoints[point][role] = dict(correct=sum(correct),episodes=len(correct),
                    accuracy=float(np.mean(correct)) if correct else None,
                    balanced_accuracy=float(np.mean(recalls)) if recalls else None)
        result['groups'][name] = dict(episodes=len(selected),checkpoints=checkpoints)
    return result
