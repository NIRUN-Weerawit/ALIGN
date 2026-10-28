"""Privileged, evaluation-only task snapshots; never normal policy inference."""
import json
from pathlib import Path
import numpy as np
import torch

from eval.task_estimation import canonical_task_name
from data.task_context_dataset import file_hash

SCHEMA = 'align-native-task-oracle-bank-v1'


class TaskContextOracleBank:
    def __init__(self, path, estimator, config, checkpoint_sha256):
        self.path = str(path)
        with np.load(path, allow_pickle=False) as source:
            self.metadata = json.loads(str(source['metadata']))
            keys = source['keys'].tolist()
            names = source['task_names'].tolist()
            tokens = source['tokens'].copy()
            steps = source['source_steps'].tolist()
        if (self.metadata.get('schema') != SCHEMA or self.metadata.get('confirmation_opened')
                or self.metadata.get('checkpoint_sha256') != checkpoint_sha256
                or self.metadata.get('context_fingerprint') != estimator.metadata['context_fingerprint']
                or self.metadata.get('estimator_sha256') != file_hash(estimator.path)):
            raise ValueError('Oracle bank is not bound to this checkpoint, estimator, and protected split')
        development = config['task_context_split']['development_episodes']
        if len(keys) != len(set(keys)) or set(keys) != set(development):
            raise ValueError('Oracle bank must contain exactly the saved development episodes')
        if (tokens.shape != (len(keys), *estimator.token_shape) or not np.isfinite(tokens).all()
                or len(names) != len(keys) or len(steps) != len(keys)):
            raise ValueError('Invalid oracle token shapes or values')
        predictions = estimator.predict(tokens)
        if any(canonical_task_name(p['predicted_task']) != canonical_task_name(name)
               for p, name in zip(predictions, names)):
            raise ValueError('Every oracle snapshot must decode to its ground-truth task')
        self.entries = {key: dict(tokens=token, task=name, source_step=int(step))
                        for key, token, name, step in zip(keys, tokens, names, steps)}
        self.estimator = estimator

    def episode(self, key, actual_task, mode):
        entry = self.entries[key]
        if canonical_task_name(actual_task) != canonical_task_name(entry['task']):
            raise ValueError('Oracle episode task differs from the evaluation task')
        return OracleTaskSnapshot(entry, mode, self.estimator)

    def provenance(self, mode):
        return dict(mode=mode, path=self.path, sha256=file_hash(self.path),
            ground_truth_is_model_input=True, privileged=True, deployable=False,
            future_demonstration_may_be_used=True, definition=self.metadata['definition'],
            guarantee='Correct known-task decoder identity, not perfect full task/context semantics')


class OracleTaskSnapshot:
    def __init__(self, entry, mode, estimator):
        if mode not in ('repair', 'episode'):
            raise ValueError('Oracle mode must be repair or episode')
        self.entry, self.mode, self.estimator = entry, mode, estimator
        self.tokens = None
        self.info = dict(strategy='oracle_'+mode, ground_truth_is_model_input=True,
            privileged=True, deployable=False, applied=False, replaced=False,
            source_step=entry['source_step'], actual_task=entry['task'],
            uses_future_demonstration=False,
            guarantee='Correct task decoder identity; complete semantic correctness is unverified')

    def observe(self, step, switch_step, inferred_tokens):
        if step < switch_step:
            return inferred_tokens
        if self.tokens is None:
            decoded = self.estimator.predict(inferred_tokens)[0]
            correct = canonical_task_name(decoded['predicted_task']) == canonical_task_name(self.entry['task'])
            replace = self.mode == 'episode' or not correct
            chosen = (torch.as_tensor(self.entry['tokens'], device=inferred_tokens.device,
                                     dtype=inferred_tokens.dtype)[None]
                      if replace else inferred_tokens)
            self.tokens = chosen.detach().clone()
            if canonical_task_name(self.estimator.predict(self.tokens)[0]['predicted_task']) != canonical_task_name(self.entry['task']):
                raise ValueError('Oracle task identity check failed at handover')
            self.info.update(applied=True, replaced=replace, handover_step=int(step),
                original_task=decoded['predicted_task'], original_correct=correct,
                uses_future_demonstration=bool(replace and self.entry['source_step'] > switch_step))
        return self.tokens

    def summary(self):
        return dict(self.info)
