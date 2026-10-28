import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eval.task_context_oracle import SCHEMA, OracleTaskSnapshot, TaskContextOracleBank
from eval.task_estimation import TaskEstimationTrace
from data.task_context_dataset import file_hash


class Estimator:
    token_shape = (4, 2)
    task_names = ['task a', 'task b']
    name_to_group = {'task a': 0, 'task b': 1}
    metadata = {'context_fingerprint': 'frozen'}
    def probabilities(self, tokens):
        value = np.asarray(tokens.detach().cpu() if isinstance(tokens,torch.Tensor) else tokens)
        p = np.where(value[:,0,0] > 0, .9, .1)
        return np.stack([1-p,p],1)
    def predict(self, tokens):
        return [dict(predicted_task=self.task_names[int(p.argmax())], predicted_group=int(p.argmax()),
                     top1_probability=float(p.max())) for p in self.probabilities(tokens)]


def test_oracle_repair_preserves_correct_native_snapshot_and_never_changes_early_context():
    estimator = Estimator()
    oracle = OracleTaskSnapshot(dict(tokens=np.full((4,2),-5.,dtype='f4'),task='task a',source_step=9),'repair',estimator)
    current = torch.full((1,4,2),-1.)
    assert oracle.observe(1,3,current) is current
    chosen = oracle.observe(3,3,current)
    assert torch.equal(chosen,current)
    current.fill_(7.)
    assert torch.equal(oracle.observe(4,3,current),torch.full((1,4,2),-1.))
    assert oracle.info['applied'] and not oracle.info['replaced'] and not oracle.info['uses_future_demonstration']


@pytest.mark.parametrize('mode',['repair','episode'])
def test_oracle_replaces_wrong_task_and_holds_snapshot(mode):
    oracle = OracleTaskSnapshot(dict(tokens=np.full((4,2),-5.,dtype='f4'),task='task a',source_step=9),mode,Estimator())
    assert (oracle.observe(3,3,torch.ones(1,4,2)) == -5).all()
    assert (oracle.observe(5,3,torch.full((1,4,2),-2.)) == -5).all()
    assert oracle.info['replaced'] and oracle.info['uses_future_demonstration']
    trace = TaskEstimationTrace(Estimator(),'task a',3,6,context_intervention=oracle.info)
    trace.observe(3,torch.ones(1,4,2),oracle.tokens,execution_token=torch.ones(1,1,2))
    summary=trace.summary()
    assert summary['ground_truth_is_model_input'] and summary['privileged'] and not summary['deployable']
    assert not summary['diagnostic_only']
    assert summary['checkpoints']['handover']['policy']['correct']
    assert not summary['checkpoints']['handover']['inferred']['correct']


def test_oracle_bank_rejects_mismatched_identity_checkpoint_or_split(tmp_path):
    estimator=Estimator(); estimator.path=str(tmp_path/'estimator'); (tmp_path/'estimator').write_text('fixed probe')
    metadata=dict(schema=SCHEMA,confirmation_opened=False,checkpoint_sha256='weights',context_fingerprint='frozen',
                  estimator_sha256=file_hash(estimator.path),definition='test')
    path=tmp_path/'bank.npz'
    def write(value=-1.):
        np.savez(path,metadata=np.asarray(json.dumps(metadata)),keys=np.asarray(['dev']),
                 task_names=np.asarray(['task a']),tokens=np.full((1,4,2),value,dtype='f4'),source_steps=np.asarray([9]))
    cfg=dict(task_context_split=dict(development_episodes=['dev']))
    write(); bank=TaskContextOracleBank(path,estimator,cfg,'weights')
    assert bank.episode('dev','task a','repair').info['privileged']
    with pytest.raises(ValueError,match='episode task'):
        bank.episode('dev','task b','repair')
    with pytest.raises(ValueError,match='bound'):
        TaskContextOracleBank(path,estimator,cfg,'changed')
    with pytest.raises(ValueError,match='development'):
        TaskContextOracleBank(path,estimator,dict(task_context_split=dict(development_episodes=['confirmation'])),'weights')
    write(1.)
    with pytest.raises(ValueError,match='ground-truth'):
        TaskContextOracleBank(path,estimator,cfg,'weights')
