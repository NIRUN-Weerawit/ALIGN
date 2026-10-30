"""Memory architecture transitions must not discard unrelated learned weights."""
import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from models.memory_bank import EpisodicMemoryModule
from training.memory_resume import (memory_patch_transition, migrate_memory_weights,
                                    archive_memory_transition, PERCEPTUAL_PREFIXES)


class Policy(nn.Module):
    def __init__(self, patch, memory=True):
        super().__init__()
        self.use_memory_bank, self.memory_patch_retrieval = memory, patch
        self.state_encoder = nn.Linear(4, 4)
        self.intention_head = nn.Linear(4, 7)
        self.memory_module = EpisodicMemoryModule(perceptual_dim=8,cognitive_dim=4,
            state_dim=4,bank_len=2,num_heads=2,patch_dim=2 if patch else None,
            retrieval_depth=2,intent_query=True) if memory else None


@pytest.mark.parametrize('source_patch,target_patch', [(False, True), (True, False)])
def test_migration_keeps_all_unaffected_weights_and_resets_changed_memory(source_patch,target_patch):
    original, model = Policy(source_patch), Policy(target_patch)
    with torch.no_grad():
        for p in original.parameters():p.fill_(.37)
    saved=dict(config=dict(memory_patch_retrieval=source_patch),epoch=7,
               model_state_dict=original.state_dict())
    before={k:v.clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters())
    assert memory_patch_transition(model,saved)
    metadata=migrate_memory_weights(model,saved)
    for name,value in model.state_dict().items():
        expected=before[name] if name.startswith(PERCEPTUAL_PREFIXES) else saved['model_state_dict'][name]
        assert torch.equal(value,expected),name
    assert metadata['continued_at_epoch']==8 and metadata['optimizer_state']=='reset'
    assert not optimizer.state
    assert not memory_patch_transition(model,dict(config=dict(memory_patch_retrieval=target_patch)))


def test_flag_change_does_not_migrate_stateless_variants():
    assert not memory_patch_transition(Policy(True,memory=False),dict(config=dict(memory_patch_retrieval=False)))


def test_patch_memory_never_initializes_giant_flattened_layers(monkeypatch):
    import models.memory_bank as bank
    dimensions=[]
    original=bank.MemoryRetrieval
    def tracked(dim,*args,**kwargs):
        dimensions.append(dim)
        return original(dim,*args,**kwargs)
    monkeypatch.setattr(bank,'MemoryRetrieval',tracked)
    with torch.device('meta'):
        memory=bank.EpisodicMemoryModule(perceptual_dim=16384,cognitive_dim=0,
            state_dim=64,patch_dim=32,num_heads=2,retrieval_depth=2)
    assert set(dimensions)=={32,64}
    assert sum(p.numel() for p in memory.parameters())==89536


def test_migration_does_not_hide_unrelated_shape_mismatches_or_missing_weights():
    saved=dict(config=dict(memory_patch_retrieval=False),epoch=1,model_state_dict=Policy(False).state_dict())
    saved['model_state_dict']['intention_head.weight']=torch.zeros(2,2)
    with pytest.raises(ValueError,match='non-perceptual weight mismatch'):
        migrate_memory_weights(Policy(True),saved)
    saved['model_state_dict']=Policy(False).state_dict()
    saved['model_state_dict'].pop('state_encoder.weight')
    with pytest.raises(ValueError,match='missing compatible weights'):
        migrate_memory_weights(Policy(True),saved)


def test_archives_preserve_old_checkpoints_after_atomic_replacement(tmp_path):
    run=tmp_path/'run';stage=tmp_path/'stage'
    run.mkdir();stage.mkdir()
    source=stage/'training_last.pt';torch.save(dict(epoch=7),source)
    best=run/'intention_best.pt';torch.save(dict(epoch=5),best)
    (run/'config.json').write_text('{"memory_patch_retrieval": false}')
    (run/'metrics.jsonl').write_text('{"epoch": 7}\n')
    metadata=archive_memory_transition(run,[source,best,source],dict(source_epoch=7))
    assert len(metadata['archived_checkpoints'])==2
    for row in metadata['archived_checkpoints']:
        assert os.stat(row['source']).st_ino==os.stat(row['archive']).st_ino
    replacement=stage/'new.pt';torch.save(dict(epoch=8),replacement);replacement.replace(source)
    archive=Path(metadata['archived_checkpoints'][0]['archive'])
    assert torch.load(archive,weights_only=True)['epoch']==7
    assert torch.load(source,weights_only=True)['epoch']==8
    assert json.loads((Path(metadata['archive'])/'config.json').read_text())['memory_patch_retrieval'] is False
    assert (Path(metadata['archive'])/'metrics.jsonl').read_text()=='{"epoch": 7}\n'
