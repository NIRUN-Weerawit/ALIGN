"""Explicit checkpoint migration between flattened and patch memory retrieval."""
import json
import os
from pathlib import Path
import shutil
import time


PERCEPTUAL_PREFIXES = ('memory_module.perceptual_retrieval.',
                       'memory_module.perceptual_gate.',
                       'memory_module.visual_intent_projection.')


def memory_patch_transition(model, checkpoint):
    return (bool(model.use_memory_bank) and
            bool(checkpoint.get('config', {}).get('memory_patch_retrieval', False)) !=
            bool(model.memory_patch_retrieval))


def migrate_memory_weights(model, checkpoint):
    """Keep compatible weights, reset changed perceptual-memory projections.

    Adam state must be reset by the caller: its unnamed parameter IDs cannot
    safely identify the changed tensors in historical checkpoints.
    """
    source = checkpoint['model_state_dict']
    own = model.state_dict()
    kept = {}
    for name, value in source.items():
        if name.startswith(PERCEPTUAL_PREFIXES) or name.startswith('vision_encoder.'):
            continue
        if name not in own or own[name].shape != value.shape:
            raise ValueError(f'Unexpected non-perceptual weight mismatch during memory migration: {name}')
        kept[name] = value
    missing = [name for name in own if name not in kept and
               not name.startswith((*PERCEPTUAL_PREFIXES, 'vision_encoder.'))]
    if missing:
        raise ValueError(f'Memory migration checkpoint is missing compatible weights: {missing[:5]}')
    model.load_state_dict(kept, strict=False)
    return dict(from_memory_patch_retrieval=bool(checkpoint.get('config', {}).get('memory_patch_retrieval', False)),
                to_memory_patch_retrieval=bool(model.memory_patch_retrieval),
                source_epoch=int(checkpoint['epoch']), continued_at_epoch=int(checkpoint['epoch']) + 1,
                compatible_tensors_loaded=len(kept), reset_modules=list(PERCEPTUAL_PREFIXES),
                optimizer_state='reset', best_selection='reset for changed memory architecture')


def archive_memory_transition(run, checkpoint_paths, metadata):
    """Preserve old weights with hard links, avoiding copies of multi-GB files.

    Each archive lives beside its source, including local checkpoint staging.
    Atomic replacement of canonical checkpoints leaves these immutable links intact.
    """
    tag = f'memory_patch_before_{time.time_ns()}'
    locations = {}
    files = []
    for source in dict.fromkeys(Path(p) for p in checkpoint_paths):
        if not source.is_file():
            continue
        directory = source.parent/tag
        directory.mkdir(exist_ok=True)
        os.link(source, directory/source.name)
        locations[str(source.parent)] = directory
        files.append(dict(source=str(source), archive=str(directory/source.name)))
    run = Path(run)
    directory = locations.get(str(run), run/tag)
    directory.mkdir(exist_ok=True)
    for name in ('config.json', 'metrics.jsonl'):
        if (run/name).is_file():
            shutil.copy2(run/name, directory/name)
    metadata = dict(metadata, archived_checkpoints=files, archive=str(directory))
    (directory/'migration.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata
