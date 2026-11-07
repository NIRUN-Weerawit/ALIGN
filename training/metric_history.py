"""Recover advisory metric logs when resuming from an authoritative checkpoint."""
import json
import os
from pathlib import Path
import tempfile
import time


def reconcile_metrics(path, start_epoch):
    """Retain valid records preceding the resumed epoch, preserving damaged input."""
    path = Path(path)
    if not path.exists():
        return
    original = path.read_bytes()
    retained = []
    damaged = []
    for number, line in enumerate(original.decode('utf-8', errors='replace').splitlines(), 1):
        if not line.strip():
            damaged.append(number)
            continue
        try:
            record = json.loads(line)
            epoch = record.get('epoch') if isinstance(record, dict) else None
            if type(epoch) is not int or epoch < 0:
                raise ValueError('Invalid epoch')
        except (ValueError, TypeError):
            damaged.append(number)
            continue
        if epoch < start_epoch:
            retained.append(line)
    if damaged:
        backup = path.with_name(f'{path.name}.before_resume_{time.time_ns()}')
        backup.write_bytes(original)
        print(f'RESUME metrics: skipped {len(damaged)} blank or invalid records; '
              f'original preserved at {backup}', flush=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write('\n'.join(retained) + ('\n' if retained else ''))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
