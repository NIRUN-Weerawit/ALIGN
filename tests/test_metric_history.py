import json
from training.metric_history import reconcile_metrics


def test_resume_recovers_partial_records_and_preserves_original(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    original = b'{"epoch":1,"loss":0.5}\n\n{"epoch":2}\n{"epoch":3}\n{"epoch":\x00\n[]\n{"epoch":"4"}\n'
    path.write_bytes(original)
    reconcile_metrics(path, 3)
    assert [json.loads(line)['epoch'] for line in path.read_text().splitlines()] == [1, 2]
    backups = list(tmp_path.glob('metrics.jsonl.before_resume_*'))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    reconcile_metrics(path, 3)
    assert len(list(tmp_path.glob('metrics.jsonl.before_resume_*'))) == 1


def test_clean_history_and_absent_history(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    reconcile_metrics(path, 2)
    assert not path.exists()
    path.write_text('{"epoch":1}\n{"epoch":2}\n')
    reconcile_metrics(path, 2)
    assert path.read_text() == '{"epoch":1}\n'
    assert not list(tmp_path.glob('*.before_resume_*'))
