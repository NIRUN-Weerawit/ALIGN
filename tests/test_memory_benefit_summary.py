import pytest
from scripts.summarize_memory_benefit import summarize


def row(batch,t,arm,error,count=16):
    return dict(batch=batch,t=t,intervention=arm,bank_count=[count,count],
                position_mse=error,rotation_mse=error,gripper_mse=error)


def test_paired_clusters_exclude_empty_banks_and_preserve_error_sign():
    rows=[row(0,0,'baseline',100,0),row(0,0,'memory_shuffle',0,0)]
    for batch in range(3):
        for t in [30,60]:
            rows.extend([row(batch,t,'baseline',1),row(batch,t,'memory_shuffle',2)])
    result=summarize(rows,draws=100)['full_observation']
    assert result['batch_clusters']==3 and result['nonempty_anchors']==6
    metric=result['metrics']['position_mse']
    assert metric['benefit']==1 and metric['relative_benefit']==.5
    assert metric['bootstrap_95_percent_interval']==[1,1]


def test_incomplete_intervention_pairs_are_rejected():
    with pytest.raises(ValueError,match='Unpaired'):
        summarize([row(0,30,'baseline',1)])
