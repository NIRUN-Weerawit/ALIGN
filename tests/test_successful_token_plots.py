"""Evidence plots must distinguish uncalibrated scores, GT, and temporal support."""
import numpy as np
import pytest
from tools.plot_successful_task_tokens import first_stable,signal_metrics


def test_stability_separates_retrospective_start_from_when_the_window_is_observed():
    mask=[False,True,True,False,True,True,True,True]
    assert first_stable(mask,np.arange(8),3)==dict(start_step=4,confirmed_step=6)
    assert first_stable([True,True],np.arange(2),3) is None
    with pytest.raises(ValueError,match='contiguous'):
        first_stable([True,True],np.array([0,2]),2)


def test_strong_wrong_prediction_does_not_count_as_confident_evidence_for_recorded_task():
    scores=np.array([[.01,.98,.01],[.02,.97,.01],[.85,.1,.05],[.9,.05,.05]])
    similarities=np.array([[.1,.9,.2],[.1,.9,.2],[.8,.3,.4],[.7,.3,.4]])
    result=signal_metrics(scores,similarities,0,np.arange(4),2,.8)
    assert result['strong_wrong_fraction']==.5
    assert result['first_stable_strong_correct']==dict(start_step=2,confirmed_step=3)
    assert result['first_stable_closest']==dict(start_step=2,confirmed_step=3)
    assert np.all(result['geometry_gap'][:2]<0)
    assert np.all(result['geometry_gap'][2:]>0)
    assert np.all(result['true_score'][:2]<.03)
