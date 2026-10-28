import numpy as np
from tools.summarize_task_token_counterfactuals import clustered_interval


def test_duplicate_evaluation_seeds_do_not_create_extra_independent_episodes():
    values=np.array([[1.,0.,-1.,0.,1.,0.,-1.,0.]])
    single=clustered_interval(values,np.ones_like(values))
    repeated=clustered_interval(np.repeat(values,3,axis=0),np.ones((3,8)))
    assert single['interval_95']==repeated['interval_95']
    assert repeated['episode_clusters']==8
    assert repeated['eligible_trials']==24


def test_effect_interval_respects_preintervention_cohort_masks():
    values=np.array([[-1.,1.,-1.,1.],[-1.,1.,-1.,1.]])
    mask=np.array([[1,0,1,0],[1,0,1,0]])
    result=clustered_interval(values,mask)
    assert result['mean']==-1
    assert result['interval_95']==[-1.,-1.]
    assert result['eligible_trials']==4


def test_zero_paired_effect_has_zero_interval():
    result=clustered_interval(np.zeros((3,44)),np.ones((3,44)))
    assert result['mean']==0
    assert result['interval_95']==[0.,0.]
