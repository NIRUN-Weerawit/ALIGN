"""State-coordinate parity near the Panda's pi rotation, not just pose equivalence."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from eval.libero_state import quat_to_axisangle


def test_dataset_branch_roundtrip_above_pi_and_quaternion_sign_invariance():
    target=np.array([3.140781164,.001675288,-.08809948])
    assert np.linalg.norm(target)>np.pi
    quat=Rotation.from_rotvec(target).as_quat()
    for q in [quat,-quat,quat*2]:np.testing.assert_allclose(quat_to_axisangle(q),target,atol=2e-7)
    shortest=quat_to_axisangle(quat,'shortest')
    assert shortest[0]<0 and np.linalg.norm(shortest-target)>6
    assert (Rotation.from_rotvec(shortest).inv()*Rotation.from_rotvec(target)).magnitude()<2e-7


def test_libero_input_coordinates_remain_continuous_across_pi():
    a=quat_to_axisangle(Rotation.from_rotvec([np.pi-.001,0,0]).as_quat())
    z=quat_to_axisangle(Rotation.from_rotvec([np.pi+.001,0,0]).as_quat())
    assert a[0]>0 and z[0]>0
    np.testing.assert_allclose(z-a,[.002,0,0],atol=2e-7)


def test_identity_and_input_are_not_mutated():
    q=np.array([0.,0.,0.,1.]);before=q.copy()
    assert np.array_equal(quat_to_axisangle(q),np.zeros(3))
    assert np.array_equal(q,before)


@pytest.mark.parametrize('bad',[np.zeros(4),np.ones(3),np.array([0.,0.,np.nan,1.])])
def test_invalid_quaternions_do_not_silently_create_zero_orientation(bad):
    with pytest.raises(ValueError,match='quaternion'):quat_to_axisangle(bad)


def test_sync_async_state_coordinates_match_the_same_branch():
    from eval.eval_libero_v4_trajectory import get_sim_eef_pose
    from eval.eval_libero_v4_async import get_sim_eef_pose as async_pose
    q=Rotation.from_rotvec([3.2,0,0]).as_quat()
    obs=dict(robot0_eef_pos=np.array([.1,.2,.3]),robot0_eef_quat=q)
    np.testing.assert_allclose(get_sim_eef_pose(obs),async_pose(obs))
    np.testing.assert_allclose(get_sim_eef_pose(obs)[3:],[3.2,0,0],atol=2e-7)
