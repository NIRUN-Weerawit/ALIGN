"""LIBERO state representation shared by synchronous and async rollouts."""
import numpy as np


def quat_to_axisangle(quat: np.ndarray, convention: str = "libero") -> np.ndarray:
    """Match LIBERO's positive-X branch, which permits angles above pi.

    The Panda training poses keep the X axis positive near the downward
    gripper orientation. A shortest rotvec flips its axis when crossing pi,
    presenting an approximately 2*pi jump to the state encoder.
    """
    q=np.asarray(quat,dtype=np.float64).copy()
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-12:
        raise ValueError('Expected a finite nonzero xyzw quaternion')
    q/=np.linalg.norm(q)
    if convention=="shortest":
        if q[3]<0:q=-q
    elif convention=="libero":
        nonzero=np.flatnonzero(np.abs(q[:3])>1e-12)
        if len(nonzero) and q[nonzero[0]]<0:q=-q
    else:
        raise ValueError(f'Unknown rotation convention: {convention}')
    length=np.linalg.norm(q[:3])
    if length<1e-12:return np.zeros(3,dtype=np.float32)
    return (q[:3]*(2*np.arctan2(length,q[3])/length)).astype(np.float32)

