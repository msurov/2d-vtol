import numpy as np
from lqr import lqr_ltv, lqr_ltv_periodic


def linsys_periodic_feedback(traj : dict, linsys : dict):
    q = traj['q']
    dq = traj['dq']
    nx = q.shape[1] * 2
    u = traj['u']
    nu = u.shape[1]

    t = linsys['t']
    n = len(t)
    J = linsys['J']
    A = linsys['A']
    B = linsys['B']
    Qx = 5 * np.diag([1,10,1,1,1,1])
    R = np.zeros((n, nu, nu))
    Q = np.zeros((n, nx-1, nx-1))

    for i in range(len(t)):
        Q[i] = J[i].T @ Qx @ J[i]
        R[i] = np.eye(2)

    K,P = lqr_ltv_periodic(t, A, B, Q, R)
    assert np.allclose(K[0], K[-1])
    assert np.allclose(P[0], P[-1])
    K[-1] = K[0]
    P[-1] = P[0]
    sol = {
        't': t,
        'K': K,
        'P': P
    }
    return sol

def load_feedback(feedbackfile):
    return np.load(feedbackfile, allow_pickle=True).item()

def main(traj : dict, linsys : dict, feedbackfile : str):
    fb = linsys_periodic_feedback(traj, linsys)
    np.save(feedbackfile, fb, allow_pickle=True)
