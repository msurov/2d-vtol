import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt


def solve_ivp_mat(rhs, tspan, X0, **kwargs):
    mshape = np.shape(X0)
    x0 = np.reshape(X0, (-1,))
    vshape = np.shape(x0)

    def vec_rhs(t, x):
        X = np.reshape(x, mshape)
        dX = rhs(t, X)
        dx = np.reshape(dX, vshape)
        return dx
    
    sol = solve_ivp(vec_rhs, tspan, x0, **kwargs)
    x = sol['y'].T
    X = np.reshape(x, (-1,) + mshape)
    t = sol['t']
    return t, X

def tangent(c, t):
    v = c(t, 1)
    return v / np.linalg.norm(v)

def curvature(c, t):
    '''
        d2gamma/ds2
    '''
    c1 = c(t, 1)
    c2 = c(t, 2)
    c1_sq = np.dot(c1, c1)
    return c2 / c1_sq - c1 * np.dot(c1, c2) / c1_sq**2

def exter(a, b):
    ab = np.outer(a,b)
    return ab - ab.T

def get_perp_vectors(v):
    v = np.reshape(v, (-1,1))
    U,l,Vt = np.linalg.svd(v.T)
    perp = Vt[1:,:] / l[0]
    return perp.T

def get_vec_basis(v):
    v = np.reshape(v, (-1,1))
    U,l,Vt = np.linalg.svd(v.T)
    d = len(v)
    basis = Vt
    if np.linalg.det(basis) < 0:
        basis[:,-1] = -basis[:,-1]
    return basis

def construct_periodic_basis(curve):

    def A(t):
        cur = curvature(curve, t)
        d = curve(t, 1)
        A = exter(cur, d)
        return A

    def rhs(t,E):
        return A(t).dot(E)

    t1 = curve.t[0]
    t2 = curve.t[-1]
    R0 = get_vec_basis(curve(t1, 1))
    t, R = solve_ivp_mat(rhs, [t1, t2], R0, max_step=5e-3)
    d = len(curve(t1, 1))

    for i in range(len(R)):
        tan = tangent(curve, t[i])
        tmp = tan.T @ R[i]
        assert np.allclose(tmp[1:], 0)
        assert np.allclose(tmp[0], 1)
        I = R[i] @ R[i].T
        assert np.allclose(I, np.eye(d))


def main():
    traj = np.load('data/traj.npy', allow_pickle=True).item()
    x = np.concatenate((traj['q'], traj['dq']), axis=1)
    t = traj['t']
    sp = make_interp_spline(t, x, k=5, bc_type='periodic')
    construct_periodic_basis(sp)

    # tt = np.linspace(-1, 10, 1000)
    # plt.plot(tt, sp(tt))
    # plt.show()

    # make_interp_spline


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    main()
