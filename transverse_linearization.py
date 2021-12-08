from casadi import SX, DM, MX, vertcat, horzcat, sin, cos, \
    simplify, substitute, pi, jacobian, nlpsol, Function, \
    pinv, interpolant, Callback, Sparsity, jtimes
import numpy as np
from dynamics import Dynamics, parameters, get_inv_dynamics
from scipy.interpolate import make_interp_spline, splrep, splprep, splev, BSpline
from construct_basis import construct_basis
from bsplinesx import bsplinesx, bsplinesf
from time import time


def get_trans_lin(dynamics : Dynamics, trajectory : dict):
    t = trajectory['t']
    x = np.concatenate((trajectory['q'], trajectory['dq']), axis=1)
    u = trajectory['u']
    _,nx = x.shape
    _,nu = u.shape

    periodic = np.allclose(x[0], x[-1])
    xsp = make_interp_spline(t, x, k=5, bc_type='periodic' if periodic else None)
    xsf = bsplinesf(xsp)

    usp = make_interp_spline(t, u, k=3, bc_type='periodic' if periodic else None)
    usf = bsplinesf(usp)

    t,E = construct_basis(t, x, periodic)
    Esp = make_interp_spline(t, E, k=3, bc_type='periodic' if periodic else None)
    Esf = bsplinesf(Esp)

    tau = MX.sym('tau')
    xi = MX.sym('tau', nx-1)
    x = MX.sym('x', nx)
    u = MX.sym('u', nu)
    f = dynamics.rhs(x[0:nx//2], x[nx//2:nx], u)
    alpha = xsf(tau) + Esf(tau) @ xi
    beta = Esf(tau).T @ (x - xsf(tau))

    J = substitute(jacobian(alpha, xi), xi, 0)
    tmp = jtimes(f, x, J)
    tmp = substitute(tmp, x, xsf(tau))
    Ax = substitute(tmp, u, usf(tau))

    tmp = jacobian(f, u)
    tmp = substitute(tmp, x, xsf(tau))
    Bx = substitute(tmp, u, usf(tau))

    dxsf = jacobian(xsf(tau), tau)
    tmp = jtimes(beta, x, dxsf) + jacobian(beta, tau)
    tmp1 = jtimes(tmp, x, J)
    tmp2 = jtimes(beta, x, Ax)
    tmp = substitute(tmp1 + tmp2, x, xsf(tau))
    tmp = substitute(tmp, u, usf(tau))
    Axi = Function('A', [tau], [tmp])

    tmp = jtimes(beta, x, Bx)
    tmp = substitute(tmp, x, xsf(tau))
    Bxi = Function('B', [tau], [tmp])

    Jsf = Function('J', [tau], [J])

    t = trajectory['t']
    A = np.zeros((len(t), nx-1, nx-1))
    B = np.zeros((len(t), nx-1, nu))
    J = np.zeros((len(t), nx, nx-1))

    for i,w in enumerate(t):
        A[i] = np.array(Axi(w))
        B[i] = np.array(Bxi(w))
        J[i] = np.array(Jsf(w))
    
    if periodic:
        assert np.allclose(A[-1], A[0])
        assert np.allclose(B[-1], B[0])
        assert np.allclose(J[-1], J[0])
        A[-1] = A[0]
        B[-1] = B[0]
        J[-1] = J[0]

    return {
        't': t,
        'A': A,
        'B': B,
        'J': J,
        'E': E
    }

def save_linsys(linsys, dstfile):
    np.save(dstfile, linsys, allow_pickle=True)

def load_linsys(linsysfile):
    return np.load(linsysfile, allow_pickle=True).item()

def main(dynamics : Dynamics, trajectory : dict, linsysfile : str):
    linsys = get_trans_lin(dynamics, trajectory)
    save_linsys(linsys, linsysfile)

if __name__ == '__main__':
    from find_singular_trajectory import load_trajectory

    trajfile = 'data/traj.npy'
    t = load_trajectory(trajfile)
    d = Dynamics(parameters)
    linsys = get_trans_lin(d, t)
    save_linsys(linsys, 'data/linsys.npy')

