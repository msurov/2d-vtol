from casadi import SX, DM, vertcat, horzcat, sin, cos, \
    simplify, substitute, pi, jacobian, nlpsol, Function, pinv
import numpy as np
from dynamics import Dynamics, parameters, get_inv_dynamics
from scipy.interpolate import make_interp_spline, splrep
from construct_basis import construct_basis


def get_trans_lin(dynamics : Dynamics, trajectory : dict):
    t = trajectory['t']
    x = np.concatenate((
        trajectory['q'],
        trajectory['dq']
    ), axis=1)
    u = trajectory['u']
    periodic = np.allclose(x[0], x[-1])
    xsp = make_interp_spline(t, x, k=5, 
        bc_type='periodic' if periodic else None)
    t,E = construct_basis(xsp, periodic)
    Esp = make_interp_spline(t, E, k=3, bc_type='periodic' if periodic else None)
