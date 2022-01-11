from casadi import MX, DM, vertcat, horzcat, sin, cos, \
    simplify, substitute, pi, jacobian, nlpsol, Function, pinv, evalf
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline, splrep
from dynamics import Dynamics, get_inv_dynamics, parameters
from trajectory import save_trajectory, load_trajectory


class ServoConnectionParametrized:
    def __init__(self):
        k = MX.sym('k', 5) # parameters of the servo-connection
        theta = MX.sym('theta')
        Q = vertcat(
            theta,
            k[0] * theta + k[1] * theta**2,
            k[2] + k[3] * theta + k[4] * theta**2
        )
        self.theta = theta
        self.parameters = k
        self.parameters_min = [-1, -2, 0, -2, -2]
        self.parameters_max = [1, 2, np.pi, 2, 2]
        self.Q = Function('Q', [theta], [Q])
    
    def subs(self, parameters):
        arg = MX.sym('dummy')
        Q = substitute(self.Q(arg), self.parameters, parameters)
        return Function('Q', [arg], [Q])
    
    def __call__(self, arg):
        return self.Q(arg)

class ReducedDynamics:
    def __init__(self, dynamics, connection):
        theta = MX.sym('theta')
        Q = connection(theta)
        dQ = jacobian(Q, theta)
        ddQ = jacobian(dQ, theta)

        alpha = dynamics.B_perp(Q) @ dynamics.M(Q) @ dQ
        beta = dynamics.B_perp(Q) @ (dynamics.M(Q) @ ddQ + dynamics.C(Q, dQ) @ dQ)
        gamma = dynamics.B_perp(Q) @ dynamics.G(Q)

        self.theta = theta
        self.alpha = Function('alpha', [theta], [alpha])
        self.beta = Function('beta', [theta], [beta])
        self.gamma = Function('gamma', [theta], [gamma])

def find_singular_connection(theta_s, theta_l, theta_r, dynamics, parametrized_connection):
    '''
        @brief Find values of parameters of `parametrized_connection` 
        which give the reduced dynamics a smooth trajectory
    '''
    rd = ReducedDynamics(dynamics, parametrized_connection)

    theta_s = MX(theta_s)
    d_alpha = rd.alpha.jac()
    d_alpha_s = d_alpha(theta_s, 0)
    alpha_s = rd.alpha(theta_s)
    beta_s = rd.beta(theta_s)

    smoothness = 3
    npts = 8

    constraints = [
        alpha_s,
        -alpha_s,
        d_alpha_s - 0.1,
        -smoothness/2 * d_alpha_s - beta_s
    ]
    pts = np.concatenate((
        np.linspace(theta_l, float(theta_s), npts)[:-1],
        np.linspace(float(theta_s), theta_r, npts)[1:]
    ))
    for p in pts:
        p = MX(p)
        constraints += [
            rd.gamma(p) - 1e-3,
            rd.alpha(p) * p - 1e-3
        ]

    constraints = vertcat(*constraints)
    nlp = {
        'x': parametrized_connection.parameters,
        'f': 0,
        'g': constraints
    }
    S = nlpsol('S', 'ipopt', nlp)
    sol = S(
        x0 = np.zeros(parametrized_connection.parameters.shape), 
        lbg=0,
        lbx=parametrized_connection.parameters_min,
        ubx=parametrized_connection.parameters_max
    )
    assert np.all(sol['g'] > -1e-8), 'Solution was not found'

    k_found = sol['x']
    print(f'parameters found {k_found}')
    return parametrized_connection.subs(k_found)


def solve_singular(rd, theta_s, theta0, step=1e-3):
    '''
        @brief Find trajectory of reduced singular dynamics

        `theta_s` is s.t. alpha(theta_s) = 0
        `theta_0` is the initial point of the trajectory
    '''
    theta = MX.sym('theta')
    y = MX.sym('y')
    alpha = rd.alpha(theta)
    beta = rd.beta(theta)
    gamma = rd.gamma(theta)
    dy = (-2 * beta * y - gamma) / alpha
    rhs = Function('rhs', [theta, y], [dy])
    dy_s_expr = (jacobian(beta, theta) * gamma - beta * jacobian(gamma, theta)) / \
        (jacobian(alpha, theta) * beta + 2 * beta**2)

    # value at singular point
    y_s = float(-rd.gamma(theta_s) / (2 * rd.beta(theta_s)))
    dtheta_s = np.sqrt(2 * y_s)
    rhs_s = jacobian(-gamma/(2*beta), theta)
    dy_s = float(evalf(substitute(dy_s_expr, theta, theta_s)))
    ddtheta_s = float(evalf(dy_s))

    # integrate left and right half-trajectories
    if theta0 < theta_s:
        sol = solve_ivp(rhs, [theta0, theta_s - step], [0], max_step=step)
    elif theta0 > theta_s:
        sol = solve_ivp(rhs, [theta0, theta_s + step], [0], max_step=step)
    else:
        assert False

    theta = np.concatenate((sol['t'], [theta_s]))
    y = np.concatenate((sol['y'][0], [y_s]))
    dy = np.reshape(rhs(sol['t'], sol['y'][0]), (-1,))
    dy = np.concatenate((dy, [dy_s]))
    dtheta = np.sqrt(2 * y)
    ddtheta = np.reshape(dy, (-1))

    # forward and backward motions
    theta = np.concatenate((theta[:0:-1], theta))
    dtheta = np.concatenate((-dtheta[:0:-1], dtheta))
    ddtheta = np.concatenate((ddtheta[:0:-1], ddtheta))
    if theta0 > theta_s:
        dtheta = -dtheta

    # find time
    h = 2 * np.diff(theta) / (dtheta[1:] + dtheta[:-1])
    t = np.concatenate(([0], np.cumsum(h)))
    sp = make_interp_spline(
        t, theta, k=5,
        bc_type=([(1, dtheta[0]), (2, ddtheta[0])], [(1, dtheta[-1]), (2, ddtheta[-1])])
    )
    ts = t[-1]

    # evaluate at uniform time-grid
    timestep = ts / 500
    npts = int((t[-1] - t[0]) / timestep + 1.5)
    tt = np.linspace(t[0], t[-1], npts)

    rd_traj = {
        't': tt,
        'theta': sp(tt),
        'dtheta': sp(tt, 1),
        'ddtheta': sp(tt, 2),
        'theta_s': theta_s,
        'dtheta_s': dtheta_s,
        'ddtheta_s': ddtheta_s,
        't_s': np.array([0, ts])
    }

    return rd_traj

def get_trajectory(dynamics, constraint, reduced_trajectory):
    R'''
        Get phase trajectory and reference control corresponding the 
        trajectory `reduced_trajectory` of the reduced dynamics
    '''
    theta = MX.sym('theta')
    dtheta = MX.sym('dtheta')
    ddtheta = MX.sym('ddtheta')
    Q = constraint(theta)
    dQ = jacobian(Q, theta)
    ddQ = jacobian(dQ, theta)
    dq_fun = Function('dQ', [theta, dtheta], [dQ * dtheta])
    ddq_fun = Function('dQ', [theta, dtheta, ddtheta], [dQ * ddtheta + ddQ * dtheta**2])

    theta = DM(reduced_trajectory['theta']).T
    dtheta = DM(reduced_trajectory['dtheta']).T
    ddtheta = DM(reduced_trajectory['ddtheta']).T
    q = constraint(theta)
    dq = dq_fun(theta, dtheta)
    ddq = ddq_fun(theta, dtheta, ddtheta)

    u_fun = get_inv_dynamics(dynamics)
    u = u_fun(q, dq, ddq)
    x = np.concatenate([np.array(q).T, np.array(dq).T], axis=1)

    traj = {
        't': reduced_trajectory['t'],
        'x': x,
        'q': x[:,0:3],
        'dq': x[:,3:6],
        'ddq': np.array(ddq).T,
        'u': np.array(u).T,
    }

    if 'theta_s' in reduced_trajectory:
        theta_s = reduced_trajectory['theta_s']
        dtheta_s = reduced_trajectory['dtheta_s']
        ddtheta_s = reduced_trajectory['ddtheta_s']
        qs = constraint(theta_s)
        dqs = dq_fun(theta_s, dtheta_s)
        ddqs = ddq_fun(theta_s, dtheta_s, ddtheta_s)

        traj['q_s'] = np.reshape(qs, (-1,))
        traj['dq_s'] = np.reshape(dqs, (-1,))
        traj['ddq_s'] = np.reshape(ddqs, (-1,))
        traj['u_s'] = np.reshape(u_fun(qs, dqs, ddqs), (-1,))
        traj['t_s'] = reduced_trajectory['t_s']

    return traj

def join_trajectories(reduced_trajectory_1, reduced_trajectory_2):
    t1 = reduced_trajectory_1['t']
    theta1 = reduced_trajectory_1['theta']
    dtheta1 = reduced_trajectory_1['dtheta']
    ddtheta1 = reduced_trajectory_1['ddtheta']
    ts1 = reduced_trajectory_1['t_s']

    t2 = reduced_trajectory_2['t']
    theta2 = reduced_trajectory_2['theta']
    dtheta2 = reduced_trajectory_2['dtheta']
    ddtheta2 = reduced_trajectory_2['ddtheta']
    ts2 = reduced_trajectory_2['t_s']

    assert np.allclose(theta1[-1], theta2[0])
    assert np.allclose(dtheta1[-1], dtheta2[0])
    assert np.allclose(ddtheta1[-1], ddtheta2[0])
    theta = np.concatenate((theta1[:-1], theta2))
    dtheta = np.concatenate((dtheta1[:-1], dtheta2))
    ddtheta = np.concatenate((ddtheta1[:-1], ddtheta2))
    t = np.concatenate((t1[:-1], t1[-1] + t2))
    ts = np.concatenate((ts1, ts2 + t1[-1]))
    ts = np.unique(ts)

    traj = reduced_trajectory_1.copy()
    traj['t'] = t
    traj['theta'] = theta
    traj['dtheta'] = dtheta
    traj['ddtheta'] = ddtheta
    traj['t_s'] = ts
    return traj

def join_several(*args):
    if len(args) == 1:
        return args[0]
    return join_several(join_trajectories(args[0], args[1]), *args[2:])

def main(dynamics, dstfile):
    c = ServoConnectionParametrized()
    theta_s = 0.
    Q = find_singular_connection(theta_s, -1, 1, dynamics, c)
    rd = ReducedDynamics(dynamics, Q)
    tr1 = solve_singular(rd, theta_s, 0.7)
    tr4 = solve_singular(rd, theta_s, -0.45)
    tr3 = solve_singular(rd, theta_s, 0.5)
    tr2 = solve_singular(rd, theta_s, -0.6)
    rd_traj = join_several(tr1, tr2, tr3, tr4)
    traj = get_trajectory(dynamics, Q, rd_traj)
    save_trajectory(dstfile, traj)

if __name__ == '__main__':
    trajfile = 'data/traj.npy'
    dynamics = Dynamics(parameters)
    main(dynamics, trajfile)
