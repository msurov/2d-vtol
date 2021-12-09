from casadi import MX, DM, vertcat, horzcat, sin, cos, \
    simplify, substitute, pi, jacobian, nlpsol, Function, pinv, evalf
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline, splrep
from dynamics import Dynamics, get_inv_dynamics, parameters


class ServoConnectionParametrized:
    def __init__(self):
        k = MX.sym('k', 4) # parameters of the servo-connection
        theta = MX.sym('theta')
        Q = vertcat(
            theta,
            k[0] * theta**2,
            k[1] + k[2] * theta + k[3] * theta**2
        )
        self.theta = theta
        self.parameters = k
        self.parameters_min = [-2, 0, -2, -2]
        self.parameters_max = [2, np.pi, 2, 2]
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
    threshold = 0.01
    npts = 5

    constraints = [
        alpha_s,
        -alpha_s,
        d_alpha_s - threshold,
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
    return parametrized_connection.subs(k_found)

def plot_reduced_trajectory(reduced_trajectory):
    '''
        @brief Plot a given phase trajectory of reduced dynamics
    '''
    plt.figure('Reduced Dynamics Trajectory')
    theta = reduced_trajectory['theta']
    dtheta = reduced_trajectory['dtheta']
    plt.plot(theta, dtheta, color='blue')

    if 'theta_s' in reduced_trajectory:
        theta_s = reduced_trajectory['theta_s']
        plt.axvline(theta_s, ls='--', color='green')
        if 'dtheta_s' in reduced_trajectory:
            dtheta_s = reduced_trajectory['dtheta_s']
            plt.plot([theta_s, theta_s], [dtheta_s, -dtheta_s], 'o')
    
    plt.xlabel(R'$\theta$')
    plt.ylabel(R'$\dot{\theta}$')
    plt.grid(True)
    
def solve_singular(rd, theta_s, theta0):
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
    step = 1e-3

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
    timestep = ts / 100
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

    t = reduced_trajectory['t']
    # ts = reduced_trajectory['t_s']
    # i = np.argmin((t - ts)**2)

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

def plot_trajectory(traj, **kwargs):
    x,z,phi = traj['q'].T
    dx,dz,dphi = traj['dq'].T
    t = traj['t']
    period = t[-1]

    if 'u' in traj:
        u1,u2 = traj['u'].T
    else:
        u1 = u2 = None

    if 'q_s' in traj:
        qs = traj['q_s']
        dqs = traj['dq_s']
        us = traj['u_s']
        ts = traj['t_s']
    else:
        qs = dqs = us = ts = None

    ax = plt.subplot(231)
    plt.plot(x, dx, **kwargs)
    if qs is not None:
        plt.plot(qs[0], dqs[0], 'o', color='green')
        plt.plot(qs[0], -dqs[0], 'o', color='green')
    plt.xlabel(R'$x$')
    plt.ylabel(R'$\dot{x}$')
    plt.grid(True)

    ax = plt.subplot(232)
    plt.plot(z, dz, **kwargs)
    if qs is not None:
        plt.plot(qs[1], dqs[1], 'o', color='green')
        plt.plot(qs[1], -dqs[1], 'o', color='green')
    plt.xlabel(R'$z$')
    plt.ylabel(R'$\dot{z}$')
    plt.grid(True)

    ax = plt.subplot(233)
    plt.plot(phi, dphi, **kwargs)
    if qs is not None:
        plt.plot(qs[2], dqs[2], 'o', color='green')
        plt.plot(qs[2], -dqs[2], 'o', color='green')
    plt.xlabel(R'$\phi$')
    plt.ylabel(R'$\dot{\phi}$')
    plt.grid(True)

    ax = plt.subplot(234)
    plt.plot(x, z, **kwargs)
    if qs is not None:
        plt.plot(qs[0], qs[1], 'o', color='green')
    plt.xlabel(R'$x$')
    plt.ylabel(R'$z$')
    plt.grid(True)

    ax = plt.subplot(235)
    plt.plot(x, phi, **kwargs)
    if qs is not None:
        plt.plot(qs[0], qs[2], 'o', color='green')
    plt.xlabel(R'$x$')
    plt.ylabel(R'$\phi$')
    plt.grid(True)

    if u1 is not None:
        ax = plt.subplot(236)
        plt.plot(t, u1, label=R'$u_1$', **kwargs)
        plt.plot(t, u2, label=R'$u_2$', **kwargs)
        if qs is not None:
            plt.plot(ts, [us[0]] * len(ts), 'o', color='green')
            plt.plot(ts, [us[1]] * len(ts), 'o', color='green')
        plt.xlabel(R'$t$')
        plt.ylabel(R'$u$')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()

def save_trajectory(dstfile, traj):
    np.save(dstfile, traj, allow_pickle=True)

def load_trajectory(trajfile):
    traj = np.load(trajfile, allow_pickle=True).item()
    return traj

def test_trajectory(dynamics : Dynamics, trajectory : dict):
    f = dynamics.rhs
    t = trajectory['t']
    q = trajectory['q']
    dq = trajectory['dq']
    u = trajectory['u']
    usp = make_interp_spline(t, u)
    x = np.concatenate((q, dq), axis=1)

    def rhs(t, x):
        q = x[0:3]
        dq = x[3:6]
        u = usp(t)
        ans = f(q,dq,u)
        return np.reshape(ans, (-1,))

    sol = solve_ivp(rhs, [t[0], t[-1]], x[0,:], t_eval=t)
    t = sol['t']
    x = sol['y'].T
    traj1 = {
        't': t,
        'q': x[:,0:3],
        'dq': x[:,3:6],
    }
    plt.figure('Compare trajectories')
    plot_trajectory(traj, ls='--', lw=2)
    plot_trajectory(traj1, alpha=0.8)
    plt.tight_layout()


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
    theta_l = -0.5
    theta_r = 0.5
    Q = find_singular_connection(theta_s, theta_l, theta_r, dynamics, c)
    rd = ReducedDynamics(dynamics, Q)
    tr1 = solve_singular(rd, theta_s, 0.7)
    tr2 = solve_singular(rd, theta_s, -0.6)
    tr3 = solve_singular(rd, theta_s, 0.5)
    tr4 = solve_singular(rd, theta_s, -0.45)
    rd_traj = join_several(tr1, tr2, tr3, tr4)
    traj = get_trajectory(dynamics, Q, rd_traj)
    save_trajectory(dstfile, traj)

if __name__ == '__main__':
    trajfile = 'data/traj.npy'
    dynamics = Dynamics(parameters)
    main(dynamics, trajfile)

    traj = load_trajectory(trajfile)
    test_trajectory(dynamics, traj)
    plt.show()
