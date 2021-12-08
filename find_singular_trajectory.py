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
    
def solve_singular(rd, theta_s, theta_l, theta_r):
    '''
        @brief Find periodic trajectory of reduced singular dynamics

        `theta_s` is s.t. alpha(theta_s) = 0
        `theta_l` is the starting point of the trajectory
        `theta_r` is the final point of the trajectory
    '''
    theta = MX.sym('theta')
    y = MX.sym('y')
    alpha = rd.alpha(theta)
    beta = rd.beta(theta)
    gamma = rd.gamma(theta)
    dy = (-2 * beta * y - gamma) / alpha
    rhs = Function('rhs', [theta, y], [dy])

    # integrate left and right half-trajectories
    step = 1e-3
    sol_left = solve_ivp(rhs, [theta_l, theta_s - step], [0], max_step=step)
    sol_right = solve_ivp(rhs, [theta_r, theta_s + step], [0], max_step=step)
    y_s = float(-rd.gamma(theta_s) / (2 * rd.beta(theta_s)))
    dtheta_s = np.sqrt(2 * y_s)
    rhs_s = jacobian(-gamma/(2*beta), theta)
    dy_s = substitute(rhs_s, theta, theta_s)
    ddtheta_s = float(evalf(dy_s))

    # concatenate left and right parts
    i = len(sol_left['t'])
    theta = np.concatenate((sol_left['t'], [theta_s], sol_right['t'][::-1]))
    y = np.concatenate((sol_left['y'][0], [y_s], sol_right['y'][0,::-1]))
    dy = rhs(theta, y)
    dy[i] = float(evalf(dy_s))

    # find time
    dtheta = np.sqrt(2 * y)
    ddtheta = np.reshape(dy, (-1))
    h = 2 * np.diff(theta) / (dtheta[1:] + dtheta[:-1])
    t = np.concatenate(([0], np.cumsum(h)))
    ts = t[i]

    # concatenate forard and backward motions
    theta = np.concatenate((theta, theta[-2::-1]))
    dtheta = np.concatenate((dtheta, -dtheta[-2::-1]))
    ddtheta = np.concatenate((ddtheta, ddtheta[-2::-1]))
    t = np.concatenate((t, 2*t[-1] - t[-2::-1]))
    period = t[-1]
    sp = make_interp_spline(t, theta, 5, bc_type='periodic')

    # evaluate at uniform time-grid
    timestep = ts/100
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
        't_s': ts
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
    ts = reduced_trajectory['t_s']
    i = np.argmin((t - ts)**2)

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
            plt.plot([ts, period-ts], [us[0], us[0]], 'o', color='green')
            plt.plot([ts, period-ts], [us[1], us[1]], 'o', color='green')
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
    plot_trajectory(traj, ls='--')
    plot_trajectory(traj1)
    plt.tight_layout()


def main(dynamics, dstfile):
    c = ServoConnectionParametrized()
    theta_s = 0.
    theta_l = -0.5
    theta_r = 0.5
    Q = find_singular_connection(theta_s, theta_l, theta_r, dynamics, c)
    rd = ReducedDynamics(dynamics, Q)
    rd_traj = solve_singular(rd, theta_s, theta_l, theta_r)

    plot_reduced_trajectory(rd_traj)
    plt.tight_layout()

    traj = get_trajectory(dynamics, Q, rd_traj)
    save_trajectory(dstfile, traj)


if __name__ == '__main__':
    trajfile = 'data/traj.npy'
    dynamics = Dynamics(parameters)
    main(dynamics, trajfile)

    traj = load_trajectory(trajfile)
    test_trajectory(dynamics, traj)
    plt.show()
