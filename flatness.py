from dynamics import Dynamics, parameters
from scipy.interpolate import make_interp_spline
from scipy.integrate import solve_ivp
from plots import plot_trajectory
import numpy as np
import matplotlib.pyplot as plt


def get_traj(t, eps):
    rt = np.sqrt(1 + t**2)
    rt3 = rt**3
    x = t + eps / rt
    z = -t**2/2 - eps * t / rt
    theta = np.pi/2 - np.arctan(t)
    dx = 1 - eps * t / rt3
    dz = -t - eps*(1/rt - t**2/rt3)
    dtheta = -1/(rt**2)
    return np.array([x,z,theta,dx,dz,dtheta]).T


def test1():
    d = Dynamics(parameters)
    f = d.rhs
    eps = parameters.epsilon

    def u(t):
        u1 = eps * 1 / (t**2 + 1)**2
        u2 = 2*t / (t**2 + 1)**2
        return np.array([u1, u2])

    def rhs(t, x):
        q = x[0:3]
        dq = x[3:6]
        ans = f(q,dq,u(t))
        return np.reshape(ans, (-1,))

    y0 = get_traj(-2, eps)
    sol = solve_ivp(rhs, [-2, 2], y0, max_step=1e-3)
    tr1 = {
        't': sol['t'],
        'q': sol['y'][0:3].T,
        'dq': sol['y'][3:6].T,
    }
    plot_trajectory(tr1)

    x = get_traj(sol['t'], eps)
    tr2 = {
        't': sol['t'],
        'q': x[:,0:3],
        'dq': x[:,3:6]
    }
    plot_trajectory(tr2, ls='--')

    plt.show()


def test2():
    eps = parameters.epsilon
    t = np.linspace(-2, 2, 100)
    tr = get_traj(t, eps)
    # sp = make_interp_spline(t, tr)
    # plt.plot(t, tr[:,3:6], '--')
    # plt.plot(t, sp(t, 1)[:,0:3])
    # plt.show()
    x = tr[:,0]
    z = tr[:,1]
    theta = tr[:,2]
    # p = np.polyfit(x, z, 3)
    # plt.plot(x, z - np.polyval(p, x))

    p = np.polyfit(x, theta, 9)
    plt.plot(x, theta)
    plt.plot(x, np.polyval(p, x))
    plt.show()


# test1()
test2()