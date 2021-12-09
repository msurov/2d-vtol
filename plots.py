import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from construct_singular_trajectory import load_trajectory

matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'Times New Roman'
rc('text', usetex=True)


def plot_trajectory(traj : dict, **kwargs):
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

def plot_reduced_trajectory(reduced_trajectory : dict):
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

def get_subtrajectory(trajectory, t1, t2):
    mask = trajectory['t'] >= t1
    mask &= trajectory['t'] <= t2

    result = trajectory.copy()
    result['t'] = trajectory['t'][mask]
    result['x'] = trajectory['x'][mask]
    result['q'] = trajectory['q'][mask]
    result['dq'] = trajectory['dq'][mask]
    result['ddq'] = trajectory['ddq'][mask]
    result['u'] = trajectory['u'][mask]

    mask = trajectory['t_s'] <= t1
    mask &= trajectory['t_s'] <= t2
    result['t_s'] = trajectory['t_s'][mask]

    return result

def plot_timed_trajectory(trajectory):
    t = trajectory['t']
    ts = trajectory['t_s']
    x = trajectory['x']
    q = trajectory['q']
    dq = trajectory['dq']
    ddq = trajectory['ddq']
    u = trajectory['u']

    plt.figure('Singular Trajectory', figsize=(16,8))

    ax = plt.subplot(221)
    for w in ts: plt.axvline(w, color='gold', lw=2)
    plt.plot(t, q, label=[R'$x$', R'$z$', R'$\phi$'])
    plt.grid(True)
    plt.legend()
    plt.title('Generalized coordinates')

    plt.subplot(222, sharex=ax)
    for w in ts: plt.axvline(w, color='gold', lw=2)
    plt.plot(t, dq, label=[R'$\dot x$', R'$\dot z$', R'$\dot\phi$'])
    plt.grid(True)
    plt.legend()
    plt.title('Generalized velocities')

    plt.subplot(223, sharex=ax)
    for w in ts: plt.axvline(w, color='gold', lw=2)
    plt.plot(t, ddq, label=[R'$\ddot x$', R'$\ddot z$', R'$\ddot\phi$'])
    plt.grid(True)
    plt.xlabel('t, sec')
    plt.legend()
    plt.title('Generalized accelerations')

    plt.subplot(224, sharex=ax)
    for w in ts: plt.axvline(w, color='gold', lw=2)
    plt.plot(t, u, label=[R'$u$', R'$w$'])
    plt.grid(True)
    plt.xlabel('t, sec')
    plt.legend()
    plt.title('Control input')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    tr = load_trajectory('data/traj.npy')

    # plt.figure('VTOL Singular Trajectory', figsize=(18,8))
    # ts = tr['t_s']
    # for i in range(1, len(ts)):
    #     t = get_subtrajectory(tr, ts[i-1], ts[i])
    #     plot_trajectory(t)

    # plt.subplot(236)
    # plt.legend([R'$u_1$', R'$u_2$', R'singularity points'])
    # plt.show()

    plot_timed_trajectory(tr)
