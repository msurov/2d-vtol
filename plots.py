import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from construct_singular_trajectory import load_trajectory
from dynamics import parameters


font = {
    'family': 'Latin Modern Math',
    'weight': 'normal',
    'size': 20,
}

def configure():
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'Latin Modern Math'
    rc('text', usetex=True)

def plot_trajectory_projections(traj : dict, **kwargs):
    x,z,phi = traj['q'].T
    dx,dz,dphi = traj['dq'].T
    t = traj['t']

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
    
    fig = plt.figure('Trajectory projections', figsize=(12,8))
    axes = fig.get_axes()
    if len(axes) == 0:
        _,axes = plt.subplots(2, 3, num='Trajectory projections')
        plt.sca(axes[0,0])
        plt.xlabel(R'$x$')
        plt.ylabel(R'$\dot{x}$')
        plt.grid(True)
    
        plt.sca(axes[0,1])
        plt.xlabel(R'$z$')
        plt.ylabel(R'$\dot{z}$')
        plt.grid(True)

        plt.sca(axes[0,2])
        plt.xlabel(R'$\phi$')
        plt.ylabel(R'$\dot{\phi}$')
        plt.grid(True)

        plt.sca(axes[1,0])
        plt.xlabel(R'$x$')
        plt.ylabel(R'$z$')
        plt.grid(True)
        
        plt.sca(axes[1,1])
        plt.xlabel(R'$x$')
        plt.ylabel(R'$\phi$')
        plt.grid(True)

        plt.sca(axes[1,1])
        plt.xlabel(R'$x$')
        plt.ylabel(R'$\phi$')
        plt.grid(True)

        plt.sca(axes[1,2])
        plt.xlabel(R'$t$')
        plt.ylabel(R'$u$')
        plt.grid(True)

        axes = fig.get_axes()

    plt.sca(axes[0])
    plt.plot(x, dx, **kwargs)
    if qs is not None:
        plt.plot(qs[0], dqs[0], 'o', color='green', alpha=0.5)
        plt.plot(qs[0], -dqs[0], 'o', color='green', alpha=0.5)

    plt.sca(axes[1])
    plt.plot(z, dz, **kwargs)
    if qs is not None:
        plt.plot(qs[1], dqs[1], 'o', color='green', alpha=0.5)
        plt.plot(qs[1], -dqs[1], 'o', color='green', alpha=0.5)

    plt.sca(axes[2])
    plt.plot(phi, dphi, **kwargs)
    if qs is not None:
        plt.plot(qs[2], dqs[2], 'o', color='green', alpha=0.5)
        plt.plot(qs[2], -dqs[2], 'o', color='green', alpha=0.5)

    plt.sca(axes[3])
    plt.plot(x, z, **kwargs)
    if qs is not None:
        plt.plot(qs[0], qs[1], 'o', color='green', alpha=0.5)

    plt.sca(axes[4])
    plt.plot(x, phi, **kwargs)
    if qs is not None:
        plt.plot(qs[0], qs[2], 'o', color='green', alpha=0.5)

    if u1 is not None:
        plt.sca(axes[5])
        axes[5].set_prop_cycle(None)
        plt.plot(t, u1, label=R'$u_1$', **kwargs)
        plt.plot(t, u2, label=R'$u_2$', **kwargs)
        if qs is not None:
            plt.plot(ts, [us[0]] * len(ts), 'o', color='green', alpha=0.5)
            plt.plot(ts, [us[1]] * len(ts), 'o', color='green', alpha=0.5)

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
            plt.plot([theta_s, theta_s], [dtheta_s, -dtheta_s], 'o', alpha=0.5)
    
    plt.xlabel(R'$\theta$')
    plt.ylabel(R'$\dot{\theta}$')
    plt.grid(True)

def get_subtrajectory(trajectory, t1, t2):
    mask = trajectory['t'] >= t1
    mask &= trajectory['t'] <= t2

    result = trajectory.copy()
    result['t'] = trajectory['t'][mask]
    result['state'] = trajectory['state'][mask]
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
    q = trajectory['q']
    dq = trajectory['dq']
    ddq = trajectory['ddq']
    u = trajectory['u']

    fig,axes = plt.subplots(2, 2, sharex=True, num='Singular trajectory of time', figsize=(12,8))

    plt.sca(axes[0,0])
    for w in ts: plt.axvline(w, color='gold', lw=2)
    lines = plt.plot(t, q)
    plt.grid(True)
    plt.legend(lines, [R'$x$', R'$z$', R'$\phi$'])

    plt.sca(axes[1,0])
    for w in ts: plt.axvline(w, color='gold', lw=2)
    lines = plt.plot(t, dq)
    plt.grid(True)
    plt.xlabel('t, sec', fontdict=font)
    plt.legend(lines, [R'$\dot x$', R'$\dot z$', R'$\dot\phi$'])

    plt.sca(axes[0,1])
    for w in ts: plt.axvline(w, color='gold', lw=2)
    lines = plt.plot(t, ddq)
    plt.grid(True)
    plt.legend(lines, [R'$\ddot x$', R'$\ddot z$', R'$\ddot\phi$'])

    plt.sca(axes[1,1])
    for w in ts: plt.axvline(w, color='gold', lw=2)
    lines = plt.plot(t, u)
    plt.grid(True)
    plt.xlabel('t, sec', fontdict=font)
    plt.legend(lines, [R'$u_1$', R'$u_2$'])

if __name__ == '__main__':
    # configure()
    tr = load_trajectory('data/traj.npy')

    ts = tr['t_s']
    for i in range(1, len(ts)):
        t = get_subtrajectory(tr, ts[i-1], ts[i])
        plot_trajectory_projections(t)

    plt.tight_layout()
    plt.savefig('fig/found_phase.pdf')

    plot_timed_trajectory(tr)
    plt.tight_layout()
    plt.savefig('fig/found_timed.pdf')

    plt.show()
