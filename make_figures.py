import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from plots import plot_trajectory_projections
from trajectory import save_trajectory, load_trajectory
from scipy.interpolate import make_interp_spline

matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'Latin Modern Math'
rc('text', usetex=True)

font = {
    'family': 'Latin Modern Math',
    'weight': 'normal',
    'size': 24,
}

def plot_segments(x, y, t, tsegments, **kwargs):
    tt = np.concatenate([[t[0]], tsegments, [t[-1]]])
    for i in range(1, len(tt)):
        t1 = tt[i-1]
        t2 = tt[i]
        b = (t >= t1) & (t < t2)
        plt.plot(x[b], y[b], **kwargs)

def plot_phase_projections():
    traj = load_trajectory('data/traj.npy')
    q = traj['q']
    u = traj['u']
    dq = traj['dq']
    qs = traj['q_s']
    dqs = traj['dq_s']
    us = traj['u_s']
    ts = traj['t_s']

    x,z,phi = q.T
    dx,dz,dphi = dq.T
    t = traj['t']
    xs,zs,phis = qs
    dxs,dzs,dphis = dqs

    # x-dx
    _,ax = plt.subplots(1, 1, num='reference trajectory: x-dx', figsize=(4,4))
    plot_segments(x, dx, t, ts, ls='-', lw=2)
    plt.plot([xs, xs], [dxs, -dxs], 'o', alpha=0.5, color='g')
    plt.grid(True, ls='--')
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=1, direction='in', right='on')
    plt.yticks([-1.0, -0.5, 0., 0.5, 1.0], [-1.0, -0.5, '', 0.5, 1.0])
    plt.xticks([-0.5, 0, 0.5], [-0.5, '', 0.5])
    plt.xlabel(R'$x$', fontdict=font, labelpad=-10)
    plt.ylabel(R'$\dot x$', fontdict=font, labelpad=-15)
    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.99, top=0.99)
    plt.savefig('fig/reference-x-dx.pdf')

    # z-dz
    _,ax = plt.subplots(1, 1, num='reference trajectory: z-dz', figsize=(4,4))
    plot_segments(z, dz, t, ts, ls='-', lw=2)
    plt.plot([zs, zs], [dzs, -dzs], 'o', alpha=0.5, color='g')
    plt.grid(True, ls='--')
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=1, direction='in', right='on')
    plt.yticks([-0.4, -0.2, 0., 0.2, 0.4], [-0.4, -0.2, '', 0.2, 0.4])
    plt.xticks([-0.3, -0.2, -0.1, 0.], [-0.3, '', -0.1, 0])
    plt.xlabel(R'$z$', fontdict=font, labelpad=-10)
    plt.ylabel(R'$\dot z$', fontdict=font, labelpad=-15)
    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.99, top=0.99)
    plt.savefig('fig/reference-z-dz.pdf')

    # phi-dphi
    _,ax = plt.subplots(1, 1, num='reference trajectory: phi-dphi', figsize=(4,4))
    plot_segments(phi, dphi, t, ts, ls='-', lw=2)
    plt.plot([phis, phis], [dphis, -dphis], 'o', alpha=0.5, color='g')
    plt.grid(True, ls='--')
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=1, direction='in', right='on')
    plt.yticks([-1.8, -1.2, -0.6, 0., 0.6, 1.2, 1.8], [-1.8, -1.2, -0.6, '', 0.6, 1.2, 1.8])
    plt.xticks([1*np.pi/4, 2*np.pi/4, 3*np.pi/4], [R'$\frac 1 4 \pi$', R'', R'$\frac 3 4\pi$'])
    plt.xlabel(R'$\phi$', fontdict=font, labelpad=-15)
    plt.ylabel(R'$\dot \phi$', fontdict=font, labelpad=-15)
    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.99, top=0.99)
    plt.savefig('fig/reference-phi-dphi.pdf')

    # t-u
    _,axes = plt.subplots(2, 1, num='reference trajectory: t-u', figsize=(4,4), sharex=True)
    ax = axes[0]
    plt.sca(ax)
    plot_segments(t, u[:,0], t, ts, ls='-', lw=2)
    for w in ts:
        plt.axvline(w, color='green', alpha=0.4, ls='--')
    plt.grid(True, ls='--')
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=1, direction='in', right='on')
    plt.yticks([-2, 0, 2], [-2, '', 2])
    plt.ylabel(R'$u_1$', fontdict=font, labelpad=0)

    ax = axes[1]
    plt.sca(ax)
    plot_segments(t, u[:,1], t, ts, ls='-', lw=2)
    for w in ts:
        plt.axvline(w, color='green', alpha=0.4, ls='--')
    plt.grid(True, ls='--')
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=1, direction='in', right='on')
    plt.xticks([0, 2, 4, 6, 8], [0, 2, '', 6, 8])
    plt.yticks([-2, 0, 2], [-2, '', 2])
    plt.xlabel(R'$t$', fontdict=font, labelpad=-5)
    plt.ylabel(R'$u_2$', fontdict=font, labelpad=0)
    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.99, top=0.99, hspace=0.)
    plt.savefig('fig/reference-t-u.pdf')


def plot_timed_transient():
    sim = load_trajectory('data/sim.npy')
    t = sim['t']
    b = t < 15
    t = t[b]
    tau = sim['tau'][b]
    usim = sim['u'][b]
    qsim = sim['q'][b]

    ref = load_trajectory('data/traj.npy')
    qsp = make_interp_spline(ref['t'], ref['q'], k=3, bc_type='periodic')
    usp = make_interp_spline(ref['t'], ref['u'], k=3, bc_type='periodic')
    qref = qsp(tau)
    uref = usp(tau)

    xref,zref,phiref = qref.T
    xsim,zsim,phisim = qsim.T
    period = ref['t'][-1]
    n = int(t[-1] / period + 1)
    ts = np.array(ref['t_s'])
    ts = np.concatenate([ts + period * i for i in range(n)])
    b = np.diff(ts) < 1e-5
    b = np.concatenate(([False], b))
    ts = ts[~b]
    b = (ts >= t[0]) & (ts <= t[-1])
    ts = ts[b]

    _,axes = plt.subplots(3, 1, num='transient phase timed', figsize=(4,6), sharex=True)
    # t-x
    ax = axes[0]
    plt.sca(ax)
    plt.plot(t, xref, '--', lw=2, color='red')
    plt.plot(t, xsim, lw=1.5, alpha=0.5, color='blue')
    for w in ts:
        plt.axvline(w, color='green', alpha=0.4, ls='--')
    plt.grid(True, ls='--')
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=1, direction='in', right='on')
    plt.ylabel(R'$x$', fontdict=font, labelpad=0)
    plt.yticks([-0.5, 0.0, 0.5, 1.0], ['', 0.0, '', 1.0])

    # t-z
    ax = axes[1]
    plt.sca(ax)
    plt.plot(t, zref, '--', lw=2, color='red')
    plt.plot(t, zsim, lw=1.5, alpha=0.5, color='blue')
    for w in ts:
        plt.axvline(w, color='green', alpha=0.4, ls='--')
    plt.grid(True, ls='--')
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=1, direction='in', right='on')
    plt.ylabel(R'$z$', fontdict=font, labelpad=-5)
    plt.yticks([-0.4, -0.2, 0.0], [-0.4, '', 0.0])

    # t-phi
    ax = axes[2]
    plt.sca(ax)
    plt.plot(t, phiref, '--', lw=2, color='red')
    plt.plot(t, phisim, lw=1.5, alpha=0.5, color='blue')
    for w in ts:
        plt.axvline(w, color='green', alpha=0.4, ls='--')
    plt.grid(True, ls='--')
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=1, direction='in', right='on')
    plt.ylabel(R'$\phi$', fontdict=font, labelpad=3)
    plt.yticks([0, 1*np.pi/4, 2*np.pi/4, 3*np.pi/4], [0, R'$\frac \pi 4$', R'$\frac \pi 2$', R'$\frac{3\pi}{4}$'])

    plt.xlabel(R'$t$', fontdict=font, labelpad=0)
    plt.subplots_adjust(left=0.16, bottom=0.10, right=0.99, top=0.99, hspace=0.05)

    plt.savefig('fig/transient-phase.pdf')

    _,axes = plt.subplots(2, 1, num='transient control timed', figsize=(4,6), sharex=True)
    # t-u1
    ax = axes[0]
    plt.sca(ax)
    plt.plot(t, uref[:,0], '--', lw=2, color='red')
    plt.plot(t, usim[:,0], lw=1.5, alpha=0.5, color='blue')
    for w in ts:
        plt.axvline(w, color='green', alpha=0.4, ls='--')
    plt.grid(True, ls='--')
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=1, direction='in', right='on')
    plt.ylabel(R'$u_1$', fontdict=font, labelpad=-3)
    plt.yticks([-2.0, 0., 2.], [-2.0, 0.0, 2.0])

    # t-u2
    ax = axes[1]
    plt.sca(ax)
    plt.plot(t, uref[:,1], '--', lw=2, color='red')
    plt.plot(t, usim[:,1], lw=1.5, alpha=0.5, color='blue')
    for w in ts:
        plt.axvline(w, color='green', alpha=0.4, ls='--')
    plt.grid(True, ls='--')
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=1, direction='in', right='on')
    plt.ylabel(R'$u_2$', fontdict=font, labelpad=-3)
    plt.yticks([-2.0, 0., 2.], [-2.0, 0.0, 2.0])
    plt.xlabel(R'$t$', fontdict=font, labelpad=0)
    plt.subplots_adjust(left=0.16, bottom=0.10, right=0.99, top=0.99, hspace=0.05)

    plt.savefig('fig/transient-control.pdf')

    t = sim['t']
    b = t < 15
    t = t[b]
    tau = sim['tau'][b]
    xi = sim['xi'][b]

    _,axes = plt.subplots(2, 1, sharex=True, num='xi', figsize=(4,6))
    ax = axes[0]
    plt.sca(ax)
    plt.plot(t, xi, lw=1)
    plt.grid(True, ls='--')
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=1, direction='in', right='on')
    plt.ylabel(R'$\xi$', fontdict=font, labelpad=-6)
    plt.yticks([-0.6, -0.3, 0., 0.3], [-0.6, -0.3, 0., 0.3])

    ax = axes[1]
    plt.sca(ax)
    plt.plot([t[0], t[-1]], [t[0], t[-1]], ls='--', color='gray', lw=1)
    plt.plot(t, tau, lw=1)
    plt.grid(True, ls='--')
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=1, direction='in', right='on')
    plt.ylabel(R'$\tau$', fontdict=font, labelpad=2)
    plt.xlabel(R'$t$', fontdict=font, labelpad=-6)
    plt.yticks([0,5,10,15], [0,5,10,15])
    plt.subplots_adjust(left=0.16, bottom=0.12, right=0.99, top=0.98, hspace=0.05)

    plt.savefig('fig/transient-transverse.pdf')


if __name__ == '__main__':
    plot_phase_projections()
    plot_timed_transient()
    plt.show()
