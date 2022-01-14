from construct_singular_trajectory import solve_singular,ReducedDynamics,join_several,ServoConnectionParametrized
from casadi import pi,arctan,vertcat,MX,substitute,Function,rootfinder
from dynamics import Dynamics, get_inv_dynamics, Parameters
from plots import plot_reduced_trajectory, configure
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import numpy as np
from scipy.integrate import solve_ivp


matplotlib.rcParams['font.size'] = 14
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

font_small = {
    'family': 'Latin Modern Math',
    'weight': 'normal',
    'size': 18,
}

class ServoConnection:
    def __init__(self):
        b = MX.sym('b')
        theta = MX.sym('theta')
        Q = vertcat(
            theta,
            -b/2 * theta**2,
            pi/2 - arctan((1 + b) * theta)
        )
        self.theta = theta
        self.parameters = b
        self.parameters_min = [0]
        self.parameters_max = [5]
        self.Q = Function('Q', [theta], [Q])
    
    def subs(self, parameters):
        arg = MX.sym('dummy')
        Q = substitute(self.Q(arg), self.parameters, parameters)
        return Function('Q', [arg], [Q])
    
    def __call__(self, arg):
        return self.Q(arg)

def uphalf_trajectory(tr):
    theta = tr['theta']
    dtheta = tr['dtheta']
    ddtheta = tr['ddtheta']
    t = tr['t']
    n = len(theta)

    if dtheta[0] < 0:
        s = slice(n//2, None)
    else:
        s = slice(0, n//2 + 1)
    
    return {
        'theta': theta[s],
        'dtheta': dtheta[s],
        'ddtheta': ddtheta[s],
        't': t[s],
        't_s': []
    }


def trajectories_various_b():
    parameters = Parameters(epsilon = 0, gravity = 1)
    dynamics = Dynamics(parameters)

    theta_s = 0
    theta_l = -1
    theta_r = 1

    bvalues = [0.3, 0.6, 1.0, 1.8]
    trajectories = []

    for b in bvalues:
        c = ServoConnection()
        c = c.subs([b])
        rd = ReducedDynamics(dynamics, c)
        tr1 = solve_singular(rd, theta_s, theta_l)
        tr2 = solve_singular(rd, theta_s, theta_r)
        rd_traj = join_several(tr1, tr2)
        trajectories += [(b, rd_traj)]

    pars = [
        (440, -0.78, 1.4),
        (480, -0.7, 1.75),
        (520, 0.7, 1.75),
        (560, 0.78, 1.4),
    ]

    _,ax = plt.subplots(1, 1, num='phase_various_b', figsize=(4,4))

    for (b,tr),(i,antx,anty) in zip(trajectories, pars):
        theta = tr['theta']
        dtheta = tr['dtheta']
        plt.plot(theta, dtheta)
        plt.annotate(f'$b = {b}$', 
            xy=[theta[i], dtheta[i]],
            xytext=[antx, anty], 
            arrowprops=dict(facecolor='black', shrink=1, width=0.5, headlength=14, headwidth=8),
            bbox=dict(boxstyle="round", fc="w"),
            horizontalalignment='center',
            verticalalignment='center'
        )

    plt.xlabel(R'$\theta$', fontdict=font, labelpad=-10)
    plt.ylabel(R'$\dot \theta$', fontdict=font, labelpad=-10)
    plt.xticks([-1,-0.5,0,0.5,1], [-1,-0.5,'',0.5,1])
    plt.yticks([0,0.5,1,1.5,2], [0,0.5,'',1.5,2])
    plt.grid(True, ls='--')
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=1, direction='in', right='on')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-0.1, 2.1)
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.99, top=0.99)
    plt.savefig('fig/singular_phase.pdf')

    _,ax = plt.subplots(1, 1, num='timed_various_b', figsize=(4,4))

    for b,tr in trajectories:
        ddtheta = tr['ddtheta']
        t = tr['t']
        plt.plot(t, ddtheta, label=f'$b = {b}$')

    plt.xlabel(R'$t$', fontdict=font, labelpad=-10)
    plt.ylabel(R'$\ddot \theta$', fontdict=font, labelpad=-10)
    plt.yticks([-4,-2,0,2,4], [-4,-2,'',2,4])
    plt.ylim(-4.1, 4.1)
    plt.grid(True, ls='--')
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=1, direction='in', right='on')
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.99, top=0.99)
    plt.savefig('fig/singular_timed.pdf')


def get_phase_curve(rd, theta_s, theta0, dtheta0, step=1e-2):
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

    # value at singular point
    y_s = float(-rd.gamma(theta_s) / (2 * rd.beta(theta_s)))

    # integrate left and right half-trajectories
    if theta0 < theta_s:
        sol = solve_ivp(rhs, [theta0, theta_s - step], [dtheta0**2/2], max_step=step)
    elif theta0 > theta_s:
        sol = solve_ivp(rhs, [theta0, theta_s + step], [dtheta0**2/2], max_step=step)
    else:
        assert False

    theta = np.concatenate((sol['t'], [theta_s]))
    y = np.concatenate((sol['y'][0], [y_s]))
    dtheta = np.sqrt(2 * y)

    return theta, dtheta

def singular_phase_portrait():
    parameters = Parameters(epsilon = 0, gravity = 1)
    dynamics = Dynamics(parameters)
    c = ServoConnection()
    c = c.subs([1])
    rd = ReducedDynamics(dynamics, c)

    theta_s = 0
    trajectories = []
    _,ax = plt.subplots(1, 1, num='singular phase portrait', figsize=(6,6))

    w = 1.8

    # areas
    plt.fill_between([-w, w], [1, 1], [-1, -1], alpha=0.2, color='aqua')
    plt.fill_between([-w, w], [1, 1], [w, w], alpha=0.2, color='yellow')
    plt.fill_between([-w, w], [-1, -1], [-w, -w], alpha=0.2, color='yellow')

    # axes
    plt.axhline(0, color='black', lw=0.5)

    # left half-plane
    color = 'brown'
    alpha = 0.8
    lw = 1
    ls = ':'
    for theta0 in np.arange(-w, 0, 0.2):
        theta, dtheta = get_phase_curve(rd, theta_s, theta0, 0)
        plt.plot(theta, dtheta, color=color, alpha=alpha, lw=lw, ls=ls)
        plt.plot(theta, -dtheta, color=color, alpha=alpha, lw=lw, ls=ls)
    
    for dtheta0 in np.arange(0.5, w, 0.25):
        theta, dtheta = get_phase_curve(rd, theta_s, -w, dtheta0)
        plt.plot(theta, dtheta, color=color, alpha=alpha, lw=lw, ls=ls)
        plt.plot(theta, -dtheta, color=color, alpha=alpha, lw=lw, ls=ls)

    for theta0 in np.arange(-1.5, 0, 0.2):
        theta, dtheta = get_phase_curve(rd, theta_s, theta0, w)
        plt.plot(theta, dtheta, color=color, alpha=alpha, lw=lw, ls=ls)
        plt.plot(theta, -dtheta, color=color, alpha=alpha, lw=lw, ls=ls)

    # right half-plane
    for theta0 in np.arange(w, 0, -0.2):
        theta, dtheta = get_phase_curve(rd, theta_s, theta0, 0)
        plt.plot(theta, dtheta, color=color, alpha=alpha, lw=lw, ls=ls)
        plt.plot(theta, -dtheta, color=color, alpha=alpha, lw=lw, ls=ls)
    
    for dtheta0 in np.arange(0.5, w, 0.25):
        theta, dtheta = get_phase_curve(rd, theta_s, w, dtheta0)
        plt.plot(theta, dtheta, color=color, alpha=alpha, lw=lw, ls=ls)
        plt.plot(theta, -dtheta, color=color, alpha=alpha, lw=lw, ls=ls)

    for theta0 in np.arange(1.5, 0, -0.2):
        theta, dtheta = get_phase_curve(rd, theta_s, theta0, w)
        plt.plot(theta, dtheta, color=color, alpha=alpha, lw=lw, ls=ls)
        plt.plot(theta, -dtheta, color=color, alpha=alpha, lw=lw, ls=ls)

    # main trajectory
    color = 'green'
    theta0 = 1
    lw = 3
    theta, dtheta = get_phase_curve(rd, theta_s, theta0, 0)
    plt.plot(theta, dtheta, color=color, lw=lw, alpha=1)
    plt.plot(theta, -dtheta, color=color, lw=lw, alpha=1)
    theta0 = -1
    theta, dtheta = get_phase_curve(rd, theta_s, theta0, 0)
    plt.plot(theta, dtheta, color=color, lw=lw, alpha=1)
    plt.plot(theta, -dtheta, color=color, lw=lw, alpha=1)

    # rectangle
    r = Rectangle([-0.03,-0.96], 2*0.03, 2*0.96, fill=True, lw=1, 
        facecolor='white', edgecolor='black', joinstyle='round', alpha=1)
    r.set_zorder(100)
    ax.add_patch(r)
    r = Rectangle([-0.03,1.04], 2*0.03, 1, fill=True, lw=1, 
        facecolor='white', edgecolor='black', joinstyle='round', alpha=1)
    r.set_zorder(100)
    ax.add_patch(r)
    r = Rectangle([-0.03,-1.04], 2*0.03, -1, fill=True, lw=1, 
        facecolor='white', edgecolor='black', joinstyle='round', alpha=1)
    r.set_zorder(2)
    ax.add_patch(r)

    # labels and box
    plt.xlabel(R'$\theta$', fontdict=font, labelpad=-8)
    plt.ylabel(R'$\dot \theta$', fontdict=font, labelpad=-10)
    plt.xticks([-1,0,1], [-1,'',1])
    plt.yticks([-1,0,1], [-1,'',1])
    plt.grid(True, ls='--')
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=1, direction='in', right='on')
    plt.xlim(-w, w)
    plt.ylim(-w, w)
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.99, top=0.99)

    periodic_bar = mpatches.Patch(color='aqua', label='periodic')
    nonperiodic_bar = mpatches.Patch(color='yellow', label='non-periodic')
    forbidden_bar = mpatches.Patch(facecolor='white', edgecolor='black', label='no trajectories')
    plt.legend(handles=[periodic_bar,nonperiodic_bar,forbidden_bar], loc='lower right', )

    plt.savefig('fig/singular_field.pdf')


def concatenate_solutions_demo():
    parameters = Parameters(epsilon = 0.2, gravity = 1)
    dynamics = Dynamics(parameters)
    c = ServoConnectionParametrized()
    c = c.subs([-0.18, -0.40, 1.6, -1.3, -0.20])
    rd = ReducedDynamics(dynamics, c)

    theta_s_fun = rootfinder('singularity', 'newton', rd.alpha)
    theta_s = float(theta_s_fun(0.))
    print('theta_s', theta_s)

    y_s = float(-rd.gamma(theta_s) / (2 * rd.beta(theta_s)))
    dtheta_s = np.sqrt(2 * y_s)
    print('dtheta_s', dtheta_s)

    d_alpha = rd.alpha.jac()
    s = -2 * d_alpha(theta_s, 0) / rd.beta(theta_s)
    print('smothness', s)

    trajectories = []
    _,ax = plt.subplots(1, 1, num='singular_solution', figsize=(6, 4))

    l = -1
    r = 0.5

    ls = '-'
    alpha = 0.3
    lw = 1

    color = 'darkblue'
    for dtheta0 in np.arange(0.5, 2.1, 0.2):
        theta, dtheta = get_phase_curve(rd, theta_s, l, dtheta0)
        plt.plot(theta, dtheta, color=color, alpha=alpha, lw=lw, ls=ls)
    color = 'darkred'
    for dtheta0 in np.arange(0.5, 2.1, 0.2):
        theta, dtheta = get_phase_curve(rd, theta_s, r, dtheta0)
        plt.plot(theta, dtheta, color=color, alpha=alpha, lw=lw, ls=ls)

    # rectangles
    patch = Rectangle([theta_s-0.01, dtheta_s + 0.03], 2*0.01, 1, fill=True, lw=1, 
        facecolor='gray', edgecolor='black', joinstyle='round', alpha=1)
    patch.set_zorder(100)
    ax.add_patch(patch)

    patch = Rectangle([theta_s-0.01, dtheta_s - 0.03], 2*0.01, -1, fill=True, lw=1, 
        facecolor='gray', edgecolor='black', joinstyle='round', alpha=1)
    patch.set_zorder(100)
    ax.add_patch(patch)

    # main trajectory
    dtheta1 = 0.7
    dtheta2 = 1.5
    alpha = 0.8
    lw = 3
    ls = '-'
    n = 13
    theta, dtheta = get_phase_curve(rd, theta_s, l, dtheta1)
    plt.plot(theta[n:], dtheta[n:], lw=lw, ls=ls)
    p1 = np.array([theta[n], dtheta[n]])
    theta, dtheta = get_phase_curve(rd, theta_s, r, dtheta2)
    plt.plot(theta[n:], dtheta[n:], lw=lw, ls=ls)
    p2 = np.array([theta[n], dtheta[n]])

    plt.plot(*p1, 'o', color='black')
    plt.plot(*p2, 'o', color='black')

    plt.annotate(f'$p_1$',
        xy = p1 + np.array([-0.05, 0.07]),
        horizontalalignment='center',
        verticalalignment='center',
        font=font
    )

    plt.annotate(f'$p_2$',
        xy = p2 + np.array([0.03, 0.09]),
        horizontalalignment='center',
        verticalalignment='center',
        font=font
    )

    plt.annotate(f'forbidden',
        xy = [theta_s - 0.02, dtheta_s - 0.2],
        xytext = [theta_s - 0.4, dtheta_s - 0.3],
        arrowprops=dict(facecolor='black', shrink=1, width=0.5, headlength=14, headwidth=8),
        bbox=dict(boxstyle="round", fc="w"),
        horizontalalignment='center',
        verticalalignment='center'
    )

    plt.xticks([p1[0], theta_s, p2[0]], [R'$\theta_1$', R'$\theta_s$', R'$\theta_2$'], font=font)
    plt.yticks([p1[1], dtheta_s, p2[1]], [R'$\dot{\theta}_1$', R'$\dot{\theta}_s$', R'$\dot{\theta}_2$'], font=font)
    plt.grid(True, ls=':')
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=1, direction='in', right='on')
    plt.xlim(l, r)
    plt.ylim(0.7, 1.6)
    plt.subplots_adjust(left=0.10, bottom=0.1, right=0.99, top=0.99)

    plt.savefig('fig/singular_solution.pdf')


def full_phase_portrait_demo():
    parameters = Parameters(epsilon = 0.2, gravity = 1)
    dynamics = Dynamics(parameters)
    c = ServoConnectionParametrized()
    c = c.subs([-0.18, -0.40, 1.6, -1.3, -0.20])
    rd = ReducedDynamics(dynamics, c)

    alpha_roots = rootfinder('singularity', 'newton', rd.alpha)
    theta_s = float(alpha_roots(0.))
    print('theta_s', theta_s)

    y_s = float(-rd.gamma(theta_s) / (2 * rd.beta(theta_s)))
    dtheta_s = np.sqrt(2 * y_s)
    print('dtheta_s', dtheta_s)

    gamma_roots = rootfinder('singularity', 'newton', rd.gamma)
    theta_1 = float(gamma_roots(-1.))
    print('theta_1', theta_1)
    theta_2 = float(gamma_roots(1.))
    print('theta_2', theta_2)

    d_alpha = rd.alpha.jac()
    s = -2 * d_alpha(theta_s, 0) / rd.beta(theta_s)
    print('smothness', s)

    l = -1
    r = 0.5
    ls = '-'
    alpha = 0.3
    lw = 1
    color = 'darkblue'

    _,ax = plt.subplots(1, 1, num='full_phase', figsize=(6, 4))

    # periodic
    color = 'darkblue'
    ls = '-'
    for theta0 in np.arange(theta_1 + 0.01, theta_s, 0.1):
        theta, dtheta = get_phase_curve(rd, theta_s, theta0, 0)
        plt.plot(theta, dtheta, color=color, alpha=alpha, lw=lw, ls=ls)
        plt.plot(theta, -dtheta, color=color, alpha=alpha, lw=lw, ls=ls)

    for theta0 in np.arange(theta_2 - 0.01, theta_s, -0.1):
        theta, dtheta = get_phase_curve(rd, theta_s, theta0, 0)
        plt.plot(theta, dtheta, color=color, alpha=alpha, lw=lw, ls=ls)
        plt.plot(theta, -dtheta, color=color, alpha=alpha, lw=lw, ls=ls)

    # non-periodic
    left = theta_1 - 0.1
    right = theta_2 + 0.1
    color = 'darkred'
    ls = '-'
    for p in np.linspace(0.035, 0.98, 30):
        dtheta0 = 3/2 * dtheta_s * np.tan(p * np.pi/2)
        theta, dtheta = get_phase_curve(rd, theta_s, left, dtheta0)
        plt.plot(theta, dtheta, color=color, alpha=alpha, lw=lw, ls=ls)
        plt.plot(theta, -dtheta, color=color, alpha=alpha, lw=lw, ls=ls)

    for p in np.linspace(0.04, 0.96, 25):
        dtheta0 = 3/2 * dtheta_s * np.tan(p * np.pi/2)
        theta, dtheta = get_phase_curve(rd, theta_s, right, dtheta0)
        plt.plot(theta, dtheta, color=color, alpha=alpha, lw=lw, ls=ls)
        plt.plot(theta, -dtheta, color=color, alpha=alpha, lw=lw, ls=ls)

    # homoclinics
    color = 'darkblue'
    ls = '-'
    lw = 1
    alpha = 1
    theta, dtheta = get_phase_curve(rd, theta_s, theta_1, 0.01)
    plt.plot(theta, dtheta, color=color, alpha=alpha, lw=lw, ls=ls)
    plt.plot(theta, -dtheta, color=color, alpha=alpha, lw=lw, ls=ls)

    theta, dtheta = get_phase_curve(rd, theta_s, theta_2, 0.01)
    plt.plot(theta, dtheta, color=color, alpha=alpha, lw=lw, ls=ls)
    plt.plot(theta, -dtheta, color=color, alpha=alpha, lw=lw, ls=ls)

    # rectangles
    patch = Rectangle([theta_s-0.01, dtheta_s + 0.05], 2*0.01, 1, fill=True, lw=1, 
        facecolor='gray', edgecolor='black', joinstyle='round', alpha=1)
    patch.set_zorder(100)
    ax.add_patch(patch)

    patch = Rectangle([theta_s-0.01, -dtheta_s - 0.05], 2*0.01, -1, fill=True, lw=1, 
        facecolor='gray', edgecolor='black', joinstyle='round', alpha=1)
    patch.set_zorder(100)
    ax.add_patch(patch)

    patch = Rectangle([theta_s-0.01, dtheta_s - 0.05], 2*0.01, -2 * (dtheta_s - 0.05), fill=True, lw=1, 
        facecolor='gray', edgecolor='black', joinstyle='round', alpha=1)
    patch.set_zorder(100)
    ax.add_patch(patch)

    # annotations

    plt.annotate(f'transition point', 
        xy=[theta_s + 0.05, dtheta_s + 0.05],
        xytext=[theta_s + 0.75, dtheta_s + 0.3], 
        arrowprops=dict(facecolor='black', shrink=1, width=0.5, headlength=14, headwidth=8),
        bbox=dict(boxstyle="round", fc="w"),
        horizontalalignment='center',
        verticalalignment='center',
        font=font_small
    )
    plt.annotate(f'saddle point', 
        xy=[theta_1, 0.07],
        xytext=[theta_1 + 0.5, 1.3], 
        arrowprops=dict(facecolor='black', shrink=1, width=0.5, headlength=14, headwidth=8),
        bbox=dict(boxstyle="round", fc="w"),
        horizontalalignment='center',
        verticalalignment='center',
        font=font_small
    )
    plt.annotate(f'forbidden set', 
        xy=[theta_s - 0.05, -dtheta_s - 0.3],
        xytext=[theta_s - 0.9, -dtheta_s - 0.3],
        arrowprops=dict(facecolor='black', shrink=1, width=0.5, headlength=14, headwidth=8),
        bbox=dict(boxstyle="round", fc="w"),
        horizontalalignment='center',
        verticalalignment='center',
        font=font_small
    )
    plt.annotate(f'forbidden set', 
        xy=[theta_s - 0.05, -dtheta_s + 0.5],
        xytext=[theta_s - 0.9, -dtheta_s - 0.3],
        arrowprops=dict(facecolor='black', shrink=1, width=0.5, headlength=14, headwidth=8),
        bbox=dict(boxstyle="round", fc="w"),
        horizontalalignment='center',
        verticalalignment='center',
        font=font_small
    )
    plt.axhline(0, color='black', alpha=0.3)
    plt.xticks([theta_1, theta_s, theta_2], [R'$\theta_1$', R'$\theta_s$', R'$\theta_2$'], font=font)
    plt.yticks([-dtheta_s, 0, dtheta_s], [R'$\dot{\theta}_s$', '0', R'$\dot{\theta}_s$'], font=font)
    plt.grid(True, ls='--')
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=1, direction='in', right='on')
    plt.xlim(left, right)
    plt.ylim(-3/2 * dtheta_s, 3/2 * dtheta_s)
    plt.subplots_adjust(left=0.10, bottom=0.1, right=0.99, top=0.99)

    plt.savefig('fig/full_phase.pdf')


if __name__ == '__main__':
    # trajectories_various_b()
    # singular_phase_portrait()
    # concatenate_solutions_demo()
    full_phase_portrait_demo()
    plt.show()
