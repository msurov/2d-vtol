from construct_singular_trajectory import solve_singular,ReducedDynamics,join_several
from casadi import pi,arctan,vertcat,MX,substitute,Function
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
    plt.show()


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

if __name__ == '__main__':
    # trajectories_various_b()
    singular_phase_portrait()
    plt.show()
