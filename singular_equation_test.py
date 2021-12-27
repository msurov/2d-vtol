from construct_singular_trajectory import solve_singular,ReducedDynamics,join_several
from casadi import pi,arctan,vertcat,MX,substitute,Function
from dynamics import Dynamics, get_inv_dynamics, Parameters
from plots import plot_reduced_trajectory, configure
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc

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

def p1():
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

def p2():
    pass

p1()
plt.show()
