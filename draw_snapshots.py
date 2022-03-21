from draw_quadrotor import Quadrotor
import matplotlib.pyplot as plt
from copy import copy
from scipy.interpolate import make_interp_spline
import numpy as np
import casadi as ca
from construct_singular_trajectory import ReducedDynamics, solve_singular_2, rd_traj_reverse, \
        join_several, get_trajectory
from dynamics import Parameters, Dynamics
import matplotlib
from matplotlib import rc
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection


def draw_quadrotor_snapshots(ax, pts, sx=1, sy=1):
    qrs = []
    qr = Quadrotor(ax, 'fig/drawing.svg')
    qr.set_default_transform(0, 0, 0, sx, sy)
    qr.move(*pts[0])
    qrs += [qr]

    for p in pts[1:]:
        qr = copy(qr)
        qr.move(*p)
        qrs += [qr]

def draw_arcarrow(center, radius, facecolor='#2693de', edgecolor='#000000', theta1=-30, theta2=180):
        # Add the ring
    rwidth = 0.02
    ring = Wedge(center, radius, theta1, theta2, width=rwidth)
    # Triangle edges
    offset = 0.02
    xcent  = center[0] - radius + (rwidth/2)
    left   = [xcent - offset, center[1]]
    right  = [xcent + offset, center[1]]
    bottom = [(left[0]+right[0])/2., center[1]-0.05]
    arrow  = plt.Polygon([left, right, bottom, left])
    p = PatchCollection(
        [ring, arrow], 
        edgecolor = edgecolor, 
        facecolor = facecolor
    )
    return p

def trajectory_snapshot(traj, step):
    t = traj['t']
    q = traj['q']
    qsp = make_interp_spline(t, q, k=1)
    t1 = t[0]
    t2 = t[-1]
    n = int((t2 - t1) / step + 0.5)
    tt = np.linspace(t[0], t[-1], n)
    qq = qsp(tt)

    plt.figure('trajectory snapshots', figsize=(10, 5))
    plt.axis('equal')
    ax = plt.gca()
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.5, 0.4)

    draw_quadrotor_snapshots(ax, qq, sx=0.02, sy=-0.02)

    p = draw_arcarrow([0, -1], 0.6, theta1=60, theta2=120)
    ax.add_collection(p)

    # plt.plot(tt, qq[:,0], tt, qq[:,1])
    plt.subplots_adjust(left=0.07, bottom=0.07, right=0.99, top=0.99, hspace=0.05)
    plt.show()


def test_draw_quadrotor_snapshots():
    plt.figure('trajectory snapshots', figsize=(10, 3))
    plt.axis('equal')
    ax = plt.gca()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1, 2)
    draw_quadrotor_snapshots(
        ax,
        [
            [0, 0, 0],
            [2, -0.5, 0.4],
        ],
        sx = 0.1, sy = -0.1
    )
    plt.show()


def get_simple_trajectory():
    parameters = Parameters(epsilon=0, gravity=1)
    dynamics = Dynamics(parameters)
    theta = ca.MX.sym('theta')
    Q = ca.vertcat(
        theta, -0.5 * theta**2,
        np.pi/2 - 1.5 * theta
    )
    Q = ca.Function('Q', [theta], [Q])
    rd = ReducedDynamics(dynamics, Q)
    theta_s = 0
    tr1 = solve_singular_2(rd, theta_s, -0.7, 0.)
    tr2 = solve_singular_2(rd, theta_s, 0.7, 0.)
    tr2 = rd_traj_reverse(tr2)
    rd_traj = join_several(tr1, tr2)
    traj = get_trajectory(dynamics, Q, rd_traj)
    return traj


if __name__ == '__main__':
    # test_draw_quadrotor_snapshots()

    matplotlib.rcParams['font.size'] = 18
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'Latin Modern Math'
    rc('text', usetex=True)

    traj = get_simple_trajectory()
    trajectory_snapshot(traj, 0.45)
