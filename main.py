import numpy as np
from scipy.interpolate import make_interp_spline
from trajectory import save_trajectory, load_trajectory
from plots import plot_trajectory_projections
import transverse_linearization
import construct_singular_trajectory
import linsys_feedback
from dynamics import Dynamics, parameters
from transverse_feedback import TransversePeriodicFeedback
from simulator import Simulator
import matplotlib.pyplot as plt


def run_simulation(dynamics, trajectory, feedback, saveto):
    def F(x, u):
        dx = dynamics.rhs(x[0:3], x[3:6], u)
        return np.reshape(dx, (-1,))

    def stopcnd():
        if np.linalg.norm(tfb.xi) > 0.1:
            return True

    xsp = make_interp_spline(trajectory['t'], trajectory['x'], k=3, bc_type='periodic')
    usp = make_interp_spline(trajectory['t'], trajectory['u'], k=3, bc_type='periodic')

    sim = Simulator(F, feedback, 1e-3)
    np.random.seed(0)
    x0 = np.zeros(6)
    x0[0] = 1.2
    x0[1] = -0.5

    t,x,u,s = sim.run(x0, 0, 20)
    tau,xi = zip(*s)
    tau = np.concatenate([[tau[0]], tau])
    xi = np.concatenate([[xi[0]], xi])
    real_traj = {
        't': t,
        'tau': tau,
        'xi': xi,
        'q': x[:,0:3],
        'dq': x[:,3:6],
        'x': x,
        'u': u
    }
    save_trajectory(saveto, real_traj)

def main():
    trajfile = 'data/traj.npy'
    linsysfile = 'data/linsys.npy'
    feedbackfile = 'data/fb.npy'
    simfile = 'data/sim.npy'

    dynamics = Dynamics(parameters)
    construct_singular_trajectory.main(dynamics, trajfile)
    traj = load_trajectory(trajfile)
    transverse_linearization.main(dynamics, traj, linsysfile)
    linsys = transverse_linearization.load_linsys(linsysfile)
    linsys_feedback.main(traj, linsys, feedbackfile)
    fb = linsys_feedback.load_feedback(feedbackfile)
    tfb = TransversePeriodicFeedback(traj, fb, linsys)
    run_simulation(dynamics, traj, tfb, simfile)
    realtraj = load_trajectory(simfile)
    plot_trajectory_projections(traj, ls='--', lw=2)
    plot_trajectory_projections(realtraj, alpha=0.5)
    plt.show()

main()
