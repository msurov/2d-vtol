import numpy as np
import find_singular_trajectory
import transverse_linearization
import matplotlib.pyplot as plt
from dynamics import Dynamics, parameters, get_inv_dynamics
from lqr import lqr_ltv
import linsys_feedback
from transverse_feedback import TransversePeriodicFeedback
from simulator import Simulator
from find_singular_trajectory import plot_trajectory


def run_simulation(dynamics, trajectory, feedback):
    def F(x, u):
        dx = dynamics.rhs(x[0:3], x[3:6], u)
        return np.reshape(dx, (-1,))

    def stopcnd():
        if np.linalg.norm(tfb.xi) > 0.1:
            return True

    sim = Simulator(F, feedback, 1e-2)
    np.random.seed(0)
    x0 = trajectory['x'][0]
    x0 += 0.2 * np.random.normal(size=x0.shape)
    t,x,u,s = sim.run(x0, 0, 20)
    tau,xi = zip(*s)
    tau = np.concatenate([[tau[0]], tau])
    xi = np.concatenate([[xi[0]], xi])
    real_traj = {
        't': tau,
        'q': x[:,0:3],
        'dq': x[:,3:6],
        'x': x,
        'u': u
    }
    plt.figure('Simulation results', figsize=(12, 8))
    plot_trajectory(trajectory, ls='--', lw=2)
    plot_trajectory(real_traj, alpha=0.5)

    plt.figure('Transverse coordinates', figsize=(12, 8))

    plt.subplot(121)
    plt.plot(t, xi)
    plt.grid(True)

    plt.subplot(122)
    plt.plot(t, tau)
    plt.grid(True)
    plt.show()


def main():
    trajfile = 'data/traj.npy'
    linsysfile = 'data/linsys.npy'
    feedbackfile = 'data/fb.npy'

    dynamics = Dynamics(parameters)

    # find_singular_trajectory.main(dynamics, trajfile)
    traj = find_singular_trajectory.load_trajectory(trajfile)
    # transverse_linearization.main(dynamics, traj, linsysfile)
    linsys = transverse_linearization.load_linsys(linsysfile)
    # linsys_feedback.main(traj, linsys, feedbackfile)
    fb = linsys_feedback.load_feedback(feedbackfile)
    tfb = TransversePeriodicFeedback(traj, fb, linsys)

    run_simulation(dynamics, traj, tfb)

main()
