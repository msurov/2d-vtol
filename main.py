import numpy as np
import find_singular_trajectory
import transverse_linearization
from dynamics import Dynamics, parameters, get_inv_dynamics

trajfile = 'data/traj.npy'
dynamics = Dynamics(parameters)
# find_singular_trajectory.main(dynamics, trajfile)
traj = find_singular_trajectory.load_trajectory(trajfile)
transverse_linearization.get_trans_lin(dynamics, traj)
