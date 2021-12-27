import numpy as np


def join_two_trajectories(tr1, tr2):

    t1 = tr1['t']
    x1 = tr1['x']
    ddq1 = tr1['ddq']
    u1 = tr1['u']

    t2 = tr2['t']
    x2 = tr2['x']
    ddq2 = tr2['ddq']
    u2 = tr2['u']

    t = np.concatenate((t1, t2 + t1[-1]))
    x = np.concatenate((x1, x2))
    ddq = np.concatenate((ddq1, ddq2))
    u = np.concatenate((u1, u2))

    return {
        't': t,
        'x': x,
        'q': x[:,0:3],
        'dq': x[:,3:6],
        'ddq': ddq,
        'u': u,
    }

def join_trajectories(*args):
    if len(args) == 1:
        return args[0]
    return join_trajectories(join_two_trajectories(args[0], args[1]), *args[2:])

def save_trajectory(dstfile, traj):
    np.save(dstfile, traj, allow_pickle=True)

def load_trajectory(trajfile):
    traj = np.load(trajfile, allow_pickle=True).item()
    return traj
