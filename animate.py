from draw_quadrotor import Quadrotor
import matplotlib.pyplot as plt
import numpy as np
from trajectory import load_trajectory
from matplotlib import animation
from scipy.interpolate import make_interp_spline
from matplotlib.colors import to_rgb


class Trace:
    def __init__(self, maxlen, color):
        self.maxlen = maxlen
        self.line = None
        self.rgb = to_rgb(color)
    
    def initialize(self, x, y):
        self.pts = np.zeros((self.maxlen, 2))
        self.pts[:,0] = x
        self.pts[:,1] = y

        self.color = np.zeros((self.maxlen, 4))
        self.color[:,0] = self.rgb[0]
        self.color[:,1] = self.rgb[1]
        self.color[:,2] = self.rgb[2]
        self.color[:,3] = np.linspace(1, 0, self.maxlen)

        self.line = plt.scatter(self.pts[:,0], self.pts[:,1], color=self.color, marker='o')

    def update(self, x, y):
        if self.line is None:
            self.initialize(x, y)
        else:
            self.pts = np.roll(self.pts, 1, 0)
            self.pts[0,0] = x
            self.pts[0,1] = y
            self.line.set_offsets(self.pts)

    @property
    def patches(self):
        if self.line is None:
            return []
        return [self.line]

def animate(traj, filepath=None, fps=60, animtime=None, speedup=None):
    t = traj['t']
    q = traj['q']
    qsp = make_interp_spline(t, q, k=1)

    x,z,phi = q.T
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(z)
    ymax = np.max(z)

    fig = plt.figure(figsize=(19.20,10.80))
    plt.axis('equal')
    ax = plt.gca()

    qrsz = 0.3

    ax.set_xlim([xmin - qrsz, xmax + qrsz])
    ax.set_ylim([ymin - qrsz, ymax + qrsz])

    if speedup is not None:
        assert speedup > 0
        nframes = int((t[-1] - t[0]) * fps / speedup)
    elif animtime is not None:
        assert animtime > 0
        nframes = int(animtime * fps)
        speedup = (t[-1] - t[0]) / animtime
    else:
        nframes = int((t[-1] - t[0]) * fps)
        speedup = 1
    
    qr = Quadrotor(ax, 'fig/drawing.svg')
    qr.set_default_transform(0, 0, 0, sx=0.02, sy=-0.02)

    trace = Trace(100, color='lightblue')

    def init():
        return qr.patches + trace.patches

    def update(i):
        ti = t[0] + i * speedup / fps
        q = qsp(ti)
        qr.move(*q)
        trace.update(q[0], q[1])
        return qr.patches + trace.patches

    plt.subplots_adjust(0.1, 0.1, 0.99, 0.99)
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=nframes, blit=True)
    if filepath is not None:
        if filepath.endswith('.gif'):
            writer='imagemagick'
        else:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=60, metadata=dict(artist='Maksim Surov'), bitrate=400*60)
        anim.save(filepath, writer)
    else:
        plt.show()

if __name__ == '__main__':
    traj = load_trajectory('data/sim.npy')
    # animate(traj, animtime=10)
    animate(traj, animtime=10, filepath='data/anim.mp4')
