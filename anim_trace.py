from trace import Trace
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


class TraceScatter:
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

        self.line = plt.scatter(self.pts[:,0], self.pts[:,1], color=self.color, marker='d')

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


class TraceLine:
    def __init__(self, maxlen, color, **kwargs):
        self.maxlen = maxlen
        self.rgb = to_rgb(color)
        nsegments = maxlen - 1
        self.pts = np.zeros((nsegments, 2, 2))
        self.colors = np.zeros((nsegments, 4))
        self.colors[:,0] = self.rgb[0]
        self.colors[:,1] = self.rgb[1]
        self.colors[:,2] = self.rgb[2]
        self.colors[:,3] = np.linspace(1, 0, nsegments)
        self.line = LineCollection(self.pts, colors=self.colors, **kwargs)
        self.first = True

    def push_point(self, x, y):
        self.pts = np.roll(self.pts, axis=0, shift=1)
        self.pts[0,0,0] = x
        self.pts[0,0,1] = y
        if self.first:
            self.pts[0,1,0] = x
            self.pts[0,1,1] = y
            self.first = False
        else:
            self.pts[0,1,0] = self.pts[1,0,0]
            self.pts[0,1,1] = self.pts[1,0,1]
    
    def update(self, x, y):
        self.push_point(x, y)
        self.line.set_segments(self.pts)

    @property
    def patches(self):
        return [self.line]


if __name__ == '__main__':
    from matplotlib import animation

    fig = plt.figure(figsize=(19.20,10.80))
    plt.axis('equal')
    ax = plt.gca()
    ax.set_xlim(-4, 4)
    ax.set_ylim(-2, 2)

    trace = TraceLine(50, 'lightblue', lw=2)
    duration = 10
    fps = 60
    nframes = duration * fps
    ax.add_collection(trace.line)

    def init():
        return trace.patches

    def update(i):
        t = i / fps
        x = np.sin(2*np.pi*t)
        y = np.sin(2*2*np.pi*t)
        trace.update(x, y)
        return trace.patches

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=nframes, blit=True)
    plt.show()
