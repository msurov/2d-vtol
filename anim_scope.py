import matplotlib.pyplot as plt
import numpy as np

class Scope:
    def __init__(self, ax, **kwargs):
        self.ax = ax
        self.line, = plt.plot([], [], **kwargs)
        self.x = []
        self.y = []
        
    def update(self, x, y):
        self.x += [x]
        self.y += [y]
        self.line.set_data(self.x, self.y)

    @property
    def patches(self):
        return [self.line]

if __name__ == '__main__':
    from matplotlib import animation

    fig = plt.figure(figsize=(5,3))
    ax = plt.gca()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.1, 1.1)
    xdiap = list(ax.get_xlim())

    scope = Scope(ax, lw=2)
    duration = 10
    fps = 10
    nframes = duration * fps

    def init():
        return scope.patches

    def update(i):
        t = i / fps
        scope.update(t, np.cos(t))
        if t > xdiap[0] + 0.9 * (xdiap[1] - xdiap[0]):
            d = xdiap[1] - xdiap[0]
            xdiap[0] = t - 0.9 * d
            xdiap[1] = xdiap[0] + d
            ax.set_xlim(*xdiap)
            ax.update()
        return scope.patches

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=nframes, blit=True)
    plt.show()
