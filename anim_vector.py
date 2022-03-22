import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, FancyArrowPatch


class Vector:
    def __init__(self, ax, **kwargs):
        self.arrow = FancyArrowPatch([0, 0], [0, 0], mutation_scale=20)
        ax.add_patch(self.arrow)

    def update(self, x, y, dx, dy):
        self.arrow.set_positions(
            np.array([x, y], float),
            np.array([x+dx, y+dy], float)
        )

    @property
    def patches(self):
        return [self.arrow]


if __name__ == '__main__':
    from matplotlib import animation

    fig = plt.figure(figsize=(5,3))
    ax = plt.gca()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.1, 1.1)
    xdiap = list(ax.get_xlim())

    vec1 = Vector(ax)
    vec2 = Vector(ax)

    duration = 10
    fps = 10
    nframes = duration * fps


    def init():
        return vec1.patches + vec2.patches

    def update(i):
        t = i / fps
        vec1.update(1, 0, np.sin(t), np.cos(t))
        vec2.update(0, 0, np.cos(t), np.sin(t))
        return vec1.patches + vec2.patches

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=nframes, blit=True)
    plt.show()
