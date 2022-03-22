from draw_quadrotor import Quadrotor
import matplotlib.pyplot as plt
import numpy as np
from trajectory import load_trajectory
from matplotlib import animation
from scipy.interpolate import make_interp_spline
from anim_trace import TraceLine
from anim_scope import Scope
from matplotlib.offsetbox import AnchoredText


def set_ax_style(ax, grid=True):
    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=1, direction='in', right='on')
    ax.grid(grid)


def set_annotation(ax, text):
    at = AnchoredText(
        text, prop=dict(size=12), frameon=True, loc='lower right')
    at.patch.set_alpha(0.7)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)


class AnimateQuadrotor:
    def __init__(self, simdata, reftraj):
        simt = simdata['tau']
        simq = simdata['q']
        simu = simdata['u']
        reft = reftraj['t']
        refq = reftraj['q']
        refu = reftraj['u']
        self.qsp = make_interp_spline(simt, simq, k=1)
        self.usp = make_interp_spline(simt, simu, k=1)
        self.tspan = [simt[0], simt[-1]]

        '''
            main window
        '''
        fig = plt.figure('Quadrotor', figsize=(19.20,10.80))
        self.fig = fig
        spec = fig.add_gridspec(nrows=4, ncols=3)
        ax_main = fig.add_subplot(spec[:, 0:2])
        plt.axis('equal')
        qr = Quadrotor(ax_main, 'fig/drawing.svg')
        qrsz = 0.4 # TODO: fix
        qr.set_default_transform(0, 0, 0, sx=0.02, sy=-0.02)
        trace = TraceLine(100, color='lightblue', lw=3)
        ax_main.add_collection(*trace.patches)
        xmin = np.min(simq[:,0])
        xmax = np.max(simq[:,0])
        ymin = np.min(simq[:,1])
        ymax = np.max(simq[:,1])
        ax_main.set_xlim([xmin - qrsz, xmax + qrsz])
        ax_main.set_ylim([ymin - qrsz, ymax + qrsz])
        set_ax_style(ax_main, grid=False)

        '''
            x,y plots
        '''
        periodic = np.allclose(refq[0], refq[-1])
        bc_type = 'periodic' if periodic else None
        refsp = make_interp_spline(reft, refq, k=3, bc_type=bc_type)
        refq = refsp(simt)

        ax_x = fig.add_subplot(spec[0, 2])
        scope_x = Scope(ax_x, color='blue', alpha=0.5, lw=2)
        plt.plot(simt, refq[:,0], color='green', lw=3, alpha=0.5, ls=(0,(5,1)))
        ax_x.set_ylim(np.min(simq[:,0]), np.max(simq[:,0]))
        set_ax_style(ax_x)
        set_annotation(ax_x, R'$x$ coordinate')

        ax_y = fig.add_subplot(spec[1, 2], sharex=ax_x)
        scope_y = Scope(ax_y, color='blue', alpha=0.5, lw=2)
        plt.plot(simt, refq[:,1], color='green', lw=3, alpha=0.5, ls=(0,(5,1)))
        ax_y.set_ylim(np.min(simq[:,1]), np.max(simq[:,1]))
        set_ax_style(ax_y)
        set_annotation(ax_y, R'$y$ coordinate')

        '''
            u1,u2 plots
        '''
        periodic = np.allclose(refu[0], refu[-1])
        bc_type = 'periodic' if periodic else None
        refsp = make_interp_spline(reft, refu, k=3, bc_type=bc_type)
        refu = refsp(simt)

        ax_u1 = fig.add_subplot(spec[2, 2], sharex=ax_x)
        scope_u1 = Scope(ax_u1, color='blue', alpha=0.5, lw=2)
        plt.plot(simt, refu[:,0], color='green', lw=3, alpha=0.5, ls=(0,(5,1)))
        ax_u1.set_ylim(np.min(simu[:,0]), np.max(simu[:,0]))
        set_ax_style(ax_u1)
        set_annotation(ax_u1, R'$u_1$ control input')

        ax_u2 = fig.add_subplot(spec[3, 2], sharex=ax_x)
        scope_u2 = Scope(ax_u2, color='blue', alpha=0.5, lw=2)
        plt.plot(simt, refu[:,1], color='green', lw=3, alpha=0.5, ls=(0,(5,1)))
        ax_u2.set_ylim(np.min(simu[:,1]), np.max(simu[:,1]))
        set_ax_style(ax_u2)
        set_annotation(ax_u2, R'$u_2$ control input')

        self.qr = qr
        self.scope_x = scope_x
        self.scope_y = scope_y
        self.trace = trace
        self.scope_u1 = scope_u1
        self.scope_u2 = scope_u2

        plt.subplots_adjust(0.04, 0.04, 0.99, 0.99, hspace=0.02, wspace=0.09)
    
    def get_patches(self):
        return self.qr.patches + self.scope_x.patches + \
            self.scope_y.patches + self.trace.patches + \
            self.scope_u1.patches + self.scope_u2.patches

    def run(self, filepath=None, fps=60, animtime=None, speedup=None):
        t1,t2 = self.tspan

        if speedup is not None:
            assert speedup > 0
            nframes = int((t2 - t1) * fps / speedup)
        elif animtime is not None:
            assert animtime > 0
            nframes = int(animtime * fps)
            speedup = (t2 - t1) / animtime
        else:
            nframes = int((t2 - t1) * fps)
            speedup = 1

        def animinit():
            return self.get_patches()

        def animupdate(iframe):
            t1,_ = self.tspan
            ti = t1 + iframe * speedup / fps
            q = self.qsp(ti)
            self.qr.move(*q)
            self.trace.update(q[0], q[1])
            self.scope_x.update(ti, q[0])
            self.scope_y.update(ti, q[1])
            u = self.usp(ti)
            self.scope_u1.update(ti, u[0])
            self.scope_u2.update(ti, u[1])
            return self.get_patches()

        anim = animation.FuncAnimation(self.fig, animupdate, init_func=animinit, frames=nframes, blit=True)
        if filepath is not None:
            if filepath.endswith('.gif'):
                writer='imagemagick'
            else:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=60, metadata=dict(artist='Maksim Surov'), bitrate=800*60)
            anim.save(filepath, writer)
        else:
            plt.show()


if __name__ == '__main__':
    sim = load_trajectory('data/sim.npy')
    traj = load_trajectory('data/traj.npy')
    anim = AnimateQuadrotor(sim, traj)
    anim.run(animtime=20, filepath='data/anim.mp4')
    # anim.run(animtime=10)
