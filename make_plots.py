from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import fsolve, brentq
from scipy.integrate import solve_ivp
import sympy as sy
from collections import namedtuple
from draw import plot_helicopter
import matplotlib.patches as mpatches


matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'sans-serif'
rc('text', usetex=True)


def maketuple(name : str, d : dict):
    T = namedtuple(name, sorted(d))
    return T(**d)


class Reduced:
    def __init__(self, _X, _Z, _Φ):
        s = sy.symbols('s', real=True)

        X = _X(s)
        dX = _X(s).diff(s)
        ddX = _X(s).diff(s, 2)

        Z = _Z(s)
        dZ = _Z(s).diff(s)
        ddZ = _Z(s).diff(s, 2)

        Φ = _Φ(s)
        dΦ = _Φ(s).diff(s)
        ddΦ = _Φ(s).diff(s, 2)

        α = dX * sy.cos(Φ) + dZ * sy.sin(Φ)
        β = ddX * sy.cos(Φ) + ddZ * sy.sin(Φ)
        γ = sy.sin(Φ)

        α = α.simplify()
        β = β.simplify()
        γ = γ.simplify()

        self.expr = maketuple('Expr', {
            's': s,
            'X': X,
            'dX': dX,
            'ddX': ddX,
            'Z': Z,
            'dZ': dZ,
            'ddZ': ddZ,
            'Φ': Φ,
            'dΦ': dΦ,
            'ddΦ': ddΦ,
            'α': α,
            'β': β,
            'γ': γ,
        })
        self.fun = maketuple('Fun', {
            'α': sy.Lambda(s, α),
            'β': sy.Lambda(s, β),
            'γ': sy.Lambda(s, γ)
        })
        self.num = maketuple('Fun', {
            'X': sy.lambdify(s, X, 'numpy'),
            'dX': sy.lambdify(s, dX, 'numpy'),
            'ddX': sy.lambdify(s, ddX, 'numpy'),
            
            'Z': sy.lambdify(s, Z, 'numpy'),
            'dZ': sy.lambdify(s, dZ, 'numpy'),
            'ddZ': sy.lambdify(s, ddZ, 'numpy'),

            'Φ': sy.lambdify(s, Φ, 'numpy'),
            'dΦ': sy.lambdify(s, dΦ, 'numpy'),
            'ddΦ': sy.lambdify(s, ddΦ, 'numpy'),

            'α': sy.lambdify(s, α, 'numpy'),
            'β': sy.lambdify(s, β, 'numpy'),
            'γ': sy.lambdify(s, γ, 'numpy')
        })


def get_numeric_linsys(reduced):
    def f(s, y):
        α = reduced.num.α(s)
        β = reduced.num.β(s)
        γ = reduced.num.γ(s)
        return -(2*β*y + γ) / α
    return f


def get_numeric_nonlin(reduced):
    def f(s, ds):
        α = reduced.num.α(s)
        β = reduced.num.β(s)
        γ = reduced.num.γ(s)
        return -(β*ds**2 + γ) / α
    return f

def integrate(rhs, y0, tspan, step):
    t1,t2 = tspan
    n = int(np.abs(np.round((t2 - t1) / step)))
    t = np.linspace(t1, t2, n)
    y0 = np.atleast_1d(y0)
    sol = solve_ivp(rhs, tspan, y0, t_eval=t)
    return sol['t'], sol['y'].T

def integrate_singular(s1, s2, ss, rhs):
    ε = 1e-5 * (ss - s1)
    sol1 = integrate(rhs, 0., [s1, ss - ε], step=1e-3)
    sol2 = integrate(rhs, 0., [s2, ss + ε], step=1e-3)
    s = np.concatenate((sol1[0], sol2[0][::-1]))
    y = np.concatenate((sol1[1], sol2[1][::-1]))
    f = make_interp_spline(s, np.sqrt(2*y), k=3)
    dt = lambda s, t: 1 / f(s)
    s,t = integrate(dt, 0, [s1 + ε*10, s2 - ε*10], step=1e-3)
    t = np.reshape(t, (-1,))
    return t, s


def phase_portrait_singular(rhs, sa, sb, ss, na, nb):
    ε = 1e-5

    left = np.linspace(sa, ss, na + 1)[:-1]
    right = np.linspace(ss, sb, nb + 1)[1:]

    ds_max = 2.
    lw = 0.5

    for s in left:
        s, y = integrate(rhs, 0, [s, ss-ε], step=1e-3)
        plt.plot(s, np.sqrt(2*y), 'b', lw=lw)
        plt.plot(s, -np.sqrt(2*y), 'b', lw=lw)

    for s in right:
        s, y = integrate(rhs, 0, [s, ss+ε], step=1e-3)
        plt.plot(s, np.sqrt(2*y), 'b', lw=lw)
        plt.plot(s, -np.sqrt(2*y), 'b', lw=lw)

    for ds0 in np.linspace(0.0, ds_max, na)[2:]:
        y0 = ds0**2 / 2
        s, y = integrate(rhs, y0, [sa, ss-ε], step=1e-3)
        plt.plot(s, np.sqrt(2*y), 'b', lw=lw)
        plt.plot(s, -np.sqrt(2*y), 'b', lw=lw)

    for ds0 in np.linspace(0.0, ds_max, nb)[2:]:
        y0 = ds0**2 / 2
        s, y = integrate(rhs, y0, [sb, ss+ε], step=1e-3)
        plt.plot(s, np.sqrt(2*y), 'b', lw=lw)
        plt.plot(s, -np.sqrt(2*y), 'b', lw=lw)

    for s in left:
        y0 = ds_max**2 / 2
        s, y = integrate(rhs, y0, [s, ss-ε], step=1e-3)
        plt.plot(s, np.sqrt(2*y), 'b', lw=lw)
        plt.plot(s, -np.sqrt(2*y), 'b', lw=lw)

    for s in right:
        y0 = ds_max**2 / 2
        s, y = integrate(rhs, y0, [s, ss+ε], step=1e-3)
        plt.plot(s, np.sqrt(2*y), 'b', lw=lw)
        plt.plot(s, -np.sqrt(2*y), 'b', lw=lw)



def example1():
    s = sy.Dummy()
    Φ = sy.Lambda(s, 1*sy.atan(s) + sy.pi/2)
    X = sy.Lambda(s, -2*s)
    Z = sy.Lambda(s, -s**2/2)

    reduced = Reduced(X, Z, Φ)
    rhs = get_numeric_linsys(reduced)

    print('α(s) = ', reduced.fun.α.expr)
    print('β(s) = ', reduced.fun.β.expr)
    print('γ(s) = ', reduced.fun.γ.expr)

    if False:
        s0 = sy.symbols('s_0', real=True)
        t = sy.symbols('t', real=True)
        s = s0 * sy.sin(t / s0 + sy.pi/2)
        ϕ = reduced.expr.Φ.subs(reduced.expr.s, s)
        dϕ = ϕ.diff(t)
        ddϕ = ϕ.diff(t, 2)

        x = reduced.expr.X.subs(reduced.expr.s, s)
        z = reduced.expr.Z.subs(reduced.expr.s, s)

        ddx = x.diff(t, 2)
        ddz = z.diff(t, 2)

        w = ddϕ.simplify()
        print(sy.latex(w))

        u = -ddx * sy.sin(ϕ) + ddz * sy.cos(ϕ) + sy.cos(ϕ)
        u = u.simplify()
        print(sy.latex(u))

    if True:
        '''
            Phase Portrait
        '''
        plt.figure('Phase Portrait', figsize=(6,6))
        phase_portrait_singular(rhs, -2, 2, 0, 10, 10)
        plt.fill_between([-2, 2], [1, 1], [-1, -1], alpha=0.1, color='aqua')
        plt.fill_between([-2, 2], [1, 1], [2, 2], alpha=0.1, color='gray')
        plt.fill_between([-2, 2], [-1, -1], [-2, -2], alpha=0.1, color='gray')
        plt.plot([-2,2], [1, 1], '--', color='black')
        plt.plot([-2,2], [-1, -1], '--', color='black')

        s, y = integrate(rhs, 0, [-1, -1e-5], step=1e-3)
        plt.plot(s, np.sqrt(2*y), 'r', lw=2)
        plt.plot(s, -np.sqrt(2*y), 'r', lw=2)

        s, y = integrate(rhs, 0, [1.6, 1e-5], step=1e-3)
        plt.plot(s, np.sqrt(2*y), 'g', lw=2)
        plt.plot(s, -np.sqrt(2*y), 'g', lw=2)

        s, y = integrate(rhs, 0, [1, 1e-5], step=1e-3)
        plt.plot(s, np.sqrt(2*y), 'r', lw=2)
        plt.plot(s, -np.sqrt(2*y), 'r', lw=2)

        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\dot{\theta}$')
        plt.grid()
        plt.gca().set_aspect(1)
        plt.axvline(0.)
        plt.axhline(0.)
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        periodic_bar = mpatches.Patch(color='aqua', label='periodic trajectories')
        nonperiodic_bar = mpatches.Patch(color='gray', label='non-periodic trajectories')
        plt.legend(handles=[periodic_bar,nonperiodic_bar], loc='lower right')
        plt.tight_layout()
        plt.savefig('vtol_phase.pdf')
        plt.show()

    if True:
        '''
            Trajectory Snapshots
        '''
        plt.figure('Trajectory Snapshots', figsize=(8,3))
        t, s = integrate_singular(-1, 1, 0., rhs)
        T = t[-1]
        s = make_interp_spline(t, s, k=3)
        plt.gca().set_aspect(1)
        t = np.linspace(0, T, 5)
        for ti in t:
            si = s(ti)
            xi = reduced.num.X(si)
            zi = reduced.num.Z(si)
            ϕi = reduced.num.Φ(si)
            plot_helicopter(xi, zi, ϕi, 0.5)

        t = np.linspace(0, T, 100)
        x = reduced.num.X(s(t))
        z = reduced.num.Z(s(t))
        plt.plot(x, z, '--')
        plt.xlabel('$x$')
        plt.ylabel('$z$')
        plt.grid()
        plt.tight_layout()
        plt.savefig('vtol_trajectory.pdf')

    if True:
        '''
            Plots
        '''
        t, s = integrate_singular(-1, 1, 0., rhs)
        s = np.concatenate((s[:-1], s[::-1]))
        t = np.concatenate((t[:-1], t[-1] + t))
        T = t[-1]
        s_ = make_interp_spline(t, s, k=5)
        t = np.linspace(0, T, 1000)
        s = s_(t)
        x = reduced.num.X(s)
        z = reduced.num.Z(s)
        ϕ = reduced.num.Φ(s)
        sys = get_numeric_nonlin(reduced)
        ds = s_(t, 1)
        dds = np.array([sys(s_(ti), s_(ti, 1)) for ti in t])

        ddx = reduced.num.ddX(s) * ds**2 + reduced.num.dX(s) * dds
        ddz = reduced.num.ddZ(s) * ds**2 + reduced.num.dZ(s) * dds
        ddϕ = reduced.num.ddΦ(s) * ds**2 + reduced.num.dΦ(s) * dds

        u = -ddx * np.sin(ϕ) + ddz * np.cos(ϕ) + np.cos(ϕ)
        w = ddϕ

        fig,axes = plt.subplots(3, 1, sharex=True)
        fig.canvas.set_window_title('coordinates')
        axes[0].plot(t, x)
        axes[0].set_ylabel(R'$x$')
        axes[0].grid(True)
        axes[1].plot(t, z)
        axes[1].set_ylabel(R'$z$')
        axes[1].grid(True)
        axes[2].plot(t, ϕ)
        axes[2].set_ylabel(R'$\phi$')
        axes[2].grid(True)
        plt.xlabel(R'$t$')
        plt.tight_layout()
        plt.savefig('vtol_coordinates.pdf')

        fig,axes = plt.subplots(2, 1, sharex=True)
        fig.canvas.set_window_title('inputs')
        axes[0].plot(t, u)
        axes[0].set_ylabel(R'$u$')
        axes[0].grid(True)
        axes[1].plot(t, w)
        axes[1].set_ylabel(R'$w$')
        axes[1].grid(True)
        plt.xlabel(R'$t$')
        plt.tight_layout()
        plt.savefig('vtol_inputs.pdf')


def dynamics_rhs(t, state):
    x,y,ϕ,dx,dy,dϕ = state
    u = 2.833
    w = -1.*(ϕ - np.pi)
    return np.array([
        dx, dy, dϕ, -u*np.sin(ϕ), -1. + u*np.cos(ϕ), w
    ])


def example2():
    state0 = [1.2, 0., -0.2, 0., 0., 0.]
    t, state = integrate(dynamics_rhs, state0, [0, np.pi], step=1e-3)

    T = t[-1]
    traj = make_interp_spline(t, state, k=3)
    plt.axis('equal')
    t = np.linspace(0, T, 9)
    for ti in t:
        x,y,ϕ,_,_,_ = traj(ti)
        plot_helicopter(x, y, ϕ, 0.25)

    plt.plot(state[:,0], state[:,1], '--')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # example1()
    # example2()
