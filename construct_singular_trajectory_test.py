def test_trajectory(dynamics : Dynamics, trajectory : dict):
    f = dynamics.rhs
    t = trajectory['t']
    q = trajectory['q']
    dq = trajectory['dq']
    u = trajectory['u']
    usp = make_interp_spline(t, u)
    x = np.concatenate((q, dq), axis=1)

    def rhs(t, x):
        q = x[0:3]
        dq = x[3:6]
        u = usp(t)
        ans = f(q,dq,u)
        return np.reshape(ans, (-1,))

    sol = solve_ivp(rhs, [t[0], t[-1]], x[0,:], t_eval=t)
    t = sol['t']
    x = sol['y'].T
    traj1 = {
        't': t,
        'q': x[:,0:3],
        'dq': x[:,3:6],
    }
    plt.figure('Compare trajectories')
    plot_trajectory(traj, ls='--', lw=2)
    plot_trajectory(traj1, alpha=0.8)
    plt.tight_layout()


--------
    traj = load_trajectory(trajfile)
    test_trajectory(dynamics, traj)
    plt.show()
