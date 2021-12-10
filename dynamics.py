from dataclasses import dataclass
from casadi import SX, MX, vertcat, horzcat, sin, cos, Function, pinv, cross

mm = MX

@dataclass
class Parameters:
    R'''
        @brief Parameters of the 2D VTOL system
    '''
    epsilon : float # engines dihedral angle
    gravity : float # gravity acceleration

'''
    Default parameters
'''
parameters = Parameters(
    epsilon = 0.2,
    gravity = 1.0
)

class Dynamics:
    R'''
        @brief Symbolic expressions of the 2D VTOL dynamics
        \[
            \ddot{x}	= -u_{1}\sin\phi + \varepsilon u_{2}\cos\phi
            \ddot{z}	= -1+u_{1}\cos\phi + \varepsilon u_{2}\sin\phi
            \ddot{\phi} = u_{2}
        \]
    '''
    def __init__(self, parameters):
        # parameters
        epsilon = parameters.epsilon
        gravity = parameters.gravity

        # phase coordinates
        q = mm.sym('q', 3)
        dq = mm.sym('dq', 3)
        u = mm.sym('u', 2)
        phi = q[2]

        M = mm.eye(3)
        C = mm.zeros(3,3)
        G = mm.zeros(3)
        G[1] = gravity
        b1 = vertcat(
            -sin(phi),
            cos(phi),
            0
        )
        b2 = vertcat(
            epsilon*cos(phi),
            epsilon*sin(phi),
            1
        )
        B = horzcat(b1, b2)
        B_perp = cross(b1, b2).T

        self.nq = 3
        self.nu = 2
        self.u = u
        self.q = q
        self.dq = dq
        self.M = Function('M', [q], [M])
        self.C = Function('C', [q,dq], [C])
        self.G = Function('G', [q], [G])
        self.B = Function('B', [q], [B])
        self.B_perp = Function('B_perp', [q], [B_perp])
        self.rhs = Function('rhs', [q,dq,u], [vertcat(dq, pinv(M) @ (-C @ dq - G + B @ u))])


def get_inv_dynamics(dynamics):
    R'''
        returns expression for 
            \[
                u = u(q,\dot{q},\ddot{q})
            \]
    '''
    q = dynamics.q
    dq = dynamics.dq
    ddq = mm.sym('ddq', *q.shape)
    lhs = dynamics.M(q) @ ddq + dynamics.C(q, dq) @ dq + dynamics.G(q)
    u = pinv(dynamics.B(q)) @ lhs
    return Function('u', [q,dq,ddq], [u])


def main():
    d = Dynamics(parameters)
    get_inv_dynamics(d)


if __name__ == '__main__':
    main()
