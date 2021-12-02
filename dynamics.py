from dataclasses import dataclass
from casadi import SX, vertcat, horzcat, sin, cos, Function, pinv

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
    epsilon=0.2,
    gravity=1.0
)

class Dynamics:
    R'''
        @brief Symbolic expressions of the 2D VTOL dynamics
        \[
            \ddot{x}&=-u\sin\phi-\varepsilon w\cos\phi\\
            \ddot{z}&=-1+u\cos\phi-\varepsilon w\sin\phi\\
            \ddot{\phi}&=w
        \]
    '''
    def __init__(self, parameters):
        # parameters
        epsilon = parameters.epsilon
        gravity = parameters.gravity

        # phase coordinates
        q = SX.sym('q', 3)
        dq = SX.sym('dq', 3)
        phi = q[2]

        M = SX.eye(3)
        C = SX.zeros(3,3)
        G = SX.zeros(3)
        G[1] = gravity
        b1 = vertcat(
            -sin(phi),
            cos(phi),
            0
        )
        b2 = vertcat(
            -epsilon*cos(phi),
            -epsilon*sin(phi),
            1
        )
        B = horzcat(b1, b2)
        B_perp = horzcat(
            cos(phi), sin(phi), epsilon
        )

        self.q = q
        self.dq = dq
        self.M = Function('M', [q], [M])
        self.C = Function('C', [q,dq], [C])
        self.G = Function('G', [q], [G])
        self.B = Function('B', [q], [B])
        self.B_perp = Function('B_perp', [q], [B_perp])


def get_inv_dynamics(dynamics):
    q = dynamics.q
    dq = dynamics.dq
    ddq = SX.sym('ddq', *q.shape)
    lhs = dynamics.M(q) @ ddq + dynamics.C(q, dq) @ dq + dynamics.G(q)
    u = pinv(dynamics.B(q)) @ lhs
    return Function('u', [q,dq,ddq], [u])


def main():
    d = Dynamics(parameters)
    get_inv_dynamics(d)


if __name__ == '__main__':
    main()
