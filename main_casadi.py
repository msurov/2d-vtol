from casadi import SX, vertcat, horzcat, sin, cos, \
    simplify, substitute


# phase variables
x = SX.sym('x', 3)
epsilon = SX.sym('epsilon')
phi = x[2]
f = SX.zeros(3)
f[1] = -1
g1 = vertcat(
    -sin(phi),
    cos(phi),
    0
)
g2 = vertcat(
    -epsilon*cos(phi),
    -epsilon*sin(phi),
    1
)
g = horzcat(g1, g2)
g_perp = horzcat(
    cos(phi), sin(phi), epsilon
)

z = g_perp @ g
z = simplify(z)
z = substitute(z, x, [1,2,3])
z = substitute(z, epsilon, 4)
print(z)