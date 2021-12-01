from sympy import sin,cos,symbols,zeros,simplify


x,z,phi = symbols('x z phi', real=True)
epsilon = symbols('epsilon', real=True)

g = zeros(3, 2)
g[0,0] = -sin(phi)
g[1,0] = cos(phi)
g[0,1] = -epsilon * cos(phi)
g[1,1] = -epsilon * sin(phi)
g[2,1] = 1

f = zeros(3,1)
f[1] = -1

g_perp = zeros(1,3)
g_perp[0,0] = cos(phi)
g_perp[0,1] = sin(phi)
g_perp[0,2] = epsilon

tmp = g_perp @ g
tmp = simplify(tmp)
print(tmp)