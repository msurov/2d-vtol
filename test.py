from casadi import SX, DM, MX, vertcat, horzcat, sin, cos, \
    simplify, substitute, pi, jacobian, nlpsol, Function, pinv, \
    interpolant, Callback, Sparsity, if_else, evalf, reshape
from bsplinesx import bsplinesf, bsplinesx
from scipy.interpolate import make_interp_spline, BSpline
from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt
from time import time


# x = MX.sym('x')
# A = spsf(x)
# print(A)

# print(spsf(x).reshape([d1, d2]))
# print(sp1(0.353))
# F = reshape(A, d1, d2).T
# F = Function('F', [x], [F])
# t = time()
# print(F(0.1254235))
# t = time() - t
# print(t)
# print(DM(sp(0.1254235)))
# print(DM(sp(0.34523)))
# sp1 = BSpline(t, c, k)
# spsf = bsplinesf(sp1)
