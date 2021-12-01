import matplotlib.pyplot as plt
import numpy as np


def cont_circle(xc, yc, r):
	t = np.linspace(0, 2 * np.pi, 100)
	x = xc + r * np.cos(t)
	y = yc + r * np.sin(t)
	return np.array([x,y]).T

def cont_ellipse(xc, yc, a, b, ϕ=0):
	t = np.linspace(0, 2 * np.pi, 100)
	x = xc + a * np.cos(t + ϕ)
	y = yc + b * np.sin(t + ϕ)
	return np.array([x,y]).T

def cont_rect(xc, yc, w, h):
	return np.array([
		[xc - w/2, yc - h/2],
		[xc + w/2, yc - h/2],
		[xc + w/2, yc + h/2],
		[xc - w/2, yc + h/2],
		[xc - w/2, yc - h/2],
	])

def plot_cont(c, *wargs, **kwargs):
	return plt.plot(c[:,0], c[:,1], *wargs, **kwargs)

def plot_helicopter(x, y, ϕ):
	c1 = cont_ellipse(0, 0, 2, 1)
	c2 = cont_circle(-4, 0.5, 0.5)
	c3 = np.array([
		[-3.51, 0.7],
		[-1.45, 0.7]
	])
	c4 = np.array([
		[-1.45, -0.69],
		[-3.57, 0.2],
	])
	c5 = cont_rect(0, 1.25, 6, 0.2)
	c6 = cont_rect(0, 1.25, 0.25, 0.5)
	c7 = cont_ellipse(1, 0, 0.5, 0.5)
	plot_cont(c1, 'b')
	plot_cont(c2, 'b')
	plot_cont(c3, 'b')
	plot_cont(c4, 'b')
	plot_cont(c5, 'b')
	plot_cont(c6, 'b')
	plot_cont(c7, 'b')


plt.axis('equal')
plot_helicopter(0, 0, 0)
# plt.grid()
plt.show()
