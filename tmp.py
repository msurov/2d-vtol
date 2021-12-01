import matplotlib.pyplot as plt
import numpy as np


t = np.linspace(0, 2, 100)
s0 = 2

if s0 > 0:
	s = s0 * np.sin(t / s0 + np.pi/2)
	ds = np.cos(t / s0 + np.pi/2)
else:
	s = -s0 * np.sin(t / s0 - np.pi/2)
	ds = -np.cos(t / s0 - np.pi/2)

plt.plot(s[0], ds[0], 'o')
plt.plot(s, ds)
plt.show()