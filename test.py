import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc

matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'Latin Modern Math'
rc('text', usetex=True)

plt.subplot2grid((4, 2), (0, 0), rowspan=2)
plt.xlabel('$x$')
plt.ylabel('$\dot x$')
plt.grid(True)

plt.subplot2grid((4, 2), (2, 0), rowspan=2)
plt.xlabel('$z$')
plt.ylabel('$\dot z$')
plt.grid(True)

plt.subplot2grid((4, 2), (0, 1), rowspan=2)
plt.xlabel(R'$\theta$')
plt.ylabel(R'$\dot \theta$')
plt.grid(True)

ax = plt.subplot2grid((4, 2), (2, 1), rowspan=1)
plt.tick_params('x', labelbottom=False)
plt.ylabel(R'$u_1$')
plt.grid(True)

plt.subplot2grid((4, 2), (3, 1), rowspan=1, sharex=ax)
plt.xlabel(R'$t$')
plt.ylabel(R'$u_2$')
plt.grid(True)

plt.tight_layout()
plt.show()
