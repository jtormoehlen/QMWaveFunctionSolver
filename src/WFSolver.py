import sys

import numpy as np
from scipy import integrate
from scipy import sparse
from scipy.misc import derivative

import matplotlib.pyplot as plt
from matplotlib import animation

# Set initial conditions
dx = 0.005                  # spatial separation
x = np.arange(-5, 5, dx)    # spatial grid points

hbar = 1                    # hbar = 1.0545718176461565e-34
m = 1                       # mass
sigma = 0.5                 # width of initial gaussian wave-packet
x0 = -1                    # center of initial gaussian wave-packet
E_0 = 1250
V_0 = 1300
a = 0.2
x_1 = 0
x_2 = 2.5
k = np.sqrt(2. * m * E_0) / hbar

# Initial Wavefunction
A = 1.0 / (sigma * np.sqrt(np.pi))  # normalization constant
psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * k * x)

# Potential V(x)
V = np.zeros(x.shape)
# for i, _x in enumerate(x):
#     if x_1 < _x < x_1 + a or x_2 < _x < x_2 + a:
#         V[i] = V_0
for i, _x in enumerate(x):
    if x_1 < _x < x_1 + a:
        V[i] = V_0
# for i in range(len(x)):
#     V[i] = 400 * x[i]

print('Total Probability: ', np.sum(np.abs(psi0)**2)*dx)

# Laplace Operator (Finite Difference)
D2 = sparse.diags([1, -2, 1],
                  [-1, 0, 1],
                  shape=(x.size, x.size)) / dx**2


# Solve Schrödinger Equation
def psi_t(t, psi):
    return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)


# Solve the Initial Value Problem
dt = 0.001  # time interval for snapshots
t0 = 0.0    # initial time
tf = 0.2    # final time
t_eval = np.arange(t0, tf, dt)  # recorded time shots

print('Solving initial value problem')
sol = integrate.solve_ivp(psi_t,
                          t_span=[t0, tf],
                          y0=psi0,
                          t_eval=t_eval,
                          method='RK23')

# Animation
fig = plt.figure(figsize=(6, 4))

# ax1 = plt.subplot(2, 1, 1)
# ax1.set_xlim(0, 10)
# ax1.set_ylim(-2, 2)
# title = ax1.set_title('')
# line11, = ax1.plot([], [], 'k--', label=r'$V(x)$')
# line12, = ax1.plot([], [], 'tab:blue', label=r'$\vert \psi \vert^2$')
# plt.legend(loc=1, fontsize=8, fancybox=False)

# ax2 = plt.subplot(2, 1, 2)
ax2 = plt.subplot(1, 1, 1)
ax2.set_xlim(-5, 5)
ax2.set_ylim(-2, 2)
title = ax2.set_title('')
line21, = ax2.plot([], [], 'k--', label=r'$V(x)$')
line22, = ax2.plot([], [], label=r'$Re\{ \psi \}$')
plt.legend(loc=4, fontsize=8, fancybox=False)


def init():
    # line11.set_data(x, V * 0.001)
    line21.set_data(x, V * 0.001)
    return line21,


def animate(i):
    # line12.set_data(x, np.abs(sol.y[:, i])**2)
    line22.set_data(x, np.abs(sol.y[:, i]) ** 2)
    title.set_text('Time = {0:1.3f}'.format(sol.t[i]))
    return line22,


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(sol.t), interval=200, blit=True)

print('Generating GIF')
anim.save('step@2x.gif', writer='imagemagick', fps=15, dpi=100, extra_args=['-layers Optimize'])

sys.exit(0)
