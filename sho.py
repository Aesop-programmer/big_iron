import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint
from scipy.fft import fft, fftfreq
n = 7
k = 0.1
N = 600


def f(x, t, k):
    ret = []
    ret.append(0)
    ret.append(0)
    for i in range(n-2):
        ret.append(x[2*i+3])
        ret.append(k*x[2*i]-k*2*x[2*i+2]+k*x[2*i+4])
    ret.append(0)
    ret.append(0)
    return ret


x0 = [
    0, 0,
    1, 0.5,
    2, 0,
    3, 0.5,
    4, 0,
    5, 0.5,
    6, 0,
]

t = np.arange(0, 1000, step=0.02)
x = odeint(f, x0, t, args=(k,))


fig = plt.figure(figsize=(4, 2), dpi=200)
ax = fig.gca()
ax.set_xlim((-0.5, n-0.5))
ax.axis('off')
title = ax.text(0.5, 0.65, '', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)

dot = []
for i in range(n):
    d, = ax.plot([], [], color='black', marker='|' if i == 0 or i == n -
                 1 else 'o', markersize=15 if i == 0 or i == n-1 else 5, linestyle='')
    dot.append(d)


def update(i):
    title.set_text('t='+str(i))
    for j in range(n):
        dot[j].set_data(x[i][2*j], 0)
    return tuple(dot)+(ax,)


def init():
    title.set_text('t=0')
    for j in range(n):
        dot[j].set_data(x[0][2*j], 0)
    return tuple(dot)+(ax,)


ani = animation.FuncAnimation(
    fig=fig, func=update, frames=50000, init_func=init, interval=1, blit=True, repeat=True)
plt.show()
