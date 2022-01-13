import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

N = 20
k = 10
b = 5
g = 0
T = 5000  # sec
timestep = 0.01  # sec
method = 'RK45'
methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']


def f(t, x, k, b, g):
    ret = []
    ret.append(0)
    ret.append(0)
    for i in range(N-1):
        ret.append(x[2*i+3])
        ret.append(k*(x[2*i]-x[2*i+2])**1+k*(x[2*i+4]-x[2*i+2])**1-b*(x[2*i]-x[2*i+2])
                   ** 2+b*(x[2*i+4]-x[2*i+2])**2+g*(x[2*i]-x[2*i+2])**3+g*(x[2*i+4]-x[2*i+2])**3)
    ret.append(0)
    ret.append(0)
    return ret


M = np.array([[math.sin(m*n*np.pi/(N)) for m in range(N+1)]
             for n in range(N+1)])
x0 = M[1]
v0 = np.zeros(N+1)
y0 = np.array([x0[i//2] if i % 2 == 0 else v0[i//2] for i in range(2*N+2)])
sol = solve_ivp(fun=f, t_span=(0, T), y0=y0, method=method, t_eval=np.arange(
    0, T, step=timestep), dense_output=True, args=(k, b, g,))
x = np.array([sol.y[i, ::] for i in range(0, 2*N+2, 2)])
v = np.array([sol.y[i, ::] for i in range(1, 2*N+2, 2)])


def energy(x, v, mode): return np.sum(np.array([1/2*k*(x[i]-x[i-1])**2+1/3*b*(x[i]-x[i-1])**3+1/4*g*(x[i]-x[i-1])**4 for i in range(1, N+1)]))+np.sum(np.array(
    [1/2*v[i]**2 for i in range(1, N)])) if (mode == True) else np.sum(np.array([1/2*k*(x[i]-x[i-1])**2 for i in range(1, N+1)]))+np.sum(np.array([1/2*v[i]**2 for i in range(1, N)]))


def amplitude(A, M, m): return A.dot(M[m])*M[m]


fig = plt.figure(dpi=300)
ax = fig.gca()
ax.set_xlim((0, N+1))
ax.set_ylim((0, 400))
bars = plt.bar(np.arange(1, N), np.zeros(N-1))
#curve, = ax.plot([], [])
title = ax.text(0.5, 0.95, '', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)


best_Rsquared = -1000
best_b = -10000
r_squared = []
b_array = []


def update(i):
    global bars
    global best_Rsquared
    global best_b
    Em = []

    for m in range(N-1):
        E_m = energy(amplitude(x[::, int(i/timestep)], M, m+1),
                     amplitude(v[::, int(i/timestep)], M, m+1), mode=0)
        bars[m].set_height(E_m)
        Em.append(E_m)

    Em = np.array(Em)

    def func(m, a, b):
        return a*(np.sin(m*np.pi/(N)))**2*np.exp(-b*(np.sin(m*np.pi/(N)))**2)

    popt, pcov = curve_fit(func, np.arange(N-1), Em)

    SSres = 0
    SStot = 0
    ybar = np.mean(Em)

    for j in range(N-1):
        SSres += (Em[j]-func(j, popt[0], popt[1]))**2
        SStot += (Em[j]-ybar)**2

    Rsquared = 1-SSres/SStot
    b_array.append(popt[1])
    r_squared.append(Rsquared)
    if Rsquared > best_Rsquared:
        best_Rsquared = Rsquared
        best_b = popt[1]

    E = np.sum([energy(amplitude(x[::, int(i/timestep)], M, m),
               amplitude(v[::, int(i/timestep)], M, m), mode=1) for m in range(1, N)])
    title.set_text('t = '+str(i)+', E = '+str(round(E, 5))+', beta = ' +
                   str(round(popt[1], 5)) + ', r^2 =' + str(round(best_Rsquared, 5)))
    return title,


for i in range(T):
    update(i)
'''
ani = animation.FuncAnimation(
    fig=fig, func=update, frames=T, interval=1, blit=True, repeat=True)

ani.save('plot.gif', writer='pillow', fps=60, dpi=300)
'''
plt.clf()
plt.plot(np.arange(T), r_squared)

plt.savefig('rsquared_5000t20Nk10b5.png', dpi=300)
plt.plot(np.arange(T), b_array)
plt.savefig('beta5000t20Nk10b5.png', dpi=300)
