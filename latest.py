import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

N = 10
k = 1000
b = 0
g = 1000
T = 10000 # sec
timestep = 0.01  # sec
method = 'RK45'
methods = ['RK45','RK23','DOP853','Radau','BDF','LSODA']

def f(t, x, k, b, g):
    ret = []
    ret.append(0)
    ret.append(0)
    for i in range(N-1):
        ret.append(x[2*i+3])
        ret.append(k*(x[2*i]-x[2*i+2])**1+k*(x[2*i+4]-x[2*i+2])**1-b*(x[2*i]-x[2*i+2])**2+b*(x[2*i+4]-x[2*i+2])**2+g*(x[2*i]-x[2*i+2])**3+g*(x[2*i+4]-x[2*i+2])**3)
    ret.append(0)
    ret.append(0)
    return ret

M = np.array([[math.sin(m*n*np.pi/(N)) for m in range(N+1)] for n in range(N+1)])
x0 = M[1]
v0 = np.zeros(N+1)
y0 = np.array([x0[i//2] if i%2==0 else v0[i//2] for i in range(2*N+2)])
sol = solve_ivp(fun=f, t_span=(0,T), y0=y0, method=method, t_eval=np.arange(0, T, step=timestep), dense_output=True, args=(k,b,g,))
x = np.array([sol.y[i,::] for i in range(0,2*N+2,2)])
v = np.array([sol.y[i,::] for i in range(1,2*N+2,2)])

energy = lambda x, v, mode : np.sum(np.array([1/2*k*(x[i]-x[i-1])**2+1/3*b*(x[i]-x[i-1])**3+1/4*g*(x[i]-x[i-1])**4 for i in range(1,N+1)]))+np.sum(np.array([1/2*v[i]**2 for i in range(1,N)])) if (mode==True) else np.sum(np.array([1/2*k*(x[i]-x[i-1])**2 for i in range(1,N+1)]))+np.sum(np.array([1/2*v[i]**2 for i in range(1,N)]))
amplitude = lambda A, M, m : A.dot(M[m])*M[m]

fig = plt.figure(dpi=300)
ax = fig.gca()
ax.set_xlim((0,N+1))
ax.set_ylim((0,10000))
curve, = ax.plot([], [])
title = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='center', va='center', fontsize=12)
def update(i):
    curve.set_data(np.arange(1,N),[energy(amplitude(x[::,int(i/timestep)], M, m), amplitude(v[::,int(i/timestep)], M, m), mode=0) for m in range(1,N)])
    E = np.sum([energy(amplitude(x[::,int(i/timestep)], M, m), amplitude(v[::,int(i/timestep)], M, m), mode=1) for m in range(1,N)])
    title.set_text('t = '+str(i)+', E = '+str(round(E,5)))
    return curve,title,
ani = animation.FuncAnimation(fig=fig, func=update, frames=T, interval=1, blit=True, repeat=True)
plt.show()


