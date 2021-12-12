import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint
from scipy.fft import fft, fftfreq

n = 10
k = 1000
b = 1
T = 2000
timestep = 0.01  # sec
window = 100  # sec
window_size = int(window/timestep)


def f(x, t, k, b):
    ret = []
    ret.append(0)
    ret.append(0)
    for i in range(n-2):
        ret.append(x[2*i+3])
        ret.append(k*x[2*i]-k*2*x[2*i+2]+k*x[2*i+4]-b*
                   (x[2*i+2]-x[2*i])**2-b*(x[2*i+4]-x[2*i+2])**2)
    ret.append(0)
    ret.append(0)
    return ret


x0 = []


for i in range(n):
    x0.append(0)
    if i == 0 or i == n-1:
        x0.append(0)
    else:
        x0.append(1)
        
t = np.arange(0, T, step=timestep)
x = odeint(f, x0, t, args=(k, b,))


def do_fft(a=0, b=len(t)):
    x_sample = x[a:b, :]
    E_x = np.zeros(x_sample.shape[0])
    freq = fftfreq(x_sample.shape[0], d=timestep)
    for i in range(n):
        y = np.abs(fft(x_sample[:,2*i], n=x_sample.shape[0]))
        E_x = E_x + (y**2)*(freq**2)*(2*np.pi)**2/2
    return freq, E_x

'''
fig = plt.figure(dpi=300)
ax = fig.gca()
#freq, E_x = do_fft(85000,90000)
freq_, E_x_ = do_fft(0, window_size)
#plt.plot(freq_/freq_[1],E_x_,linestyle='-')
#plt.plot(freq_/freq_[1], E_x_, linestyle='--')
#plt.show()


ax.set_xlim((np.min(freq_/freq_[1]), np.max(freq_/freq_[1])))
ax.set_ylim((0, np.max(E_x_)))
curve, = ax.plot([], [], lw=2, color='blue')
title = ax.text(0.5, 0.95, '', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)


def update(i):
    freq, E_x = do_fft(10000*i, 10000*i+window_size)
    curve.set_data(freq/freq[1], E_x)
    title.set_text('t='+str(i)+'~'+str(i+10)+'s')
    return curve, title,


ani = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=200, blit=True, repeat=True)
plt.show()
'''
def CMposition(x):
    CM_x = np.zeros(x.shape[0])
    for i in range(1, n-1):
        CM_x = CM_x + x[:,2*i]
    return CM_x


#plt.plot(t, x[:,2*5])
#plt.show()

E = np.zeros(int(T/10))
for i in range(n-1):
    E = E + 0.5*k*(x[::1000,2*i]-x[::1000,2*i+2])**2 + b/3*(x[::1000,2*i]-x[::1000,2*i+2])**3 + 0.5*x[::1000,2*i+1]**2

plt.plot(t[::1000], E)
plt.show()   
    