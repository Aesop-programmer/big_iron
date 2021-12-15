import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq

n = 10
k = 1000
b = 0
T = 10000 #sec
timestep = 0.01  # sec
window = 100  # sec
window_size = int(window/timestep)

def f(t, x, k, b):
    ret = []
    ret.append(0)
    ret.append(0)
    for i in range(n-2):
        ret.append(x[2*i+3])
        ret.append(k*(x[2*i]-x[2*i+2])**1+k*(x[2*i+4]-x[2*i+2])**1
                  -b*(x[2*i]-x[2*i+2])**2+b*(x[2*i+4]-x[2*i+2])**2)
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
sol = solve_ivp(fun=f, t_span=(0,T), y0=x0, method='LSODA', t_eval=t, dense_output=True, args=(k,b,))
print(sol.message)
x = sol.y

def do_fft(a=0, b=len(x)):
    x_sample = x[a:b,:]
    E_x = np.zeros(x_sample.shape[1])
    freq = fftfreq(x_sample.shape[1], d=timestep)
    for i in range(n):
        y = np.abs(fft(x_sample[2*i,:], n=x_sample.shape[1]))
        E_x = E_x + (y**2)*(freq**2)*(2*np.pi)**2/2
    return freq, E_x

E = []
for i in range(100):
    freq, E_x = do_fft(10000*i, 10000*i+window_size)
    E.append(np.sum(E_x))

plt.plot(t[::10000],E)
plt.savefig('E-t_plot_k%db%d.png'%(k,b),dpi=300) 

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

'''
E = np.zeros(int(T/10))
for i in range(n-1):
    E = E + 0.5*x[2*i+1,::1000]**2 + k/2*(x[2*i,::1000]-x[2*i+2,::1000])**2 + b/3*(x[2*i,::1000]-x[2*i+2,::1000])**3

plt.plot(t[::1000], E)
#plt.show()
plt.savefig('E-t_plot_k%db%d.png'%(k,b),dpi=300)  
'''