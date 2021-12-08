import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint
from scipy.fft import fft, fftfreq

n = 10
k = 1000
T = 1000
timestep = 0.01 #sec
window = 20 #sec
window_size = int(window/timestep)

def f(x,t,k): 
    ret = []
    ret.append(0);ret.append(0)
    for i in range(n-2):
        ret.append(x[2*i+3])
        ret.append(k*x[2*i]-k*2*x[2*i+2]+k*x[2*i+4])
    ret.append(0);ret.append(0)
    return ret

x0 = []
for i in range(n):
    x0.append(i)
    if i==0 or i==n-1: x0.append(0)
    else:              x0.append(random.uniform(0,1))

t = np.arange(0,T,step=timestep)
x = odeint(f,x0,t,args=(k,))

def do_fft(a=0,b=len(t)):
    x_sample = x[a:b,:]
    E_x = np.zeros(x_sample.shape[0])
    freq = fftfreq(x_sample.shape[0], d=timestep)
    for i in range(n):
        y = np.abs(fft(x_sample[:,2*i]-x0[2*i]))
        E_x = E_x + (y**2)*(freq**2)*(2*np.pi)**2/2
    return freq, E_x

fig = plt.figure(dpi=300)
ax = fig.gca()
freq, E_x = do_fft()
#freq_, E_x_ = do_fft(0,window_size)
plt.plot(freq/freq[1],E_x,linestyle='-')
#plt.plot(freq_/freq_[1],E_x_,linestyle='--')
plt.show()
'''
ax.set_xlim((np.min(freq/freq[1]),np.max(freq/freq[1])))
ax.set_ylim((0,np.max(E_x)))
curve, = ax.plot([],[], lw=2, color='blue')
title = ax.text(0.5, 0.95, '', transform = ax.transAxes, ha='center', va='center',fontsize=12)

def update(i):
    freq, E_x = do_fft(i,i+window_size)
    curve.set_data(freq/freq[1]*len(t)/window_size, E_x)
    title.set_text('t='+str(i)+'~'+str(i+10)+'s')
    return curve,title,

ani = animation.FuncAnimation(fig=fig, func=update, frames=len(t)-window_size, interval=1, blit=True, repeat=True)
plt.show()
'''

'''
fig = plt.figure(figsize=(4,2),dpi=200)
ax = fig.gca()
ax.set_xlim((-0.5,n-0.5))
ax.axis('off')
title = ax.text(0.5, 0.65, '', transform = ax.transAxes, ha='center', va='center',fontsize=12)

dot = []
for i in range(n):
    d, = ax.plot([], [], color='black', marker='|' if i==0 or i==n-1 else 'o', markersize=15 if i==0 or i==n-1 else 5, linestyle='')
    dot.append(d)

def update(i):
    title.set_text('t='+str(i))
    for j in range(n): dot[j].set_data(x[i][2*j],0)
    return tuple(dot)+(ax,)

def init():
    title.set_text('t=0')
    for j in range(n): dot[j].set_data(x[0][2*j],0)
    return tuple(dot)+(ax,)

ani = animation.FuncAnimation(fig=fig, func=update, frames=100000, init_func=init, interval=1, blit=True, repeat=True)
plt.show()
'''