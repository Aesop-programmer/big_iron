import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq

n = 10
k = 1000
b = 30000
T = 30000 #sec
timestep = 0.01  # sec
window = 100  # sec
window_size = int(window/timestep)
method = 'DOP853'
methods = ['RK45','RK23','DOP853','Radau','BDF','LSODA']

def do_fft(x, a=0, b=int(T/timestep)):
    x_sample = x[:,a:b]
    E_x = np.zeros(x_sample.shape[1])
    freq = fftfreq(x_sample.shape[1], d=timestep)
    for i in range(n):
        y = np.abs(fft(x_sample[2*i,:], n=x_sample.shape[1]))
        E_x = E_x + (y**2)*(freq**2)*(2*np.pi)**2/2
    return freq, E_x

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
    x0.append(1e-3*math.sin(i/(n-1)*math.pi))
    x0.append(0)
        
t = np.arange(0, T, step=timestep)

sol = solve_ivp(fun=f, t_span=(0,T), y0=x0, method=method, t_eval=t, dense_output=True, args=(k,b,))
x = sol.y

fig = plt.figure(dpi=300)
ax = fig.gca()
freq_, E_x_ = do_fft(x, 0, window_size)
ax.set_xlim((np.min(freq_/freq_[1])*1.05, np.max(freq_/freq_[1])*1.05))
ax.set_ylim((0, np.max(E_x_)*1.05))
#ax.set_ylim((0,1E8))
curve, = ax.plot([], [], lw=2, color='blue')
title = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='center', va='center', fontsize=12)
def update(i):
    freq, E_x = do_fft(x, 10000*i, 10000*i+window_size)
    curve.set_data(freq/freq[1], E_x)
    title.set_text('t='+str(100*i)+'~'+str(100*i+100)+'s')
    return curve, title,
ani = animation.FuncAnimation(fig=fig, func=update, frames=int(T/window), interval=200, blit=True, repeat=False)
ani.save('%s_f-t_k%db%d.gif'%(method,k,b), writer='pillow', fps=60, dpi=300)

E = np.zeros(int(T/10))
for i in range(n-1):
    E = E + 0.5*x[2*i+1,::1000]**2 + k/2*(x[2*i,::1000]-x[2*i+2,::1000])**2 + b/3*(x[2*i,::1000]-x[2*i+2,::1000])**3
plt.clf()
plt.plot(t[::1000], E)
plt.tight_layout()
plt.savefig('%s_E-t_k%db%d.png'%(method,k,b),dpi=300)


'''

'''   

'''

'''  

'''
E = []
for i in range(100):
    freq, E_x = do_fft(x, 10000*i, 10000*i+window_size)
    E.append(np.sum(E_x))

plt.plot(t[::10000],E)
plt.savefig('E-t_plot_k%db%d_fft.png'%(k,b),dpi=300) 
'''