# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 22:40:55 2014

@author: rlabbe
"""
from __future__ import division, print_function
import filterpy.kalman as kf
import numpy as np
import math
import numpy.random as random
import matplotlib.pyplot as plt
from ekf_7_1 import project_falling_position, acc



""" This is the first EKF in the text. It diverges for reasons discussed
in the text. It is not a good filter."""


signoise = 1000.
beta = 500
ts = 0.1 # time step, dt in my world
tf = 30.  #end time
phis = 0. # spectral noise in Q. set to 100 to avoid divergence in filter
t = 0.   # time
s = 0.
h = 0.001
g = 32.2

ekf = kf.KalmanFilter(dim_x=2,dim_z=1)

ekf.F = np.zeros((2,2))  # PHI
ekf.x = np.array([[200025],[-6150.]]) #xh, xdh
ekf.Q = np.zeros((2,2))
ekf.P = np.array([[signoise**2, 0.],
                  [0, 20000.]])

ekf.H = np.array([[1., 0.]])



pos = (200000., -6000.)
poss = []
fzs = []
fs = []


def project (ekf, tp, ts, beta):
    """ replaces the predict step of the kalman filter. Fixes the divergence
    problem in Zarchan's filter. This is equivelent to the code in listing
    7.5 """
    
    t = 0
    x = ekf.x[0,0]
    xd = ekf.x[1,0]
    H = .001
    while t <= ts-0.0001:
        xdd = acc(x, xd, beta)
        xd += H*xdd
        x  += H*xd
        t += h
        
    ekf.x[0,0] = x
    ekf.x[1,0] = xd



while t < tf:
    pos = project_falling_position(pos, h)
    # step time
    t += h
    s += h
    if s >= ts-0.00001:
        poss.append(pos[0])
        print(t)
        s = 0

        x   = ekf.x[0,0]
        vel = ekf.x[1,0]

        rho_h = .0034*math.exp(-x/22000.)
        f21 = -g*rho_h*vel*vel/(44000*beta)
        f22 = rho_h * g * vel/beta

        ekf.F = np.array([[1., ts],
                         [f21*ts, 1 + f22*ts]])


        ts_3 = (ts**3) / 3
        ts_2 = (ts**2) / 2


        ekf.Q[0,0] = ts_3
        ekf.Q[0,1] = ts_2 + f22*ts_3
        ekf.Q[1,0] = ekf.Q[0,1]
        ekf.Q[1,1] = ts + f22*(ts**2) + (f22**2) * ts_3
        ekf.Q *= phis

        ekf.predict()
        # to fix divergence, comment call ekf.predict(), and uncomment
        # line below.
        #project(ekf, t, ts, beta)

        
        z = np.array([[pos[0] + random.randn()*signoise]])
        ekf.update(z)
        fs.append(ekf.x[0,0])

        fzs.append(z)



        

times = [i*ts for i in range(len(fs))]
p1 = plt.scatter (times, fzs,color='red')
p2, = plt.plot(times, fs, 'green')
p3, = plt.plot(times, poss, 'blue')

plt.legend([p1,p2,p3], ['measurements', 'filter', 'actual'],1)










