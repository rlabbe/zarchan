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


signoise = 1000.
beta = 500
ts = 0.1 # time step, dt in my world
tf = 50.  #end time
phis = 0.
t = 0.   # time
s = 0.
h = 0.001
g = 32.2

ukf = kf.KalmanFilter(dim_x=2,dim_z=1)

ukf.F = np.zeros((2,2))  # PHI
ukf.x = np.array([[200025],[-6150.]]) #xh, xdh
ukf.Q = np.zeros((2,2))
ukf.P = np.array([[signoise**2, 0.],
                  [0, 20000.]])

ukf.H = np.array([[1., 0.]])


def acc(x,vel):
    return .0034*g*vel*vel*math.exp(-x/22000.)/(2.*beta) - g



def project_falling_position(X, h):
    """ 2nd order runge kutta projection of a falling object in air"""

    x = x_old = X[0]
    xd = xd_old= X[1]

    xdd = acc(x,xd)
    x += xd*h
    xd += xdd*h

    xdd = acc(x,xd)
    x = .5*(x_old + x + xd*h)
    xd = .5*(xd_old + xd + xdd*h)

    return (x,xd)



pos = (200000., -6000.)
poss = []
fzs = []
fs = []

while t < tf:
    pos = project_falling_position(pos, h)
    # step time
    t += h
    s += h
    if s >= ts-0.00001:
        poss.append(pos[0])
        print(t)
        s = 0

        x   = ukf.x[0,0]
        vel = ukf.x[1,0]

        rho_h = .0034*math.exp(-x/22000.)
        f21 = -g*rho_h*vel*vel/(44000*beta)
        f22 = rho_h * g * vel/beta

        ukf.F = np.array([[1., ts],
                         [f21*ts, 1 + f22*ts]])


        ts_3 = (ts**3) / 3
        ts_2 = (ts**2) / 2


        ukf.Q[0,0] = ts_3
        ukf.Q[0,1] = ts_2 + f22*ts_3
        ukf.Q[1,0] = ukf.Q[0,1]
        ukf.Q[1,1] = ts + f22*(ts**2) + (f22**2) * ts_3
        ukf.Q *= phis

        ukf.predict()
        z = np.array([[pos[0] + random.randn()*30000.]])
        ukf.update(z)
        fs.append(ukf.x[0,0])

        fzs.append(z)



times = [x/ts for x in range(len(poss))]
p1 = plt.scatter (times, fzs,color='red')
p2, = plt.plot(times, fs, 'green')
p3, = plt.plot(times, poss, 'blue')

plt.legend([p1,p2,p3], ['measurements', 'filter', 'actual'],1)










