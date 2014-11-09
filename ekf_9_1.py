# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 07:51:06 2014

@author: RL
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 22:40:55 2014

@author: rlabbe
"""
from __future__ import division, print_function
from filterpy.kalman import KalmanFilter
import numpy as np
import math
import numpy.random as random
import matplotlib.pyplot as plt




""" this is the filter for the projectile problem, listing 9.1"""




ts = 0.1 # time step, dt in my world
order = 4
tf = 30.  #end time
phis = 0. # spectral noise in Q. set to 100 to avoid divergence in filter
sigth = 0.01  # std dev of the angle theta
sigr=100.     # std dev of the distance
vt=3000.
gamdeg=45.
gamrad = radians(gamdeg)
g = 32.2
xt=0.
yt = 0.
xtd = vt*cos(gamrad)
ytd = vt*sin(gamrad)
xr = 100000.
yr = 0.
t = 0.   # time
s = 0.
h = 0.001


ekf = kf.KalmanFilter(dim_x=4, dim_z=2)

ekf.F = np.eye(order)  # PHI
ekf.x = np.array([[200025],[-6150.]])fuck #xh, xdh
ekf.Q = np.zeros((4,4))
ekf.P = np.zeros((4,4))
ekf.P[0,0] = 1000.**2
ekf.P[1,1] = 100.**2
ekf.P[2,2] = 1000.**2
ekf.P[3,3] = 100.**2


ts2 = ts*ts
ts3 = ts2*ts

ekf.Q[0,0] = np.array([
   [phis*ts3/3, phis*ts2/2, 0.        , 0.],
   [0.        , phis*ts2/2, 0.        , 0.],
   [0.        , 0.        , phis*ts3/3, phis*ts2/2],
   [0.        , 0.        , 0.        , phis*ts2/2]])

ekf.F[0,1] = ts
ekf.F[2,3] = ts
ekf.H = np.array([[1., 0.]])
ekf.R = np.array([[sigth**2, 0],
                  [0, sigr**2]])




tch = xt+1000
xtdh = xtd-100



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


def project_falling_position(X, Y, h):
    """ 2nd order runge kutta projection of a falling object in air"""

    xt = xt_old = X[0]
    xd = xd_old= X[1]
    xtdd = 0

    yt = ytold = Y[0]
    ytdold = Y[1]
    ytdd = -G

    xt  += h*xtd
    xtd += h*xtdd
    yt  += h*ytd
    ytd += h*ytdd

    xt  = .5 * (xtold  + xt  + h*xtd)
    xtd = .5 * (xtdold + xtd + h*xtdd)
    yt  = .5 * (ytold  + yt  + h*ytd)
    ytd = .5 * (ytdold + ytd + h*ytdd)


    return (xt, xtd, yt, ytd)



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
        ekf.update(z)A
        fs.append(ekf.x[0,0])

        fzs.append(z)





times = [i*ts for i in range(len(fs))]
p1 = plt.scatter (times, fzs,color='red')
p2, = plt.plot(times, fs, 'green')
p3, = plt.plot(times, poss, 'blue')

plt.legend([p1,p2,p3], ['measurements', 'filter', 'actual'],1)










