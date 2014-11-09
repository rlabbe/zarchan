# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 15:55:47 2014

@author: rlabbe


This is an implementation of the kalman filter used in listing 1 of chapter 8
of Zarchan's Fundamental of Kalman Filtering, Third Edition. It simulates an
object falling from high in the atmosphere where we do not know the ballistic
coefficient.

I have changed it significantly from Zarchan's implemention in that I am
using my filterpy library rather than hard coding the filter equations.

"""




import numpy as np
from numpy import zeros, array, eye, exp
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.random import randn
from math import sqrt
from kalman import ExtendedKalmanFilter
from c8_sim import simulate_fall

#np.random.seed(1234)


sig_noise = 25.
var = sig_noise**2


class DragKalmanFilter(ExtendedKalmanFilter):


    def __init__(self, dim_x, dim_z, dim_u=0, dt=0.1):
        ExtendedKalmanFilter.__init__(self, dim_x, dim_z, dim_u)
        self.dt = dt

    def predict_x(self, X):
        x    = X[0,0]
        xd   = X[1,0]
        beta = X[2,0]
        sim_dt = .001
        g = 32.2

        t = 0.
        while t <= (self.dt-.0001):
            xdd = .0034*g*xd*xd*exp(-x/22000.)/(2.*beta) - g
            xd += sim_dt * xdd
            x  += sim_dt*xd
            t  += sim_dt

        self._x[0,0] = x
        self._x[1,0] = xd


def FQ_jacobians(X):
    """ Computes F jabobian and the process noise (Q) given the state X.

    Returns
    -------
    (F,Q) : ndarray, ndarray

        F and Q matrices
    """

    x    = X[0,0]
    xd   = X[1,0]
    beta = X[2,0]

    PHI_s = 0.  # amount of error in Q

    RHOH = .0034*exp(-x/22000.)
    F21 = -32.2*RHOH*xd*xd/(44000.*beta)
    F22 = RHOH*32.2*xd/beta
    F23 = -RHOH*32.2*xd*xd/(2.*beta*beta)

    F = array([[1.,             dt,     0.],
               [F21*dt, 1 + F22*dt, F23*dt],
               [    0.,         0.,    1.]])

    Q = array([[0.,                   0.,           0.],
               [0.,  F23*F23*dt*dt*dt/3., F23*dt*dt/2.],
               [0,          F23*dt*dt/2.,           dt]]) * PHI_s

    return (F, Q)


def HJacobian(x):
    return array([[1., 0., 0,]])


def Hx(x):
    return x[0,0]


dt = .1


kf = DragKalmanFilter(dim_x=3, dim_z=1, dt=dt)

kf._x = array([[200025., -6150., 800.]]).T

kf.P = array([[var,    0,       0],
              [0, 20000.,       0],
              [0,      0, 300.**2]])

kf.R *= var
kf.H = array([[1., 0., 0,]])

# create nominal data for comparison
beta = 500.
data = simulate_fall(200000., -6000., beta, dt, sim_time=30.)


#storage for results
ArrayT = []
ArrayX = []
ArrayXH = []
ArrayXD = []
ArrayXDH = []
Arraybeta = []
ArraybetaH = []
ArrayERRX = []
ArraySP11 = []
ArraySP11P = []
ArrayERRXD = []
ArraySP22 = []
ArraySP22P = []
ArrayERRbeta = []
ArraySP33 = []
ArraySP33P = []

t = 0.
for d in data:
    kf.F, kf.Q = FQ_jacobians(kf._x)

    z = array([[d[0] + sig_noise*randn()]])
    kf.predict(kf.x)
    kf.update(z, HJacobian, Hx)


    # compute everything needed for the plots
    err_x    = d[0] - kf.x[0,0]
    err_xd   = d[1] - kf.x[1,0]
    err_beta = beta - kf.x[2,0]

    SP11 = sqrt(kf.P[0,0])  # std dev in x
    SP22 = sqrt(kf.P[1,1])  # std dev in vel
    SP33 = sqrt(kf.P[2,2])  # std dev in beta

    ArrayT.append(t)
    ArrayX.append(d[0])
    ArrayXH.append(kf.x[0,0])
    ArrayXD.append(d[1])
    ArrayXDH.append(kf.x[1,0])
    Arraybeta.append(beta)
    ArraybetaH.append(kf.x[2,0])
    ArrayERRX.append(err_x)
    ArraySP11.append(SP11)
    ArraySP11P.append(-SP11)
    ArrayERRXD.append(err_xd)
    ArraySP22.append(SP22)
    ArraySP22P.append(-SP22)
    ArrayERRbeta.append(err_beta)
    ArraySP33.append(SP33)
    ArraySP33P.append(-SP33)

    t += dt




plt.subplot(311)
plt.plot(ArrayT,ArrayERRX,ArrayT,ArraySP11,ArrayT,ArraySP11P)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Altitude (Ft)')
plt.axis([0, 30, -25, 25])

plt.subplot(312)
plt.plot(ArrayT,ArrayERRXD,ArrayT,ArraySP22,ArrayT,ArraySP22P)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Velocity (Ft/Sec)')
plt.axis([0, 30, -25, 25])

'''
plt.subplot(313)
plt.plot(ArrayT,ArrayERRbeta,ArrayT,ArraySP33,ArrayT,ArraySP33P)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Ballistic Coefficient (Lb/Ft^2)')
'''
plt.show()