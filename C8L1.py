# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 09:40:44 2014

@author: rlabbe
"""


import numpy as np
from numpy import zeros, array, eye, exp
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.random import randn
from math import sqrt


np.random.seed(1234)


def dot(a,b):
    """ ndarrays can't handle treating 1x1 array ([[1.]]) as a scalar,
    so handle it for it. sigh.
    """

    if a.shape == (1,1):
        return np.dot(a[0,0],b)
    if b.shape == (1,1):
        return np.dot(a,b[0,0])

    return np.dot(a,b)

def dot3(a,b,c):
    return dot(a,dot(b,c))


def project2(TP,dt,XP,XDP,beta_sim,HP):
    t = 0.
    x_sim = XP
    xd_sim = XDP
    sim_dt = HP
    while t <= (dt-.0001):
        XDD = .0034*32.2*xd_sim*xd_sim*exp(-x_sim/22000.)/(2.*beta_sim)-32.2
        xd_sim = xd_sim + sim_dt*XDD
        x_sim = x_sim + sim_dt*xd_sim
        t = t + sim_dt

    return x_sim, xd_sim, XDD



ITERM = 1
sig_noise = 25.

# state of simulation
x_sim = 200000.
xd_sim = -6000.
beta_sim = 500.

#state of Kalman filter
x_hat = 200025.
xd_hat = -6150.
beta_hat = 800.

order = 3
dt = .1
time_end = 30.
PHI_s = 0.  # amount of error in Q
t = 0.
sim_t = 0.
sim_dt = .001
HP = .001  # integration interval
F = zeros((order,order))


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


P = array([[sig_noise*sig_noise, 0, 0],[0, 20000., 0],[0, 0, 300.**2]])
IDNP = eye(order)
Q = zeros((order,order))
H = array([[1., 0., 0,]])
HT = H.T
R = sig_noise**2


count = 0
while t <= time_end:
    XOLD = x_sim
    XDOLD = xd_sim
    XDD = .0034*32.2*xd_sim*xd_sim*exp(-x_sim/22000.)/(2.*beta_sim)-32.2
    x_sim = x_sim + sim_dt*xd_sim
    xd_sim = xd_sim + sim_dt*XDD
    t = t + sim_dt
    XDD = .0034*32.2*xd_sim*xd_sim*exp(-x_sim/22000.)/(2.*beta_sim)-32.2
    x_sim = .5*(XOLD + x_sim + sim_dt*xd_sim)
    xd_sim = .5*(XDOLD + xd_sim + sim_dt*XDD)
    sim_t = sim_t + sim_dt
    if sim_t >= (dt-.00001):
        sim_t = 0.
        RHOH = .0034*exp(-x_hat/22000.)
        F21 = -32.2*RHOH*xd_hat*xd_hat/(44000.*beta_hat)
        F22 = RHOH*32.2*xd_hat/beta_hat
        F23 = -RHOH*32.2*xd_hat*xd_hat/(2.*beta_hat*beta_hat)
        if ITERM == 1:
            F[0,0] = 1.
            F[0,1] = dt
            F[1,0] = F21*dt
            F[1,1] = 1. + F22*dt
            F[1,2] = F23*dt
            F[2,2] = 1.
        else:
            F[0,0] = 1. + .5*dt*dt*F21
            F[0,1] = dt + .5*dt*dt*F22
            F[0,2] = .5*dt*dt*F23
            F[1,0] = F21*dt + .5*dt*dt*F22*F21
            F[1,1] = 1. + F22*dt + .5*dt*dt*(F21 + F22*F22)
            F[1,2] = F23*dt + .5*dt*dt*F22*F23
            F[2,2] = 1.

        Q[1,1] = F23*F23*dt*dt*dt/3.
        Q[1,2] = F23*dt*dt/2.
        Q[2,1] = Q[1,2]
        Q[2,2] = dt
        Q *= PHI_s


        #predict
        XNOISE = sig_noise*randn()
        (XB,XDB,XBETA) = project2(t,dt,x_hat,xd_hat,beta_hat,HP)
        print ('xpre',XB,XDB,XBETA)
        M = dot3(F, P, F.T) + Q

        # update
        S = dot3 (H, M, H.T) + R
        gain = dot3(M, HT, inv(S))
        IKH = IDNP-dot(gain, H)
        P = dot(IKH, M)
        #print ('P=', P)

        residual = x_sim + XNOISE-XB

        # X state [x_sim, x_sim', beta_sim]
        x_hat = XB + gain[0,0]*residual
        xd_hat = XDB + gain[1,0]*residual
        beta_hat = beta_hat + gain[2,0]*residual


        count += 1
        #if count == 5:
        #    break

        print (x_hat, xd_hat, beta_hat)
        err_x = x_sim-x_hat
        SP11 = sqrt(P[0,0])
        err_xd = xd_sim-xd_hat
        SP22 = sqrt(P[1,1])
        err_beta = beta_sim-beta_hat
        SP33 = sqrt(P[2,2])
        SP11P = -SP11
        SP22P = -SP22
        SP33P = -SP33


        ArrayT.append(t)
        ArrayX.append(x_sim)
        ArrayXH.append(x_hat)
        ArrayXD.append(xd_sim)
        ArrayXDH.append(xd_hat)
        Arraybeta.append(beta_sim)
        ArraybetaH.append(beta_hat)
        ArrayERRX.append(err_x)
        ArraySP11.append(SP11)
        ArraySP11P.append(-SP11)
        ArrayERRXD.append(err_xd)
        ArraySP22.append(SP22)
        ArraySP22P.append(-SP22)
        ArrayERRbeta.append(err_beta)
        ArraySP33.append(SP33)
        ArraySP33P.append(-SP33)




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

plt.subplot(313)
plt.plot(ArrayT,ArrayERRbeta,ArrayT,ArraySP33,ArrayT,ArraySP33P)
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Ballistic Coefficient (Lb/Ft^2)')

plt.show()
