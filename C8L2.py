# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 19:26:49 2014

@author: rlabbe
"""
import numpy as np
from numpy import zeros, array, eye, exp
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy. random import randn
from math import sqrt


def dot(a,b):
    if a.shape == (1,1):
        return np.dot(a[0,0],b)
    if b.shape == (1,1):
        return np.dot(a,b[0,0])
    
    return np.dot(a,b)
    
    
    
def project2(TP,TS,XP,XDP,BETA,HP):
    T = 0.
    X = XP
    XD = XDP
    H = HP
    while T <= (TS-.0001):
        XDD = .0034*32.2*XD*XD*exp(-X/22000.)/(2.*BETA)-32.2
        XD = XD+H*XDD
        X = X+H*XD
        T = T+H

    return X, XD, XDD


ITERM = 1
G = 32.2
SIGNOISE = 25.
X = 200000.
XD = -6000.
BETA = 500.
XH = 200025.
XDH = -6150.
BETAH = 800.
BETAINV = 1./BETA
BETAINVH = 1./BETAH
ORDER = 3
TS = .1
TF = 30.
PHIS = 0.
T = 0.
S = 0.
H = .001
HP = .001
PHI = zeros((ORDER,ORDER))

P = array ([[SIGNOISE*SIGNOISE, 0, 0],
           [0, 20000., 0],
           [0, 0, (BETAINV-BETAINVH)**2]])
           
IDNP = eye(ORDER)
Q = zeros((ORDER,ORDER))
HMAT = array([[1, 0, 0]])
HT = HMAT.T
RMAT = SIGNOISE**2


ArrayT = []   
ArrayX = []
ArrayXH = []
ArrayXD = []
ArrayXDH = []
ArrayBETA = []
ArrayBETAH = []
ArrayERRX = []
ArraySP11 = []
ArraySP11P = []
ArrayERRXD = []
ArraySP22 = []
ArraySP22P = []
ArrayERRBETAINV = []
ArraySP33 = []
ArraySP33P = []
count = 0

while T <= TF: 
    XOLD = X
    XDOLD = XD
    XDD = .0034*32.2*XD*XD*exp(-X/22000.)/(2.*BETA)-32.2
    X = X+H*XD
    XD = XD+H*XDD
    T = T+H
    XDD = .0034*32.2*XD*XD*exp(-X/22000.)/(2.*BETA)-32.2
    X = .5*(XOLD+X+H*XD)
    XD = .5*(XDOLD+XD+H*XDD)
    S = S+H
    if S >= (TS-.00001):
        S = 0.
        RHOH = .0034*exp(-XH/22000.)
        F21 = -G*RHOH*XDH*XDH*BETAINVH/44000.
        F22 = RHOH*G*XDH*BETAINVH
        F23 = .5*RHOH*XDH*XDH*G
        PHI[0,0] = 1.
        PHI[0,1] = TS
        PHI[1,0] = F21*TS
        PHI[1,1] = 1.+F22*TS
        PHI[1,2] = F23*TS
        PHI[2,2] = 1.
        Q[1,1] = F23*F23*PHIS*TS*TS*TS/3.
        Q[1,2] = F23*PHIS*TS*TS/2.
        Q[2,1] = Q[1,2]
        Q[2,2] = PHIS*TS
        PHIT = PHI.T
        PHIP = dot(PHI,P)
        PHIPPHIT =dot(PHIP,PHIT)
        M = PHIPPHIT+Q
        HM = dot(HMAT,M)
        HMHT = dot(HM,HT)
        HMHTR = HMHT+RMAT
        HMHTRINV = inv(HMHTR)
        MHT = dot(M,HT)
        GAIN = dot(HMHTRINV,MHT)
        KH = dot(GAIN,HMAT)
        IKH = IDNP-KH
        P = dot(IKH,M)
        XNOISE = SIGNOISE*randn()
        BETAH = 1./BETAINVH
        (XB,XDB,XDDB) = project2(T,TS,XH,XDH,BETAH,HP)
        RES = X+XNOISE-XB
        XH = XB+GAIN[0,0]*RES
        XDH = XDB+GAIN[1,0]*RES
        BETAINVH = BETAINVH+GAIN[2,0]*RES
        ERRX = X-XH
        SP11 = sqrt(P[0,0])
        ERRXD = XD-XDH
        SP22 = sqrt(P[1,1])
        ERRBETAINV = 1./BETA-BETAINVH
        SP33 = sqrt(P[2,2])
        BETAH = 1./BETAINVH
        SP11P = -SP11
        SP22P = -SP22
        SP33P = -SP33
        count = count+1
        ArrayT.append(T)
        ArrayX.append(X)
        ArrayXH.append(XH)
        ArrayXD.append(XD)
        ArrayXDH.append(XDH)
        ArrayBETA.append(BETA)
        ArrayBETAH.append(BETAH)
        ArrayERRX.append(ERRX)
        ArraySP11.append(SP11)
        ArraySP11P.append(SP11P)
        ArrayERRXD.append(ERRXD)
        ArraySP22.append(SP22)
        ArraySP22P.append(SP22P)
        ArrayERRBETAINV.append(ERRBETAINV)
        ArraySP33.append(SP33)
        ArraySP33P.append(SP33P)



plt.plot(ArrayT,ArrayERRX,ArrayT,ArraySP11,ArrayT,ArraySP11P)#,grid
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Altitude (Ft)')
plt.axis([0, 30, -25, 25])
plt.figure ()
plt.plot(ArrayT,ArrayERRXD,ArrayT,ArraySP22,ArrayT,ArraySP22P)#,grid
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of Velocity (Ft/Sec)')
plt.axis([0, 30, -25, 25])
plt.figure()
plt.plot(ArrayT,ArrayERRBETAINV,ArrayT,ArraySP33,ArrayT,ArraySP33P)#,grid
plt.xlabel('Time (Sec)')
plt.ylabel('Error in Estimate of 1/BETA (Ft^2/Lb)')
plt.axis([0, 30, -.0008, .0008])


