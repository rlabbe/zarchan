# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 22:40:55 2014

@author: rlabbe
"""
from __future__ import division, print_function
import numpy as np
import math
import matplotlib.pyplot as plt

signoise = 1000.
beta = 500
ts = 0.1 # time step, dt in my world
tf = 30.  #end time
phis = 0.
t = 0.   # time
s = 0.
h = 0.001
g = 32.2


def acc(x,vel, beta):
    return .0034*g*vel*vel*math.exp(-x/22000.)/(2.*beta) - g


def project_falling_position(X, h):
    """ 2nd order runge kutta projection of a falling object in air"""

    x = x_old = X[0]
    xd = xd_old= X[1]

    xdd = acc(x, xd, beta)
    x += xd*h
    xd += xdd*h

    xdd = acc(x, xd, beta)
    x = .5*(x_old + x + xd*h)
    xd = .5*(xd_old + xd + xdd*h)

    return (x,xd)

if __name__ == '__main__':
    pos = (200000., -6000)
    while t < tf and pos[0]>0:
        pos = project_falling_position(pos, h)
    
    
        # step time
        t += h
        s += h
        if s >= ts-0.00001:
            s = 0
    
            plt.scatter (t, pos[0])



