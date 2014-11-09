# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 18:46:59 2014

@author: rlabbe


This is an implementation of the simulation used in chapter 8 of Zarchan's
Fundamental of Kalman Filtering, Third Edition. It simulates an object
falling from high in the atmosphere.

"""

from math import exp
from numpy import array, asarray

def simulate_fall(x, xd, beta, step, sim_time):
    """ simulate an object falling in air.

    Parameters
    ----------
    x : float
        altitude

    xd : float
        velocity (derivative x)

    beta : float
        ballistic coefficient of falling object

    step : float
        time interval in seconds to produce data records for

    sim_time : float
        simulation time in seconds


    Returns

    data : ndarray[ ndarray[float, float]]
       array of [altitude, velocity] arrays
    """

    data = []

    t = 0.
    sim_t = 0.
    dt = 0.001

    assert dt < step

    time_end = 30.
    while t <= time_end:
        x_old = x
        xd_old = xd

        xdd = .0034*32.2*xd*xd*exp(-x/22000.)/(2.*beta)-32.2
        x  += dt*xd
        xd += dt*xdd
        t  += dt

        xdd = .0034*32.2*xd*xd*exp(-x/22000.)/(2.*beta)-32.2
        x   = .5*(x_old  + x  + dt*xd)
        xd  = .5*(xd_old + xd + dt*xdd)

        sim_t += dt
        if sim_t >= step - .00001:
            data.append(array([x, xd]))
            sim_t = 0.
    return asarray(data)
