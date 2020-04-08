# -*- coding: utf-8 -*-
"""
Created on 28.3.20

@author: mw141
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#plt.ion() # to see immediate output.

# if you want latex mathematical lables on the figure, uncomment the lines below:
# from matplotlib import rc
# rc('font', family = 'serif')
# rc('text',usetex = True)


def driver(x,t,eta):
    """
    driver() is on rhs and equals the derivative vector on the lhs
    n' = (1-n-m)n - eta n
    m' = eta n
    scaled form of infection-immune equations
    n = x[0], m = x[1]
    """
    return [(1 - x[0] -x[1]) * x[0] - eta * x[0], eta * x[0]]

# jacobian matrix of derivatives helps ODE solver
# best to put in analytically, as here, rather than rely on numerical estimates
# of gradients about points in the 2-space
def jacob(x,t,eta):
    return [[(1-x[1]) - 2*x[0]-eta, -x[0]],[eta, 0]]

# the one parameter left after scaling
etas = [0.1, 0.25, 0.33, 0.5, 0.75, 1, 1.1]

for eta in etas:
    # array of points in time to get x[0] and x[1] at
    t = np.linspace(0, 20, 2001)

    # initial condition n,m: n small seed, m = 0 (no one immune yet)
    x0 = np.array([0.001, 0])

    # integrate the differential equation x' = driver(x) with x a vector in array evaluated at all the t-points
    sol = odeint(driver, x0, t, args=(eta,), Dfun=jacob)
     # unpack the solution array sol into its component arrays
    n = sol[:,0] # at the various t-points
    m = sol[:,1]

    t_nonzero_m = t[m > 0]
    log_m = np.log(m[m > 0])

    # plot on same graph
    # plt.plot(t,n,color='r',label='n(t)') # comment out to not plot n - fraction infected
    plt.plot(t_nonzero_m, log_m, label=eta) # log(m) is log of immune ie log of proxy for dead
    # when plotting Log(m) add 0.001 to m to avoid m=0 problem. Better to start plotting at t > 0
    # put legend on:
    plt.legend(loc='best')
    # put axis labels on:
    plt.xlabel('t', style='italic')
    plt.ylabel('n, log(m)', style='italic')
    plt.ylim([-6,0])

#plt.savefig('C:\\Users\\warne\\Documents\\Python Scripts\\infected_immune_t.pdf')
# plt.savefig('D:\\Users\\mw141\\Documents\\Python Scripts\\infected_immune_t.pdf')'
plt.show()