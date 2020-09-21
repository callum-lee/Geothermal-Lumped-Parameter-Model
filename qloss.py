import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def dq_lossdt(t, q_loss, params_known, i):
    """Differential equation for CO2 loss to groundwater
        Parameters:
        -----------
            t : float
                Time value to be used
            q_loss : float
                Corresponding CO2 loss value
            params_known : array-like
                Other parameters that do not need fitting
                Expands to (p_ambient,pm,cm,a,b)
            i : int
                Gives the index to be used if parameters are array-like
    
    Returns:
    --------
        dq_lossdt : float
    """
    p_ambient, pm, cm, a, b, = params_known
    # there are no unknown paramters
    return b*(pm[i]-p_ambient)*carbon_prime_q(cm[i],pm[i],p_ambient)/a


def carbon_prime_q(c, p, p_ambient):
    """Determines the value of c' for a given step

    Parameters:
    -----------
        c : float
            Carbon concentration
        p : float
            Pressure
        p_ambient : float
            Ambient pressure
    
    Returns:
    --------
        c_prime : float
            Value of c' at that step
    """
    if p > p_ambient:
        c_prime = c
    else:
        c_prime = 0.
    return c_prime

def improved_euler_q_loss(f, t0, t1, dt, x0, pars):
    # Allocate return arrays
    t = np.arange(t0, t1+dt, dt)
    x = np.zeros(len(t))
    x[0] = x0
    for i in range(0, (len(t)-1)):
        # Compute normal euler step
        x_temp = x[i] + dt*f(t[i], x[i], pars,i)
        # Corrector step
        x[i+1] = x[i] + (dt/2)*(f(t[i], x[i], pars,i) + f(t[i+1], x_temp, pars,i))
    return x

def solve_q_loss(pm, cm, t0, t1, dt, pars):
    """Solves the CO2 loss equation for the LPM

    Parameters:
    -----------
        pm : Array-like
            Array containing values of pressure
        cm: Array-like
            Array containing calues of concentration
        t0 : float
            Initial time value
        t1 : float
            Final time value
        dt : float
            Time step used
        pars : array-like
            Parameters to be passed from pressure model in the form (a,b,p_ambient)
            
    Returns:
    --------
        q_lossm : Array-like
            Array containing CO2 loss values
        t : Array-like
            Array containing corresponding time values
    """

    t = np.arange(t0, t1+dt, dt)
    tf = np.arange(t[-1], t[-1]+30, dt)
    t = np.append(t,tf[1:])

    #Initial parameter values
    q_loss0 = 0
    a,b,p_ambient, = pars
    
    params_known = [p_ambient, pm, cm, a, b]

    # Solve euler 
    q_lossm = improved_euler_q_loss(dq_lossdt, t0, t[-1], dt, q_loss0, params_known)


    
    return q_lossm, t  
