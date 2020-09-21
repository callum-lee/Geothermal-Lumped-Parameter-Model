import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from pressure_functions import *
from concentration_functions import *


def convergence(f, t0, t1, dt, x0, pars):
    """Conducts convergence analysis for numerical solution

    Parameters:
    -----------
        f : function
            Function to test convergence on
        t0 : float
            Initial time value
        t1 : float
            Final time value
        dt : float
            Time step used
        x0 : float 
            Initial y value
        pars : Array-like
            Parameters to be passed to the function
            Ensure that these are passed in the correct order

    Notes:
    ------
        For simplicity, we will use q as a constant in this analysis
    """
    # Get figure
    fig, ax = plt.subplots(1,1)
    # Create loop to define a series of different step sizes (h) and plot
    y_converge = []
    inverse_h = []
    for i in range(2,100,4):
        h = 1/i
        t = np.arange(t0, t1+h, h)
        y = improved_euler_pressure(f, t0, t1, h, x0, pars=pars)
        y_1975 = np.abs(t-1975.34).argmin() # Find the index of t where t = 1975.34
        y_converge.append(y[y_1975]) # Find and store the corresponding y value, ie y(t=1975.34)
        inverse_h.append(1/h) # Store the step used for this solution
    
    # Add titles etc
    ax.plot(inverse_h, y_converge, 'b.', label='Convergence Test')
    ax.set_title('Convergence Test for Improved Euler')
    ax.set_xlabel('1/h')
    ax.set_ylabel('y(t=1975)')

    # Add annotation denoting appropriate step size h
    ax.text(26, 4.5098+0.0001, 'Any step size beyond here \nproduces a similar enough result \nto be considered converged', ha='center', va='bottom', size=10, color='r')
    ax.arrow(26, 4.5098+0.0001, 0, -0.0001, length_includes_head=True, head_width=2, head_length=0.000025, color='r')
    
    # Make sure all the axis labels etc fit on figure
    plt.tight_layout()

    # Show or save
    if False:
        plt.show()
    else:
        plt.savefig('dpdt_convergence.png')
        plt.close()
    
    return 1/inverse_h[6]


def benchmark_pressure(t0,t1,dt,x0,pars):
    """Benchmarks the pressure differential equation with the analytical one

        Parameters:
        -----------
        t0 : float
            Initial time value
        t1 : float
            Final time value
        dt : float
            Time step used
        x0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to dPdt function via the improved_euler function.

        Returns:
        --------
        x_analytical : array-like
            Dependent variable solution vector of the analytical solution.
        x_euler : array-like
            Dependent variable solution vector of the numerical solution with the improved Euler's method.

        Notes:
        ------
        Pars should be in the form [params_unknown, params_known] where
            params_unknown = [a,b,c]
            params_known = [q,P0]
        For the numerical solution to be valid, we are assuming a constant q value of only co2 injection where q = q_water-q_co2 = 0-q_co2 = -q_co2
    """
    #initialise time array
    t=np.arange(t0,t1+dt,dt)
    t_analytical=t-t0
    #unpack parameters
    params_unknown, params_known = pars
    q = params_known[0]
    a,b,c,P0 = params_unknown
    #compute solutions
    x_analytical=P0-a*q/b*(1-np.exp(-b*t_analytical))
    x_eulers=improved_euler_pressure(dPdt,t0,t1,dt,x0,pars)
    #plot

    f,ax = plt.subplots(1,1)
    ax.plot(t,x_analytical,'b-', label='Analytical Solution')
    ax.plot(t,x_eulers,'k.', label='Numerical Solution (Improved Euler\'s)')
    plt.xlabel('Time (Years)')
    plt.ylabel('Pressure (MPa)')
    plt.title('Benchmark of the Analytical and Numerical \nSolutions of the Pressure ODE')
    ax.legend()
    if False:
        plt.show()
    else:
        plt.savefig('pressure_benchmark.png')
        plt.close()
    return x_analytical,x_eulers



def benchmark_concentration(t0,t1,dt,x0,pars):  
    """Benchmarks the concentration differential equation with the analytical one

        Parameters:
        -----------
        t0 : float
            Initial time value
        t1 : float
            Final time value
        dt : float
            Time step used
        x0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to dPdt function via the improved_euler function.

        Returns:
        --------
        x_analytical : array-like
            Dependent variable solution vector of the analytical solution.
        x_euler : array-like
            Dependent variable solution vector of the numerical solution with the improved Euler's method.

        Notes:
        ------
        Pars should be in the form [params_unknown, params_known] where
            params_unknown = [d,M0]
            params_known = [q_co2_interp,p0,C0_ambient,a,b,c,pm]
        For the numerical solution to be valid, we are assuming p>p0 is always true as a result of the assumptions in the pressure benchmark, i.e. q_co2 is constant
    """
    #initialise time array
    t=np.arange(t0,t1+dt,dt)
    t_analytical=t-t0
    #unpack parameters
    params_unknown, params_known = pars
    q_co2,p0,C0_ambient,a,b,c,pm = params_known
    d,M0 = params_unknown
    #solve analytically
    x_analytical=(q_co2+M0*d*C0_ambient)/(q_co2+d*M0)+q_co2*(x0-1)/(q_co2+d*M0)*np.exp(-(q_co2/M0+d)*t_analytical)
    #interpolate q_co2 into an array to fit the improved_euler_concentration function input requirements
    q_co2_interp = np.full(len(t),q_co2)
    #solve numerically
    x_eulers=improved_euler_concentration(dCdt,t0,t1,dt,x0,[[d,M0],[q_co2_interp,p0,C0_ambient,a,b,c,pm]])
    #plot
    f,ax = plt.subplots(1,1)
    ax.plot(t,x_analytical,'b-', label='Analytical Solution')
    ax.plot(t,x_eulers,'k.', label='Numerical Solution (Improved Euler\'s)')
    plt.xlabel('Time (Years)')
    plt.ylabel('Concentration of CO2 (wt%)')
    plt.title('Benchmark of the Analytical and Numerical \nSolutions of the Concentration ODE')
    ax.legend()
    if False:
        plt.show()
    else:
        plt.savefig('concentration_benchmark.png')
        plt.close()
    return x_analytical,x_eulers

