import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def dPdt(t, P, params_unknown, params_known, i=None):
    """Define differential equation for reservoir pressure

    Parameters:
    -----------
        t : float
            Time value to be used
        P : float
            Corresponding pressure value
        params_unknown : array-like
            Values of parameters that will later be fit to curve
            Expands to (a,b,c)
        params_known : array-like
            Other parameters that do not need fitting
            Expands to (q,P0,dqdt)
        i : int, optional
            Gives the index for the corresponding q and dqdt values. Only used if parameters are array-like.  
    
    Returns:
    --------
        dPdt : float
    """
    q,dqdt = params_known
    a,b,c,P0 = params_unknown
    if i is not None:
        return -a*q[i] - b*(P-P0) - c*dqdt[i]
    else:
        return -a*q - b*(P-P0) - c*dqdt

def improved_euler_pressure(f,t0, t1, dt, x0, pars):
    """Solve an ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time value
        t1 : float
            Final time value
        dt : float
            Time step used
        x0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to ODE function f.

        Returns:
        --------
        x : array-like
            Dependent variable solution vector.

        Notes:
        ------
        Assume that ODE function f takes the following inputs, in order:
            1. independent variable
            2. dependent variable
            3. Parameters that will later be passed to the curve fitting function
            4. all other parameters
            Refer to the function definitions for order within (3) and (4)
            5. Optional - counter number to be passed to array parameters
    """
    # Allocate return arrays
    t = np.arange(t0, t1+dt, dt)
    params_unknown, params_known = pars
    x = np.zeros(len(t))
    x[0] = x0
    i=0
    # Check if q is iterable or const
    if isinstance(params_known[0], float) == True:
        dqdt = 0
        if len(params_known) != 2:
            params_known.append(dqdt)
        # Loop through time values, finding corresponding x value
        for i in range(0, (len(t) - 1)):
            # Compute normal euler step
            x_temp = x[i] + dt*f(t[i], x[i], params_unknown, params_known)
            # Corrector step
            x[i+1] = x[i] + (dt/2)*(f(t[i], x[i], params_unknown, params_known) + f(t[i+1], x_temp, params_unknown, params_known))
    else:
        # Get dqdt and append to known parameters
        dqdt = np.gradient(params_known[0])
        if len(params_known) != 2:
            params_known.append(dqdt)
        # Loop through time values, finding corresponding x value
        for i in range(0, (len(t) - 1)):
            # Compute normal euler step
            x_temp = x[i] + dt*f(t[i], x[i], params_unknown, params_known, i=i)
            # Corrector step
            x[i+1] = x[i] + (dt/2)*(f(t[i], x[i], params_unknown, params_known, i=i) + f(t[i+1], x_temp, params_unknown, params_known, i=i))
        
    return x

def model_ensemble(samples, t, params_known, yo, to):
    """Compute amd plot the model for a set of parameter samples

    Parameters:
    -----------
        samples : array-like
            List of parameter samples from the multivariate normal
        t : array-like
            Time array over which to model the samples
        params_known : array like
            Known parameters with fixed values (to be passed to improved_euler)
        yo : array-like
            Observed dependent variable values
        to : array-like
            Corresponding time values for yo
        
    Returns:
    --------


    """
    f, ax = plt.subplots(1,1, figsize=(10,6))
    p_samples = []

    # Loop through samples
    for params_fitted in samples:
        # Solve for given parameter set
        pm = improved_euler_pressure(dPdt, t[0], t[-1], (t[1]-t[0]), yo[0], pars=[params_fitted, params_known])
        # Append the sample so we can forecast from it
        p_samples.append(pm)
        # Plot
        ax.plot(t, pm, 'k-', lw=0.2, alpha=0.2)

    ax.plot([],[], 'k-', label='Model Ensemble')

    ax.plot(to,yo,'r.', label='Observed Data')

    # Add titles etc
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Reservoir Pressure (MPa)')
    ax.set_title('Model Posterior for Reservoir Pressure')
    ax.legend(loc=1)

    # Show or save
    if False:
        plt.show()
    else:
        #plt.savefig('dpdt_ensemble.png')
        return ax, p_samples