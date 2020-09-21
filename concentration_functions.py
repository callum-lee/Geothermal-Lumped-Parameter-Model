import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def dCdt(t,C,params_unknown, params_known, i):
    """Differential equation for CO2 concentration within reservoir

    Parameters:
    -----------
        t : float
            Time value to be used
        C : float
            Corresponding concentration value
        params_unknown : array-like
            Values of parameters that will later be fit to curve
            Expands to (d,M0)
        params_known : array-like
            Other parameters that do not need fitting
            Expands to (q_co2,P0,C0,a,b,c,P)
        i : int
            Gives the index of the parameters. 
    
    Returns:
    --------
        dCdt : float

    Notes:
    ----
        The parameter 'c' in params_known is not used, but we pass it with a and b. The

    """
    q_co2_interp,P0,C0,a,b,c,P = params_known
    d,M0 = params_unknown
    return (1-C)*q_co2_interp[i]/M0 -b/a/M0*(P[i]-P0)*(carbon_prime(C,P[i],P0)-C)-d*(C-C0)

def improved_euler_concentration(f, t0, t1, dt, x0, pars):
    # Allocate return arrays
    t = np.arange(t0, t1+dt, dt)
    params_unknown, params_known = pars
    x = np.zeros(len(t))
    x[0] = x0
    for i in range(0, (len(t) - 1)):
        # Compute normal euler step
        x_temp = x[i] + dt*f(t[i], x[i], params_unknown, params_known,i)
        # Corrector step
        x[i+1] = x[i] + (dt/2)*(f(t[i], x[i], params_unknown, params_known,i) + f(t[i+1], x_temp, params_unknown, params_known,i))
    return x

def carbon_prime(C,p,p0):
    """Outputs the C' value for the carbon diffential equation

    Parameters:
    -----------
        C : float
            current C value
        p : float
            current pressure value
        p0 : float
            Initial pressure value

    Returns:
    --------
        C : float
            Value of C' for the recharge rate
    """
    
    if p > p0:
        return C
    else:
        return .03

def model_ensemble_conc(samples, t, params_known, yo, to):
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

    """
    f, ax = plt.subplots(1,1)
    c_samples = []

    # Loop through samples
    for params_fitted in samples:
        # Solve for given parameter set
        cm = improved_euler_concentration(dCdt, t[0], t[-1], (t[1]-t[0]), yo[0], pars=[params_fitted, params_known])
        # Append sample so we can forecast from it
        c_samples.append(cm)
        # Plot
        ax.plot(t, cm, 'k-', lw=0.2, alpha=0.2)

    ax.plot([],[], 'k-', label='Model Ensemble')

    ax.plot(to,yo,'r.', label='Observed Data')

    # Add titles etc
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('CO2 Concentration (wt%)')
    ax.set_title('Model Posterior for Reservoir CO2 Concentration')
    ax.legend(loc=2)

    # Show or save
    if False:
        plt.show()
    else:
        #plt.savefig('dcdt_ensemble.png')
        return ax, c_samples