
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


# Tasks to do

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
            Gives the index to be used if q and dqdt are arrays
    
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

def dCdt(t,C,params_unknown, params_known,i):
    """Differential equation for CO2 concentration within reservoir
        Parameters:
    -----------
        t : floar
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
            Gives the index to be used if parameters are array-like
    
    Returns:
    --------
        dCdt : float
    """
    q_co2_interp,P0,C0,a,b,c,P = params_known
    d,M0 = params_unknown
    return (1-C)*q_co2_interp[i]/M0 -b/a/M0*(P[i]-P0)*(carbon_prime(C,P[i],P0)-C)-d*(C-C0)

def improved_euler(f,t0, t1, dt, x0, pars):
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

def plot_benchmark():
    """Plots numerical and analytical solutions for differential equations
    """
    pass

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
        y = improved_euler(f, t0, t1, h, x0, pars=pars)
        y_1975 = np.abs(t-1975.34).argmin() # Find the index of t where t = 1975.34
        y_converge.append(y[y_1975]) # Find and store the corresponding y value, ie y(t=1975.34)
        inverse_h.append(1/h) # Store the step used for this solution
    
    # Add titles etc
    ax.plot(inverse_h, y_converge, 'b.', label='Convergence Test')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Reservoir Pressure (MPa)')
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
    
    return 1/inverse_h[6]

def read_data():
    """Read in plotting data
    """
    # Import pressure values
    cs_p = np.genfromtxt('cs_p.txt', delimiter=',', skip_header=1)
    tp = cs_p[:,0]
    po = cs_p[:,1]
    # Import extraction values
    cs_q = np.genfromtxt('cs_q.txt', delimiter=',', skip_header=1)
    t_water = cs_q[:,0]
    q_water = cs_q[:,1]
    # Import injection values
    cs_c = np.genfromtxt('cs_c.txt', delimiter=',', skip_header=1)
    t_co2 = cs_c[:,0]
    q_co2 = cs_c[:,1]
    # Import concentration values
    cs_cc = np.genfromtxt('cs_cc.txt', delimiter=',', skip_header=1)
    tcc = cs_cc[:,0]
    co = cs_cc[:,1]

    # Return
    return tp, po, t_water, q_water, t_co2, q_co2, tcc, co
   
def calibration(f, t0, t1, dt, x0, params_unknown, params_known, to, yo, sigma=None):
    """Use built-in curve fitting functions to obtain satisfactory parameter values

    Parameters:
    -----------
        f : function
            Function to fit to curve
        t0 : float
            Initial time value
        t1 : float
            Final time value
        dt : float
            Time step used
        x0 : float
            Initial value for solution
        params_unknown : array-like
            Parameters to fit to curve
        params_known : array-like
            Parameters of which the value is already known
        to : array-like
            Time values of observed points/data
        yo : array-like
            Observed data points
        sigma : float
            Uncertainty in data

    Returns:
    --------
        params_fitted : array-like
            Parameters fitted to curve/best fit values
    """
    # Interpolate observed variables - this makes the shapes coherent
    t = np.arange(t0, t1+dt, dt)
    yi = np.interp(x=t, xp=to, fp=yo)
    sigma = [sigma]*len(t)
    #noise = np.random.normal(0,1,len(t))
    #sigma += noise
    # Define lambda function to hide some of our known parameters from curve_fit
    if f.__name__ == 'dPdt':
        func = lambda z, *params_tofit : improved_euler(f, t0, t1, dt, x0, pars=[params_tofit, params_known])
    else:
        func = lambda z, *params_tofit : improved_euler_concentration(f, t0, t1, dt, x0, pars=[params_tofit, params_known])
    popt, pcov = curve_fit(f=func , xdata=t, ydata=yi, p0=params_unknown)

    return popt

def forecast_model(f, t, tf, x0, params_fitted, params_known):
    """Forecasts a given model into the future, to a given time/date

    Parameters:
    -----------
        f : function
            Differential equation to be forecast
        t : array-like
            Array of time values for which true values are known (non-forecasted array)
        tf : float
            Time to be forecast to
        x0 : float
            Initial value of independent variable
        params_fitted : array-like
            Array of fitted parameters to be passed to the model
        params_known : array-like
            Array of other known parameters to be passed to the model
        
    Returns:
    --------
        t_forecast : array-like
            Array of time values extended to tf
        xf : array-like
            Array of dependent variable values extended to tf
        
    """
    # Extend time array
    dt = t[1]-t[0]
    t0 = t[-1]
    t_forecast = np.arange(t0, tf+dt, dt)

    if f.__name__ == 'dPdt':
        # Solve xf using improved euler method
        xf = improved_euler(f, t0, tf, dt, x0, pars=[params_fitted, params_known])
    else:
        # Solve xf using improved euler method
        params_known[0] = [params_known[0]]*len(t_forecast)
        xf = improved_euler_concentration(f, t0, tf, dt, x0, pars=[params_fitted, params_known])
        
    return t_forecast, xf

def plot_residuals(f, xm, xo, t, to):
    """Generate a plot showing the residuals between the model and observed data points

    Parameters:
    -----------
        f : function
            DE that is being solved
        xm : Array-like
            Array of modelled dependent variable values
        xo : Array-like
            Array of observed dependent variable values
        t : Array-like
            Array of modelled independent variable values
        to : Array-like
            Array of observed independent variable values
    """
    # Allocate array
    residuals = np.zeros(len(to))
    # Loop through the arrays finding the difference at each observed value
    for i in range(len(to)):
        j = np.abs(t-to[i]).argmin()
        residuals[i] = xm[j]-xo[i]
    # Plot residuals
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[1].plot(to, residuals, 'k.')
    ax[1].axhline(y=0,ls='--',color='r')
    ax[1].set_title('Best fit LPM model')
    ax[1].set_xlabel('Time value (year)')
    # Plot main model
    ax[0].set_xlabel('Time (years)')
    

    # Label differently depending on which quantity being modelled
    if (f.__name__ == 'dPdt'):
        ax[1].set_ylabel('Pressure misfit (MPa)')
        ax[0].plot(t,xm,'b-', label='Pressure - Model')
        ax[0].plot(to,xo,'k.', label='Pressure - Observed')
        ax[0].set_ylabel('Pressure (MPa)')
        ax[0].set_title('Comparison of model to observed \npressure values (post-calibration)')
        ax[0].legend(loc=1)
    else:
        ax[1].set_ylabel('Concentration misfit (wt%)')
        ax[0].plot(t,xm,'b-', label='Concentration - Model')
        ax[0].plot(to,xo,'k.', label='Concentration - Observed')
        ax[0].set_ylabel('Concentration (weight fraction)')
        ax[0].set_title('Comparison of model to observed \nconcentration values (post-calibration)')
        ax[0].legend(loc=2)
    
    if False:
        plt.show()
    else:
        plt.savefig('{:s}_calibrated.png'.format(f.__name__))
    
def solve_pressure(t_water, q_water, t_co2, q_co2, t0, t1, dt, po, tp):
    """Solves the pressure differential equation for the LPM

    Parameters:
    -----------
        q_water : Array-like
            Array containing values of H2O extraction
        t_water : Array-like
            Array containing corresponding time values
        q_co2 : Array-like
            Array containing values of CO2 injection
        t_co2 : Array-like
            Array containing corresponding time values
        t0 : float
            Initial time value
        t1 : float
            Final time value
        dt : float
            Time step used
        po : Array-like
            Observed pressure values

    Returns:
    --------
        pm : Array-like
            Array containing pressure measurements
        t : Array-like
            Array containing corresponding time values
        a : float
            Best fitting parameter value
        b : float
            Best fitting parameter value
        c : float
            Best fitting parameter value
    """
    t = np.arange(t0, t1+dt, dt)
    # Interpolate q terms 
    q_out = np.interp(x=t, xp=t_water, fp=q_water, left=0.0)
    q_in = np.interp(x=t, xp=t_co2, fp=q_co2, left=0)
    
    # q = q_extract - q_inject
    q = q_out - q_in

    # Initial pressure value for euler solution
    p0 = po[0]
    
    # These parameters will not be fitted, but still need to be passed to dPdt
    # I don't think this is the correct value, but we will come back to this
    p_ambient = po[0]
    params_known = [q]
    
    # Get euler solution for first attempt - initial values from notebook
    a = 2.2e-3
    b = 1.1e-1
    c = 6.8e-3

    # These parameters will be fitted later
    params_unknown = [a,b,c,p_ambient]

    pm = improved_euler(dPdt, t0, t1, dt, p0, pars=[params_unknown,params_known])

    # Conduct convergence analysis
    # We pass the average q as a float for simplicity - model misfit doesn't really matter for this step
    converged_dt = convergence(dPdt, t0, t1, dt, p0, pars=[params_unknown,[np.mean(q),p_ambient]])

    t = np.arange(t0, t1+converged_dt, converged_dt)

    # Re-Interpolate q terms according to the selected dt 
    q_out = np.interp(x=t, xp=t_water, fp=q_water, left=0.0)
    q_in = np.interp(x=t, xp=t_co2, fp=q_co2, left=0)
    
    # q = q_extract - q_inject
    q = q_out - q_in
    params_known = [q]
    
    # Just negate uncertainty for now
    uncertainty = [1]
    for sig in uncertainty:
        # Fit curve
        params_fitted = calibration(dPdt, t0, t1, converged_dt, x0=p0, params_unknown=params_unknown, params_known=params_known, to=tp, yo=po, sigma=sig)

        # Re solve with correct parameters
        pm = improved_euler(dPdt, t0, t1, converged_dt, p0, pars=[params_fitted,params_known])

        # Plot calibrated model with residuals
        plot_residuals(dPdt, pm, po, t, tp)

    fig2, ax = plt.subplots(1,1)
    # Start forecasting
    forecast = [(4.0,'g-'),(2.0,'r-'),(1.0,'b-'),(0.5,'m-')]
    # Lets say 30 years
    tf = t[-1]+30
    for n, form in forecast:
        # Calculate q
        q_inj = n*q_co2[-1]
        q_ext = q_water[-1]
        q_forecast = q_ext - q_inj
        # Forcast
        t_forecast, pf = forecast_model(dPdt, t, tf, pm[-1], params_fitted, [q_forecast,p_ambient])
        # Extend p to include forcast
        # Plot
        ax.plot(t_forecast, pf, form, label='{:.1f} times current injection rate'.format(n))

    ax.plot(t,pm,'b-')
    ax.set_ylabel('Pressure (MPa)')
    ax.set_xlabel('Time (years)')
    ax.set_title('Forecast of pressure under different injection rates')
    ax.plot(tp,po,'k.', label='Pressure - Observed')
    ax.axvline(t[-1], color='k', ls='--', lw=0.5, alpha=0.4)
    ax.legend(loc=2)

    if False:
        plt.show()
    else:
        plt.savefig('dpdt_forecast')

    a,b,c,p_ambient = params_fitted

    # Output fitted curve and best parameters
    return pm, t, a, b, c, p_ambient, converged_dt

def solve_concentration(pm,t_pm,t_cc, co, t_co2, q_co2, t0, t1, dt,pars):
    """Solves the concentration differential equation for the LPM

    Parameters:
    -----------
        pm : Array-like
            Array containing values of pressure
        t_pm : Array-like
            Array containing corresponding time values
        q_co2 : Array-like
            Array containing values of CO2 injection
        t_co2 : Array-like
            Array containing corresponding time values
        t0 : float
            Initial time value
        t1 : float
            Final time value
        dt : float
            Time step used
        co : Array-like
            Array containing values of carbon concentration
        t_cc : Array-like
            Array containing corresponding time values
        pars : array-like
            Parameters to be passed from pressure model in the form (a,b,c)
            


    Returns:
    --------
        cm : Array-like
            Array containing concentration measurements
        t : Array-like
            Array containing corresponding time values
        d : float
    """
    # Turn fraction into % for clarity
    co = co
    t = np.arange(t0, t1+dt, dt)
    # Interpolate terms 
    q_co2_interp = np.interp(x=t, xp=t_co2, fp=q_co2, left=0.0)
    
    # Initial/parameter values
    C_ambient = .03
    C0 = co[0]
    a,b,c, p_ambient = pars

    params_known = [q_co2_interp,p_ambient,C_ambient,a,b,c,pm]
    
    # Get euler solution for first attempt - random number
    d=0.2
    M0 = 10000
    # These parameters will be fitted later
    params_unknown = [d,M0]

    # Solve euler
    cm = improved_euler_concentration(dCdt, t0, t1, dt, C0, pars=[params_unknown,params_known])

    # Calibrate
    params_fitted = calibration(dCdt, t0, t1, dt, C0, params_unknown, params_known, t_cc, co)

    # Re-solve with fitted parameters
    cm = improved_euler_concentration(dCdt, t0, t1, dt, C0, pars=[params_fitted,params_known])

    # plot
    plot_residuals(dCdt, cm, co, t, t_cc)

    fig4, ax = plt.subplots(1,1)
    # Start forecasting
    forecast = [(4.0,'g-'),(2.0,'r-'),(1.0,'b-'),(0.5,'m-')]
    # Lets say 30 years
    tf = t[-1]+30
    for n, form in forecast:
        # Calculate q
        q_forecast = n*q_co2_interp[-1]
        # Forcast
        t_forecast, pf = forecast_model(dCdt, t, tf, cm[-1], params_fitted, [q_forecast,p_ambient,C_ambient,a,b,c,cm])
        # Extend p to include forcast
        # Plot
        ax.plot(t_forecast, pf, form, label='{:.1f} times current injection rate'.format(n))

    ax.plot(t,cm,'b-')
    ax.set_ylabel('Concentration (weight fraction)')
    ax.set_xlabel('Time (years)')
    ax.set_title('Forecast of concentration under different injection rates')
    ax.plot(t_cc,co,'k.', label='Concentration - Observed')
    ax.axvline(t[-1], color='k', ls='--', lw=0.5, alpha=0.4)
    ax.axhline(y=0.1, color='r', ls='-.', lw=1, alpha=0.4, label='10% limit')
    ax.legend(loc=2)

    if False:
        plt.show()
    else:
        plt.savefig('dcdt_forecast.png')

    return cm,t,d

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
        C0 : float
            Initial C value

    Returns:
    --------
        C : float
            Value of C' for the recharge rate
    """
    
    if p > p0:
        return C
    else:
        return .03

# Benchmarking
# Time calibration - convergence
# Parameter calibration - gradient descent
# Unit tests
# ------------------
# Scenario
# Posterior

def main():
    tp, po, t_water, q_water, t_co2, q_co2, t_cc, co = read_data()
    #f, ax = plt.subplots(1,1)
    #ax.plot(tp, po, 'r.')
    t0 = tp[0]
    t1 = tp[-1]
    dt = 0.5
    pm, tp, a, b, c, p_ambient, dt = solve_pressure(t_water, q_water, t_co2, q_co2, t0, t1, dt, po, tp)
    cm,t,d = solve_concentration(pm,tp,t_cc, co, t_co2, q_co2, t0, t1, dt,[a,b,c,p_ambient])
    

if __name__ == "__main__":
    main()