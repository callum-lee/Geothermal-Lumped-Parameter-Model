import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from pressure_functions import *
from concentration_functions import *
from functions_for_plotting import *
from benchmarking import *
from qloss import *

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
        pcov : array-like
            Covariance matrix for fitted parameters
    """
    # Interpolate observed variables - this makes the shapes coherent
    t = np.arange(t0, t1+dt, dt)
    yi = np.interp(x=t, xp=to, fp=yo)
    sigma = [sigma]*len(t)

    # Define lambda function to hide some of our known parameters from curve_fit
    if f.__name__ == 'dPdt':
        func = lambda z, *params_tofit : improved_euler_pressure(f, t0, t1, dt, x0, pars=[params_tofit, params_known])
    else:
        func = lambda z, *params_tofit : improved_euler_concentration(f, t0, t1, dt, x0, pars=[params_tofit, params_known])
    popt, pcov = curve_fit(f=func , xdata=t, ydata=yi, p0=params_unknown, sigma=sigma, absolute_sigma=True)

    return popt, pcov

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
        xf = improved_euler_pressure(f, t0, tf, dt, x0, pars=[params_fitted, params_known])
    else:
        # Solve xf using improved euler method
        params_known[0] = [params_known[0]]*len(t_forecast)
        xf = improved_euler_concentration(f, t0, tf, dt, x0, pars=[params_fitted, params_known])
        
    return t_forecast, xf

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
        p_ambient : float
            Best guess at ambient pressure
        dt : float
            Largest suitable time step based on convergence analysis
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
    # I don't think this is the correct value, but we will fit this later
    p_ambient = po[0]
    params_known = [q]
    
    # Get euler solution for first attempt - initial values from notebook
    a = 2.2e-3
    b = 1.1e-1
    c = 6.8e-3

    # These parameters will be fitted later
    params_unknown = [a,b,c,p_ambient]

    pm = improved_euler_pressure(dPdt, t0, t1, dt, p0, pars=[params_unknown,params_known])

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
    
    # Pass uncertainty for out covariance matrix
    uncertainty = 0.6
    # Fit curve
    params_fitted, pcov = calibration(dPdt, t0, t1, converged_dt, x0=p0, params_unknown=params_unknown, params_known=params_known, to=tp, yo=po, sigma=uncertainty)

    # Re solve with correct parameters
    pm = improved_euler_pressure(dPdt, t0, t1, converged_dt, p0, pars=[params_fitted,params_known])

    # Plot calibrated model with residuals
    plot_residuals(dPdt, pm, po, t, tp)

    # Get samples
    samples = np.random.multivariate_normal(params_fitted, pcov, 100)

    # Plot ensemble
    ax, p_models = model_ensemble(samples, t, params_known, po, tp)

    # Start forecasting
    forecast = [(4.0,'g-'),(2.0,'r-'),(1.0,'b-'),(0.5,'m-')]
    data = [[],[],[],[]]
    i=0
    # Lets say 30 years
    tf = t[-1]+30
    p_for_loss = []
    for (sample, p_sample) in zip(samples, p_models):
        p_temp = p_sample.tolist()
        p_for_loss.append([])
        for n, form in forecast:
            # Calculate q
            q_inj = n*q_co2[-1]
            q_ext = q_water[-1]
            q_forecast = q_ext - q_inj
            # Forecast
            t_forecast, pf = forecast_model(dPdt, t, tf, p_sample[-1], sample, [q_forecast,p_ambient])
            # Extend p to include forecast
            p_for_loss[int(i/4)].append(p_temp + pf[1:].tolist())
            # Plot
            ax.plot(t_forecast, pf, form, lw=0.2, alpha=0.3)
            # Save pf for histogram
            data[i%4].append(pf[-1])
            if i < 4:
                ax.plot([],[], form, label='Injection = {:.1f} kg/s'.format(q_inj))
            i += 1

    ax.set_ylabel('Pressure (MPa)')
    ax.set_xlabel('Time (years)')
    ax.set_title('Forecast of pressure under different injection rates')
    ax.axvline(t[-1], color='k', ls='--', lw=0.5, alpha=0.4)
    ax.legend(loc=3)

    if False:
        plt.show()
    else:
        plt.savefig('dpdt_forecast_uncertainty')
        plt.close()

    # Also generate a model that does not account for uncertainty - illustrative purposes
    f, ax = plt.subplots(1,1)
    for n, form in forecast:
            # Calculate q
            q_inj = n*q_co2[-1]
            q_ext = q_water[-1]
            q_forecast = q_ext - q_inj
            # Forecast
            t_forecast, pf = forecast_model(dPdt, t, tf, pm[-1], params_fitted, [q_forecast])
            # Extend p to include forecast
            # Plot
            ax.plot(t_forecast, pf, form, label='Injection = {:.1f}kg/s'.format(q_inj))
    ax.plot(t,pm,'b-')
    ax.set_ylabel('Pressure (MPa)')
    ax.set_xlabel('Time (years)')
    ax.set_title('Forecast of pressure under different injection rates')
    ax.plot(tp,po,'k.', label='Pressure - Observed')
    ax.axvline(t[-1], color='k', ls='--', lw=0.5, alpha=0.4)
    ax.legend(loc=3)
    if False:
        plt.show()
    else:
        plt.savefig('dpdt_forecast.png')
        plt.close()



    # Clear the file 'confint' if it exists, in preparation for 95% confints
    fp = open('confint.txt', 'w+')
    fp.write('\n')

    # Get histograms for data
    for i in range(len(data)):
        n,_ = forecast[i]
        title = '{:d} Pressure at {:.1f} times current injection'.format(int(tf), n)
        get_hist(data[i], title)


    a,b,c,p_ambient = params_fitted

    p_models = [x for x in zip(*p_for_loss)]

    # Output fitted curve and best parameters
    return pm, t, a, b, c, p_ambient, converged_dt, p_models

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
            Array containing observed values of carbon concentration
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
            Best guess at parameter d
        M0 : float
            Best guess at reservoir mass
    """

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

    # Input uncertainty for covariance matrix
    uncertainty = 0.015
    
    # Calibrate
    params_fitted, pcov = calibration(dCdt, t0, t1, dt, C0, params_unknown, params_known, t_cc, co, sigma=uncertainty)

    # Re-solve with fitted parameters
    cm = improved_euler_concentration(dCdt, t0, t1, dt, C0, pars=[params_fitted,params_known])

    # plot
    plot_residuals(dCdt, cm, co, t, t_cc)

    # Get samples
    samples = np.random.multivariate_normal(params_fitted, pcov, 100)

    # Plot ensemble
    ax, c_models = model_ensemble_conc(samples, t, params_known, co, t_cc)

    # Start forecasting
    forecast = [(4.0,'g-'),(2.0,'r-'),(1.0,'b-'),(0.5,'m-')]
    # Lets say 30 years
    tf = t[-1]+30

    # Allocate data lists for histogram
    data = [[],[],[],[]]
    i=0

    # Pre-allocate for solving qloss later
    c_for_loss = []
    
    for (sample, c_sample) in zip(samples, c_models):
        c_temp = c_sample.tolist()
        c_for_loss.append([])
        for n, form in forecast:
            # Calculate q
            q_forecast = n*q_co2_interp[-1]
            # Forcast
            t_forecast, cf = forecast_model(dCdt, t, tf, c_sample[-1], sample, [q_forecast,p_ambient,C_ambient,a,b,c,cm])
            c_for_loss[int(i/4)].append(c_temp + cf[1:].tolist())
            # Plot
            ax.plot(t_forecast, cf, form, lw=0.2, alpha=0.3)
            # Save cf for histogram
            data[i%4].append(cf[-1])
            if i < 4:
                ax.plot([],[], form, label='Injection: {:.1f} kg/s'.format(q_forecast))
            i += 1

    ax.set_ylabel('Concentration (weight fraction)')
    ax.set_xlabel('Time (years)')
    ax.set_title('Forecast of concentration under different injection rates')
    ax.axhline(y=0.1, color='r', ls='-.', lw=1, alpha=0.4, label='10% limit')
    ax.legend(loc=2)
    
    if False:
        plt.show()
    else:
        plt.savefig('dcdt_forecast_uncertainty.png')
        plt.close()

    # Also generate a model that does not account for uncertainty - illustrative purposes
    f, ax = plt.subplots(1,1)
    for n, form in forecast:
            # Calculate q
            q_forecast = n*q_co2_interp[-1]
            # Forcast
            t_forecast, cf = forecast_model(dCdt, t, tf, cm[-1], params_fitted, [q_forecast,p_ambient,C_ambient,a,b,c,cm])
            # Plot
            ax.plot(t_forecast, cf, form, label='Injection = {:.1f}kg/s'.format(q_forecast))
    ax.plot(t,cm,'b-')
    ax.set_ylabel('C02 Concentration (wt fraction)')
    ax.set_xlabel('Time (years)')
    ax.set_title('Forecast of concentration under different injection rates')
    ax.plot(t_cc,co,'k.', label='Concentration - Observed')
    ax.axvline(t[-1], color='k', ls='--', lw=0.5, alpha=0.4)
    ax.legend(loc=2)
    ax.axhline(0.1, color='r', ls='-.', lw=0.5, alpha=0.4, label='10% limit')
    if False:
        plt.show()
    else:
        plt.savefig('dcdt_forecast.png')
        plt.close()

    # Get histograms for data
    for i in range(len(data)):
        n,_ = forecast[i]
        title = '{:d} Concentration at {:.1f} times current injection'.format(int(tf), n)
        get_hist(data[i], title)

    c_models = [x for x in zip(*c_for_loss)]

    return cm,t,d,M0, c_models

def main():

    # Read Data
    tp, po, t_water, q_water, t_co2, q_co2, t_cc, co = read_data()
    # Arbitrarily define timestep to be used before convergence testing
    t0 = tp[0]
    t1 = tp[-1]
    dt = 0.5
    # Solve pressure
    pm, tp, a, b, c, p_ambient, dt, p_models = solve_pressure(t_water, q_water, t_co2, q_co2, t0, t1, dt, po, tp)
    # Solve concentration
    cm,t,d,m0, c_models = solve_concentration(pm, tp, t_cc, co, t_co2, q_co2, t0, t1, dt, [a,b,c,p_ambient])

    # Benchmark
    x_analytical_pressure,x_eulers_pressure=benchmark_pressure(1967.34,2018.50,0.5,6.17e+00,[[0.0019,0.1534,0.0265,6.17e+00],[-63.]])
    benchmark_concentration(1967.34,2018.50,0.5,2.98e-02,[[0.25246,9167.33],[63,6.17e+00,0.03,0.0019,0.1534,0.0265,x_eulers_pressure]])

    # Determine CO2 loss to waterways
    # For each scenario
    f, ax = plt.subplots(1,2, figsize=(12,6))
    forecast = [(4.0,'g-'),(2.0,'r-'),(1.0,'b-'),(0.5,'m-')]
    i=0
    for n, form in forecast:
        # For each sample
        for (p, c) in zip(p_models[i], c_models[i]):
            qloss, t = solve_q_loss(p,c,t0,tp[-1],dt,[a,b,p_ambient])
            ax[0].plot(t,qloss,form, lw=0.3, alpha=0.3)
            ax[1].plot(t,qloss,form, lw=0.3, alpha=0.3)
        i += 1
        ax[0].plot([],[], form, label='{:.1f} times current injection'.format(n))
    ax[0].set_xlabel('Time (Years)')
    ax[1].set_xlabel('Time (Years)')
    ax[0].set_ylabel('CO2 loss (kg)')
    ax[0].set_title('CO2 Loss to surrounding waterways at \nvarious injection rates')
    ax[1].set_title('Zoomed')
    ax[1].set_ylim(bottom=0, top=0.25)
    ax[0].legend(loc=2)
    if False:
            plt.show()
    else:
        plt.savefig('CO2 Loss.png')
        plt.close()

        

    print('Done')
    

if __name__ == "__main__":
    main()