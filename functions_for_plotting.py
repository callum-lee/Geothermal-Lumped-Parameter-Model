import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from pressure_functions import *
from concentration_functions import *


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
        plt.close()
 
def get_hist(data, title):
    """Generates a histogram showing a 95% confidence interval for a set of data

    Parameters:
    -----------
        data : array-like
            Array of data to be plotted
        title : string
            Data name
    """
    # Generate figure
    f, ax = plt.subplots(1,1)

    # Plot histogram
    ax.hist(data, bins=20)
    
    # Get 95% interval
    data_sorted = np.sort(data)
    lower = int(0.025*len(data))
    upper = int(0.975*len(data))
    x0 = data_sorted[lower]
    x1 = data_sorted[upper]

    # Write exact confint values to .txt (this is just for ease of use in report writing)
    fp = open('confint.txt', 'a+')
    fp.write('{:s}: [{:.3f}, {:.3f}]\n'.format(title, x0, x1))

    # Plot confint
    ax.axvline(x0, color='r', ls='--', lw=0.5, alpha=0.5)
    ax.axvline(x1, color='r', ls='--', lw=0.5, alpha=0.5)

    ax.set_ylabel('Count')
    ax.set_xlabel('{:s}'.format(title))
    ax.set_title('Histogram showing 95% confidence interval \nfor {:s}'.format(title))

    if False:
        plt.show()
    else:
        plt.savefig('{:s}.png'.format(title))
        plt.close()
