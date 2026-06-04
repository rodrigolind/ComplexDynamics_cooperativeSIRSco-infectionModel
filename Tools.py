import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy as scp
#import icomo
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import fsolve

plt.rcParams['font.size'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.titlesize'] = 20

#########################################

def MultiplyPotentialList(PotentialList, Faktor):
    Res = np.array(PotentialList) * Faktor
    if Res.size == 1:
        return Res.item()
    return Res.tolist()

def ChangeXTicksFromAlphaToR0(ax=None, gamma=0.1, resetToxlims=False, ticks=None):
    """
    Change x-ticks from alpha to R0 by dividing by gamma.
    Works with both plt and axes objects.
    """
    if ax is None: ax = plt.gca() # Use the current active axes if no axes object is provided
    
    if(ticks==None): xticks = ax.get_xticks()  # Get current x-ticks
    else: xticks = ticks

    ax.set_xlabel(r'$R_0$')
    ax.set_xticks(xticks, labels=[f'{tick / gamma:.1f}' for tick in xticks])

    if(resetToxlims==True): # Needed for JacEV-plot but does weird things in analytical SIRS FPS for X,P,R tripple plot
        x_vals = ax.get_lines()[0].get_xdata()
        ax.set_xlim(min(x_vals), max(x_vals))  # Set x-limits to the range of x-values

    return 0

def ChangeColorbarTicksFromAlphaToR0(cbar, gamma=0.1, Label=r'$R_0$'):
    """
    Change colorbar tick labels from alpha to R0 = alpha / gamma.
    Works on the Colorbar object.
    """
    ticks = cbar.get_ticks()
    new_labels = [f'{tick / gamma:.1f}' for tick in ticks]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(new_labels)
    cbar.set_label(Label)

    return 0

def SetInCondSIR(epsi, p, InInfComp='AB'):
    if InInfComp == 'AB': p['y0'] = [1-epsi, epsi, 0]
    elif InInfComp == 'A and B': p['y0'] = [1-epsi, epsi/2, epsi/2]
    else: 
        raise ValueError("InInfComp must be 'AB' or 'A and B'")
    return 0

###########################################################
######################## for SIRS #########################
###########################################################

def SetInCondredSIRS(epsi, p, InInfComp='A and B'):
    if InInfComp == 'AB': p['y0'] = [1-epsi, epsi, 0, epsi, 0, epsi]
    elif InInfComp == 'A and B': p['y0'] = [1-epsi, epsi/2, epsi/2, 0, epsi/2, 0]
    else: 
        raise ValueError("InInfComp must be 'AB' or 'A and B'")
    return 1

def SetInCondfullSIRS(epsi, p, InInfComp='A and B'):
    if InInfComp == 'AB': p['y0'] = [1-epsi,0,0,epsi,0,0,0,0,0,0,0]
    elif InInfComp == 'A and B': p['y0'] = [1-epsi,epsi/2,epsi/2,0,0,0,0,0,0,0,0]
    else: 
        raise ValueError("InInfComp must be 'AB' or 'A and B'")
    return 1

def PlotDFE(ax=None, x_space='alpha', gamma=0.1, kappa=1, lw=3):
    if ax is None: ax = plt.gca() # Use the current active axes if no axes object is provided

    if(x_space=='alpha'):
        StabChange = gamma/kappa
    if(x_space=='R0'):
        StabChange = 1
    ax.plot([0,StabChange], [0,0], linewidth = lw, ls='-', c='b', label='DFE')
    ax.plot([StabChange,StabChange*2], [0,0], linewidth = lw, ls='--', c='b')
    return 0

########### without S,M ###########

def GetBifIdxOfEE(UpperBranch, LowerBranch):
        Bif_Idx = np.max(np.where(np.abs(UpperBranch-LowerBranch) < 0.001))
        if(Bif_Idx == len(UpperBranch)-1): # Both curves are the same because probably C<=2 => Set BifIdx when curves become positive (should be at R_0=1)
            Bif_Idx = np.min(np.where(UpperBranch > -0.00001))
        return Bif_Idx

########## with S+M #########

def is_Fixpoint(trajectory, checkOnlyLastValues=False, threshold=0.1):
    if checkOnlyLastValues:
        trajectory = trajectory[-100:]
        if(np.std(trajectory)==0): return True
        else: return np.std(trajectory)
    # Normalize the trajectory
    normalized = (trajectory - np.mean(trajectory)) / np.std(trajectory)
    # Compute autocorrelation
    autocorr = np.correlate(normalized, normalized, mode='full')
    autocorr = autocorr[autocorr.size // 2:]  # Keep only the second half
    # Find peaks in the autocorrelation
    peaks, _ = scp.signal.find_peaks(autocorr, height=threshold)
    return len(peaks) < 2  # More than one peak indicates periodicity

def CheckIfPeriodic(X, ACI_thresh=.8, plot=False):
    # Normalize time series
    X_forac = (X - np.mean(X)) / np.std(X)
    # Calculate the autocorrelation function
    X_ac = np.correlate(X_forac, X_forac, mode='full')
    X_ac = X_ac[X_ac.size // 2:] # only second half
    # Normalize the autocorrelation function
    X_ac = X_ac / np.max(X_ac)
    Max_idces, _ = scp.signal.find_peaks(X_ac)
    ACI = np.max([X_ac[Idx] for Idx in Max_idces]) # Autocorr Index = Largest secondary maximum (cite Ulrich Parlitz)
    
    if(plot==True):
        plt.plot(X_ac)
        plt.axhline(y=ACI, color='g', linestyle='--', label=f'ACI={ACI:.2f}')
        for i in range(len(Max_idces)):
            plt.axvline(x=Max_idces[i], color='r', linestyle='--')
        plt.legend()
        plt.show()
    
    if(ACI>ACI_thresh): # periodic
        return True
    return False

def amplitude_decay(X, in_percent=True, min_peaks=10):
    peaks, _ = scp.signal.find_peaks(X)
    if len(peaks) == 0: # probably a fixpoint
        return None
    elif len(peaks) < min_peaks:
        print('Need 10 peaks for amplitude decay fit, found only', len(peaks))
        return None
    peak_vals = X[peaks]
    # Fit linear trend in log space
    t = np.arange(len(peak_vals))
    coeff = np.polyfit(t, np.log(peak_vals), 1)
    decay_rate = coeff[0]
    if in_percent==True:
        decay_rate /= (np.max(X)-np.min(X))/2
    return decay_rate

def SetTolerances_for_kappa_s(s,kappa,p):
    # Compute line in log-log space: use lower tolerances if below line
    x1, x2, y1, y2 = 0.5, 1, 0.039810717055349734, 0.4 # Values for mmax = 0.2 (approximated moreless)
    a = (np.log(y2) - np.log(y1)) / (np.log(x2) - np.log(x1))  # slope in log-log
    b = np.log(y1) - a * np.log(x1)                            # intercept in log-log
    if(np.log(kappa) > a * np.log(s) + b): p['tolerances'] = [1e-10, 1e-13]
    else: 
        p['tolerances'] = [1e-15, 1e-18]
        print('tolerances lowered for kappa =', kappa, 'and s =', s)
    return 0

def SetTolerances_for_kappa_mmax(mmax,kappa,p):
    # Compute line in log-log space: use lower tolerances if below line
    x1, x2, y1, y2 = 0.78, 1, 0.2290867652767773, 0.32 # Values for s = 0.25 (approximated moreless)
    a = (np.log(y2) - np.log(y1)) / (np.log(x2) - np.log(x1))  # slope in log-log
    b = np.log(y1) - a * np.log(x1)                            # intercept in log-log
    if(np.log(kappa) > a * np.log(mmax) + b): p['tolerances'] = [1e-10, 1e-13]
    else: 
        p['tolerances'] = [1e-15, 1e-18]
        print('tolerances lowered for kappa =', kappa, 'and mmax =', mmax)
    return 0

def connect_points(ax, xvals, yvals, p1, p2, shift=('none', 'none'),
                   logspace=True, color='k', lw=2, label=None):
    """
    Draw a line connecting two grid points, with optional half-cell shifts.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw.
    xvals, yvals : array-like
        Parameter arrays for x and y directions.
    p1, p2 : tuple(int, int)
        Grid indices (i, j) for the two points to connect.
    shift : tuple(str, str)
        Direction to shift each point: each entry can be one of
        {'none', 'up', 'down', 'left', 'right'}.
        The shift is applied by half a cell width in that direction.
    logspace : bool, default=True
        Whether axes are logarithmic (use geometric midpoints).
    color : str, default='red'
        Line color.
    lw : float, default=2
        Line width.
    """

    def midpoint(a, b):
        """Half-step between consecutive grid points (geometric or arithmetic)."""
        return np.sqrt(a * b) if logspace else 0.5 * (a + b)

    def apply_shift(i, j, direction):
        """Return (x, y) coordinate shifted half a cell."""
        # base position
        x, y = xvals[i], yvals[j]
        if direction == 'up' and j < len(yvals) - 1:
            y = midpoint(yvals[j], yvals[j+1])
        elif direction == 'up' and j == len(yvals) - 1:
            y = 100
        elif direction == 'down' and j > 0:
            y = midpoint(yvals[j-1], yvals[j])
        elif direction == 'down' and j == 0:
            y = -100
        elif direction == 'right' and i < len(xvals) - 1:
            x = midpoint(xvals[i], xvals[i+1])
        elif direction == 'right' and i == len(xvals) - 1:
            x = 100
        elif direction == 'left' and i > 0:
            x = midpoint(xvals[i-1], xvals[i])
        elif direction == 'left' and i == 0:
            x = -100
        return x, y

    # Compute shifted coordinates for both points
    x1, y1 = apply_shift(*p1, shift[0])
    x2, y2 = apply_shift(*p2, shift[1])

    # Draw the connecting line
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=4, label=label)
    return ax