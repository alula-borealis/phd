# packages
import astropy as ap
import astropy.constants
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats
import tqdm.notebook as tqdm

import kalepy as kale
import kalepy.utils
import kalepy.plot

# constants from holodeck.constants
YR = ap.units.year.to(ap.units.s)
MSOL = ap.constants.M_sun.cgs.value   

# parameters

NUM = 1e6   #: number of starting, sample binaries
MASS_EXTR = [1e6, 1e10] #: range of total-masses to construct (units of [Msol])


# this selection comes from Sesana 08 discussed in 3.3
# Specify PTA frequency range of interest
TMAX = (20.0 * YR)  #: max observing time in units of [sec]
NFREQS = 100    #: number of frequency bins to consider

# Construct target PTA frequency bins
fobs_gw = np.arange(1, NFREQS+1) / TMAX #: frequency bin-centers in units of [Hz]
df = fobs_gw[0] / 2 #: half of frequency bin-width
# [fobs_gw[-1] + df] gives right edge of last freq bin
fobs_gw_edges = np.concatenate([fobs_gw - df, [fobs_gw[-1] + df]])  #: frequency bin-edges

# construct sample population
# likely want to change this to a class which can take actual models
MASS_DENS_POWER_LAW = -3    #: power-law index of mass-distribution

# Choose random masses following power-law distribution with given index in number-density
rr = np.random.random(size=int(NUM))    # generates NUM random numbers between 0-1 for randomness
# 1 is added to convert to the CDF
plaw = MASS_DENS_POWER_LAW + 1.0    # calculates exponant for power law
masses = np.array(MASS_EXTR) ** plaw    #  raises array of masses to power law index
# produces masses linearly interpolated in the mass range given
# a power-law distribution is characterized by a 
# probability density function of the form P(x) \propto x^{-a} where a is the power law exponant
masses = (masses[0] + (masses[1] - masses[0])*rr) ** (1./plaw)
masses *= MSOL  # convert to solar masses
del rr  # free memory

# WHY??
# Set fixed values of redshift and mass-ratio
redz = 0.05      #: redshift of all binaries
mrat = 0.3      #: mass-ratio of all binaries

# draw 1D data distributions
def figax(figsize=[7, 5], ncols=1, nrows=1, sharex=False, sharey=False, squeeze=True,
          scale=None, xscale='log', xlabel='', xlim=None, yscale='log', ylabel='', ylim=None,
          left=None, bottom=None, right=None, top=None, hspace=None, wspace=None,
          widths=None, heights=None, grid=True, **kwargs):
    """Create matplotlib figure and axes instances.

    Convenience function to create fig/axes using `plt.subplots`, and quickly modify standard
    parameters.

    Parameters
    ----------
    figsize : (2,) list, optional
        Figure size in inches.
    ncols : int, optional
        Number of columns of axes.
    nrows : int, optional
        Number of rows of axes.
    sharex : bool, optional
        Share xaxes configuration between axes.
    sharey : bool, optional
        Share yaxes configuration between axes.
    squeeze : bool, optional
        Remove dimensions of length (1,) in the `axes` object.
    scale : [type], optional
        Axes scaling to be applied to all x/y axes.  One of ['log', 'lin'].
    xscale : str, optional
        Axes scaling for xaxes ['log', 'lin'].
    xlabel : str, optional
        Label for xaxes.
    xlim : [type], optional
        Limits for xaxes.
    yscale : str, optional
        Axes scaling for yaxes ['log', 'lin'].
    ylabel : str, optional
        Label for yaxes.
    ylim : [type], optional
        Limits for yaxes.
    left : [type], optional
        Left edge of axes space, set using `plt.subplots_adjust()`, as a fraction of figure.
    bottom : [type], optional
        Bottom edge of axes space, set using `plt.subplots_adjust()`, as a fraction of figure.
    right : [type], optional
        Right edge of axes space, set using `plt.subplots_adjust()`, as a fraction of figure.
    top : [type], optional
        Top edge of axes space, set using `plt.subplots_adjust()`, as a fraction of figure.
    hspace : [type], optional
        Height space between axes if multiple rows are being used.
    wspace : [type], optional
        Width space between axes if multiple columns are being used.
    widths : [type], optional
    heights : [type], optional
    grid : bool, optional
        Add grid lines to axes.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        New matplotlib figure instance containing axes.
    axes : [ndarray] `matplotlib.axes.Axes`
        New matplotlib axes, either a single instance or an ndarray of axes.

    """

    if scale is not None:
        xscale = scale
        yscale = scale

    scales = [xscale, yscale]
    for ii in range(2):
        if scales[ii].startswith('lin'):
            scales[ii] = 'linear'

    xscale, yscale = scales

    if (widths is not None) or (heights is not None):
        gridspec_kw = dict()
        if widths is not None:
            gridspec_kw['width_ratios'] = widths
        if heights is not None:
            gridspec_kw['height_ratios'] = heights
        kwargs['gridspec_kw'] = gridspec_kw

    fig, axes = plt.subplots(figsize=figsize, squeeze=False, ncols=ncols, nrows=nrows,
                             sharex=sharex, sharey=sharey, **kwargs)

    plt.subplots_adjust(
        left=left, bottom=bottom, right=right, top=top, hspace=hspace, wspace=wspace)

    if ylim is not None:
        shape = (nrows, ncols, 2)
        if np.shape(ylim) == (2,):
            ylim = np.array(ylim)[np.newaxis, np.newaxis, :]
    else:
        shape = (nrows, ncols,)

    ylim = np.broadcast_to(ylim, shape)

    if xlim is not None:
        shape = (nrows, ncols, 2)
        if np.shape(xlim) == (2,):
            xlim = np.array(xlim)[np.newaxis, np.newaxis, :]
    else:
        shape = (nrows, ncols)

    xlim = np.broadcast_to(xlim, shape)
    _, xscale, xlabel = np.broadcast_arrays(axes, xscale, xlabel)
    _, yscale, ylabel = np.broadcast_arrays(axes, yscale, ylabel)

    for idx, ax in np.ndenumerate(axes):
        ax.set(xscale=xscale[idx], xlabel=xlabel[idx], yscale=yscale[idx], ylabel=ylabel[idx])
        if xlim[idx] is not None:
            ax.set_xlim(xlim[idx])
        if ylim[idx] is not None:
            ax.set_ylim(ylim[idx])

        if grid is True:
            ax.set_axisbelow(True)
            # ax.grid(True, which='major', axis='both', c='0.6', zorder=2, alpha=0.4)
            # ax.grid(True, which='minor', axis='both', c='0.8', zorder=2, alpha=0.4)
            # ax.grid(True, which='major', axis='both', c='0.6', zorder=2, alpha=0.4)
            # ax.grid(True, which='minor', axis='both', c='0.8', zorder=2, alpha=0.4)

    if squeeze:
        axes = np.squeeze(axes)
        if np.ndim(axes) == 0:
            axes = axes[()]

    return fig, axes

fig, ax = figax(xlabel='Total Mass $[M_\odot]$', ylabel='Number Density $[1/M_\odot]$')
kale.dist1d((masses/MSOL), carpet=False, density=False)
plt.show()



### need to figure out how to make the graph - check the plot.py for holodeck
# not sure what kalepy does



