# packages
import cosmopy
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

# cosmopy - code and parameters from holodeck
class Parameters:
    # These are WMAP9 parameters, see [WMAP9], Table 3, WMAP+BAO+H0
    Omega0 = 0.2880                #: Matter density parameter "Om0"
    OmegaBaryon = 0.0472           #: Baryon density parameter "Ob0"
    HubbleParam = 0.6933           #: Hubble Parameter as H0/[100 km/s/Mpc], i.e. 0.69 instead of 69

cosmo = cosmopy.Cosmology(h=Parameters.HubbleParam, Om0=Parameters.Omega0, Ob0=Parameters.OmegaBaryon)

# constants from holodeck.constants
YR = ap.units.year.to(ap.units.s)
MSOL = ap.constants.M_sun.cgs.value   
PC = ap.constants.pc.cgs.value
MPC = 1.0e6*PC # mega-parsec

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
# holodeck has a function figax() which defines all plot parameters and scaling
# this is rewritten to clearly see how they produce this graph
fig, ax = plt.subplots(2,1, figsize=(7,6))
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set(xlabel="Total Mass $[M_\odot]$", ylabel="Number Density $[1/M_\odot]$")
kale.dist1d((masses/MSOL), carpet=False, density=False, ax=ax[0])

#############################

## constructing a number-density distribution

NBINS = 123 #: number of mass-bins for number-density distribution

mbin_edges = MSOL*np.logspace(*np.log10(MASS_EXTR), NBINS+1)    #: edges of mass-bins, units of [gram]
mbin_cents = 0.5*(mbin_edges[:-1] + mbin_edges[1:]) #: centers of mass-bins, units of [gram]

# Volume of the Universe out to the given redshift
vcom = cosmo.comoving_volume(redz).cgs.value    #: Comoving volume in units of [cm^3]

# Calculate binary number-density, units of [1/ (cm^3 * g)]
ndens, *_ = sp.stats.binned_statistic(masses, None, statistic='count', bins=mbin_edges)   # histogram the binaries
ndens /= np.diff(mbin_edges)    #: divide by the bin-widths to get number-density
ndens /= vcom                   #: divide by volume to get a comoving volume-density

# setting up axes
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set(xlabel='Total Mass [$M_\odot$]', ylabel='Differential number density $[M_\odot^{-1} \, \\mathrm{Mpc}^{-3}]$')

# Function to convert bin-edges and histogram heights to specifications for step lines
# this was taken from holodecks plot.py
# probably want to rewrite this or see if a module can do it for you
def _get_hist_steps(xx, yy, yfilter=None):
    size = len(xx) - 1

    xnew = [[xx[ii], xx[ii+1]] for ii in range(xx.size-1)]
    ynew = [[yy[ii], yy[ii]] for ii in range(xx.size-1)]
    xnew = np.array(xnew).flatten()
    ynew = np.array(ynew).flatten()

    if yfilter not in [None, False]:
        if yfilter is True:
            idx = (ynew > 0.0)
        elif callable(yfilter):
            idx = yfilter(ynew)
        else:
            raise ValueError()

        xnew = xnew[idx]
        ynew = ynew[idx]

    return xnew, ynew

# Function to draw histogram steps
def draw_hist_steps(ax, xx, yy, yfilter=None, **kwargs):
    return ax.plot(*_get_hist_steps(xx, yy, yfilter=yfilter), **kwargs)

draw_hist_steps(ax[1], mbin_edges/MSOL, ndens*MSOL*(MPC**3))

plt.tight_layout()
plt.savefig("distribution.png")
plt.show()