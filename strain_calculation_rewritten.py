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
# holodeck has a function figax() which defines all plot parameters and scaling
# this is rewritten to clearly see how they produce this graph
fig, ax = plt.subplots(figsize=[7,5])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set(xlabel="Total Mass $[M_\odot]$", ylabel="Number Density $[1/M_\odot]$")
kale.dist1d((masses/MSOL), carpet=False, density=False)
plt.show()



