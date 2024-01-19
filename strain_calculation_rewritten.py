# packages
import cosmopy
import astropy as ap
import astropy.constants
import h5py
import matplotlib as mplP
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
NWTG = ap.constants.G.cgs.value             #: Newton's Gravitational Constant [cm^3/g/s^2]
SPLC = ap.constants.c.cgs.value             #: Speed of light [cm/s]

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
fig, ax = plt.subplots(3,1, figsize=(7,6))
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


######### SEMI-ANALYTICAL CALCULATION
## assuming circular, GW-driven evolution

## get bin-edges in chirp-mass

mchirp_edges = mbin_edges*np.power(mrat, 3.0/5.0) / np.power(1 + mrat, 6.0/5.0)
mchirp_cents = 0.5 * (mchirp_edges[:-1] + mchirp_edges[1:])

## construct the integrand
integrand = ndens * np.power(NWTG*mchirp_cents, 5.0/3.0)*np.power(1+redz, -1.0/3.0)

## sum over bins
gwb_sa = ((4.0*np.pi)/(3*SPLC**2))*np.power(np.pi*fobs_gw, -4.0/3.0)*np.sum(integrand*np.diff(mbin_edges))
gwb_sa = np.sqrt(gwb_sa)

#######

######## MONTE CARLO CALCULATION
# from holodeck
def gwb_number_from_ndens(ndens, medges, mc_cents, dcom, fro):
    """Convert from binary (volume-)density [dn/dM], to binary number [dN/dM].
    
    Effectively, [Sesana+2008] Eq.6.
    
    """
    # `fro` = rest-frame orbital frequency
    integrand = ((20*np.pi*(SPLC**6))/96)*ndens*np.diff(medges)
    integrand *= (dcom**2)*(1.0 + redz)*np.power(NWTG*mc_cents, -5.0/3.0)
    integrand = integrand[:, np.newaxis]*np.power(2.0*np.pi*fro, -8.0/3.0)
    return integrand

# Convert from observer-frame GW frequency to rest-frame orbital frequency (assuming circular binaries)
frst_orb = fobs_gw[np.newaxis, :]*(1.0 + redz)/2.0

# Get comoving distance, units of [cm]
dcom = cosmo.comoving_distance(redz).cgs.value

# Calculate spectral strain of binaries at bin-centers
hs_mc = (8.0 / np.sqrt(10))*np.power(NWTG*mchirp_cents, 5.0/3.0)/(dcom*(SPLC**4))
hs_mc = hs_mc[:, np.newaxis]*np.power(2*np.pi*frst_orb, 2.0/3.0) 

# Get the distribution of number of binaries
integrand = gwb_number_from_ndens(ndens, mbin_edges, mchirp_cents, dcom, frst_orb)

# Sum over bins to get GWB amplitude
gwb_mc = np.sum(integrand*(hs_mc**2), axis=0)
gwb_mc = np.sqrt(gwb_mc)

## both match so far - haven't actually done MC sampling
## they both assume a smooth continuous distribution

### ensuring no. binaries in a bin is an integer

NREALS = 100    #: choose a number of realizations to model

"""
NOTE: `gwb_number_from_ndens` returns ``dN/dln(f)``.  We want to create realizations based on ``N``
    the actualy number of binaries.  So we multiply by ``Delta ln(f)``, to get the number of
    binaries in each frequency bin (``Delta N_i``).  Then we calculate the discretizations.
    Then we divide by ``Delta ln(f)`` again, to get the number of binaries per frequency bin,
    needed for the GW characteristic strain calculation.
"""

integrand = gwb_number_from_ndens(ndens, mbin_edges, mchirp_cents, dcom, frst_orb)

# get the number of binaries in each frequency bin
integrand = integrand*np.diff(np.log(fobs_gw_edges))

num_exp = np.sum(integrand[:, 0])
print(f"Expected number of binaries in zero freq bin: {num_exp:.4e}")

# Calculate "realizations" by Poisson sampling distribution of binary number
realized = np.random.poisson(integrand[..., np.newaxis], size=integrand.shape + (NREALS,))

# convert back to number of binaries per log-frequency interval, for GWB calculation
realized = realized/np.diff(np.log(fobs_gw_edges))[np.newaxis, :, np.newaxis]

num_real = np.sum(realized[:, 0, :], axis=0)
num_real_ave = np.mean(num_real)
num_real_std = np.std(num_real)
print(f"Realized number of binaries in zero freq bin: {num_real_ave:.4e} ± {num_real_std:.2e}")

# Calculate GWB amplitude
gwb_mc_real = np.sum(realized*(hs_mc**2)[..., np.newaxis], axis=0)
gwb_mc_real = np.sqrt(gwb_mc_real)

## plot gwb
ax[2].set_xscale("log")
ax[2].set_yscale("log")
ax[2].set(xlabel='Frequency [$yr^{-1}$]', ylabel='Characteristic Strain')
xx = fobs_gw*YR
ax[2].plot(xx, gwb_sa, label="Semi-analytic")
ax[2].plot(xx, gwb_mc, label="Monte-Carlo")

color = 'r'
gwb_mc_med = np.median(gwb_mc_real, axis=-1)
gwb_mc_span = np.percentile(gwb_mc_real, [25, 75], axis=-1)
ax[2].plot(xx, gwb_mc_med, lw=0.5, color=color)
ax[2].fill_between(xx, *gwb_mc_span, alpha=0.25, color=color, label='MC realized')

ax[2].legend()
plt.show()