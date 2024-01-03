import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as syc
import astropy.constants as ayc

# constants
G = syc.g
Clight = syc.c
pi = syc.pi
m_sun = ayc.M_sun
r_sun = ayc.R_sun
kpc = ayc.kpc
pc = ayc.pc

# constants
fm     = 3.168753575e-8   # LISA modulation frequency
YEAR   = 3.15581497632e7  # year in seconds
AU     = 1.49597870660e11 # Astronomical unit (meters)

f = np.logspace(np.log10(1.0e-5), np.log10(1.0e0), 1000)
Larm = 2.5e9
fstar = Clight/(2*pi*Larm)

def Pn(f):
    """
    Caclulate the Strain Power Spectral Density
    """
    
    # single-link optical metrology noise (Hz^{-1}), Equation (10)
    P_oms = (1.5e-11)**2*(1. + (2.0e-3/f)**4) 
    
    # single test mass acceleration noise, Equation (11)
    P_acc = (3.0e-15)**2*(1. + (0.4e-3/f)**2)*(1. + (f/(8.0e-3))**4) 
    
    # total noise in Michelson-style LISA data channel, Equation (12)
    Pn = (P_oms + 2.*(1. + np.cos(f/fstar)**2)*P_acc/(2.*pi*f)**4)/2.5e9**2
    
    return Pn

def SnC(f):
    """
    Get an estimation of the galactic binary confusion noise
    """
    
    Tobs = 4*YEAR   # default observation time
    NC   = 2    # default data channels

    # parameters for Tobs = 4 years
    alpha  = 0.138
    beta   = -221.
    kappa  = 521.
    gamma  = 1680.
    f_knee = 1.13e-3 
    
    A = 1.8e-44/NC
    
    Sc  = 1. + np.tanh(gamma*(f_knee - f))
    Sc *= np.exp(-f**alpha + beta*f*np.sin(kappa*f))
    Sc *= A*f**(-7./3.)
    
    return Sc

def Sn(f):
    """ Calculate the sensitivity curve """
    R = 3./20./(1. + 6./10.*(f/fstar)**2)*2 # fit for R, needs to otherwise be computed numerically
    Sn = Pn(f)/R

    return Sn

Sn = Sn(f) + SnC(f)

#########################



#################### plotting
fig, ax = plt.subplots(1, figsize=(5,3))

ax.set_xlabel(r'f [Hz]', fontsize=12, labelpad=10)
ax.set_ylabel(r'Characteristic Strain', fontsize=12, labelpad=10)
ax.tick_params(axis='both', which='major', labelsize=12)
    
ax.set_xlim(1.0e-5, 1.0e0)
ax.set_ylim(3.0e-22, 1.0e-15)
    
ax.loglog(f, np.sqrt(Sn*f)) # plot the characteristic strain
    
plt.show()
    


