import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as syc
import astropy.constants as ayc
from scipy import interpolate
# hasasia
import hasasia.sensitivity as hsen
import hasasia.sim as hsim

# constants
G = syc.G
Clight = syc.c
pi = syc.pi
m_sun = ayc.M_sun
r_sun = ayc.R_sun
kpc = ayc.kpc
pc = ayc.pc
fm     = 3.168753575e-8   # LISA modulation frequency
YEAR   = 3.15581497632e7  # year in seconds
AU     = 1.49597870660e11 # Astronomical unit (meters)
MPC = pc*1e6
cspeed = syc.c

# variables
#f = np.logspace(np.log10(1.0e-5), np.log10(1.0e0), 1000)
Larm = 2.5e9
fstar = Clight/(2*pi*Larm)
NC = 2    # default data channels
Tobs = 4*YEAR   # default observation time

# loading in R numerical data (eq 5)
transfer_data = np.genfromtxt("R.txt") # read in the data
        
f = transfer_data[:,0]*fstar        # convert to frequency
R = transfer_data[:,1]*NC           # response gets improved by more data channels

R_INTERP = interpolate.splrep(f, R, s=0)

# functions

def Pn(f):
    """
    Caclulate the Strain Power Spectral Density
    """
    
    # single-link optical metrology noise (Hz^{-1}), eq 10
    P_oms = (1.5e-11)**2*(1. + (2.0e-3/f)**4) 
    
    # single test mass acceleration noise, eq 11
    P_acc = (3.0e-15)**2*(1. + (0.4e-3/f)**2)*(1. + (f/(8.0e-3))**4) 
    
    # total noise in Michelson-style LISA data channel, eq 12
    Pn = (P_oms + 2.*(1. + np.cos(f/fstar)**2)*P_acc/(2.*pi*f)**4)/Larm**2
    
    return Pn

def SnC(f):
    """
    Get an estimation of the galactic binary confusion noise
    """

    # parameters for Tobs = 4 years, see Table 1 for other values
    alpha  = 0.138
    beta   = -221.
    kappa  = 521.
    gamma  = 1680.
    f_knee = 1.13e-3 
    
    A = 1.8e-44/NC
    
    Sc  = 1 + np.tanh(gamma*(f_knee - f))
    Sc *= np.exp(-f**alpha + beta*f*np.sin(kappa*f))
    Sc *= A*f**(-7/3)
    
    return Sc

def Sn(f):
    """ Calculate the sensitivity curve """
    # numerical R
    R = interpolate.splev(f, R_INTERP, der=0)
    # analytical R
    # R = 3./20./(1. + 6./10.*(f/fstar)**2)*2 # fit for R, needs to otherwise be computed numerically
    Sn = Pn(f)/R

    return Sn

Sn = Sn(f) + SnC(f) #Pn/R + SnC

##################### sources
# quasi-circular, non-spinning comparable mass binaries & dominant quadrupole moment

TSUN = 4.92549232189886339689643862e-6
m1 = 0.5e6*TSUN 
m2 = 0.5e6*TSUN
z = 3.0
MPC = 3.08568025e22/cspeed
#### source frame to detector frame
##### this fixed the issues with the x-axis
m1 *= 1. + z
m2 *= 1. + z
M = m1 + m2
Mc = ((m1*m2)**(3/5))/(M**(1/5))
eta = (m1*m2)/M**2
T_merge = 1.*YEAR
H0      = 69.6      # Hubble parameter today
Omega_m = 0.286     # density parameter of matter

# PhenomA, frequency coefficients
a = np.array([2.9740e-1, 5.9411e-1, 5.0801e-1, 8.4845e-1])
b = np.array([4.4810e-2, 8.9794e-2, 7.7515e-2, 1.2848e-1])
c = np.array([9.5560e-2, 1.9111e-1, 2.2369e-2, 2.7299e-1])

def Lorentzian(f, f_ring, sigma):
    """ """ 
    return sigma/(2*np.pi)/( (f-f_ring)**2 + 0.25*sigma**2 )
    
    
def get_freq(M, eta, name):
    """ """
    if (name == "merg"):
       idx = 0
    elif (name == "ring"):
        idx = 1
    elif (name == "sigma"):
        idx = 2
    elif (name == "cut"):
        idx = 3
        
    result = a[idx]*eta**2 + b[idx]*eta + c[idx]
    
    return result/(np.pi*M)
    
def Aeff(f, M, eta, Dl):
    # generate phenomA frequency parameters
    f_merg = get_freq(M, eta, "merg")
    f_ring = get_freq(M, eta, "ring")
    sigma  = get_freq(M, eta, "sigma")
    f_cut  = get_freq(M, eta, "cut")
    
    # break up frequency array into pieces
    mask1 = (f<f_merg)
    mask2 = (f>=f_merg) & (f<f_ring)
    mask3 = (f>=f_ring) & (f<f_cut)
    
    C = M**(5./6)/Dl/np.pi**(2./3.)/f_merg**(7./6)*np.sqrt(5.*eta/24)
    w = 0.5*np.pi*sigma*(f_ring/f_merg)**(-2./3)
    
    A = np.zeros(len(f))
    
    A[mask1] = C*(f[mask1]/f_merg)**(-7./6)
    A[mask2] = C*(f[mask2]/f_merg)**(-2./3)
    A[mask3] = C*w*Lorentzian(f[mask3], f_ring, sigma)
    
    return A

def Dl(z, Omega_m, H0):
    """ calculate luminosity distance in geometric units """
    # see http://arxiv.org/pdf/1111.6396v1.pdf
    x0 = (1. - Omega_m)/Omega_m
    xZ = x0/(1. + z)**3

    Phi0  = (1. + 1.320*x0 + 0.4415*x0**2  + 0.02656*x0**3)
    Phi0 /= (1. + 1.392*x0 + 0.5121*x0**2  + 0.03944*x0**3)
    PhiZ  = (1. + 1.320*xZ + 0.4415*xZ**2  + 0.02656*xZ**3)
    PhiZ /= (1. + 1.392*xZ + 0.5121*xZ**2  + 0.03944*xZ**3)
    
    return 2.*cspeed/H0*((1.0e-3)*MPC)*(1 + z)/np.sqrt(Omega_m)*(Phi0 - PhiZ/np.sqrt(1. + z))


f_cut = get_freq(M, eta, "cut")
f_start = (5.*Mc/T_merge)**(3./8.)/(8.*np.pi*Mc)


f_signal = np.logspace(np.log10(f_start), np.log10(f_cut), 508, endpoint=False)


Dl = Dl(z, Omega_m, H0)
A = Aeff(f_signal, M, eta, Dl)

heff = (np.sqrt(16/5*f_signal**2)*A)

####### tests

#### this is how long, in seconds, the binary is until it's merger at
#### certain frequency
T_merger = 5.*Mc/(8.*np.pi*f_start*Mc)**(8./3.) ### ref: https://par.nsf.gov/servlets/purl/10310155
#print(T_merger)

##### PTAS

totalT = 15*YEAR
spacet = 2*7*24*60*60

ptarange = np.linspace(1/totalT, 1/spacet, 300)

###### hasasia
phi = np.random.uniform(0, 2*np.pi,size=34) # intialize 34 pulsar positions and lifespans
theta = np.random.uniform(0, np.pi,size=34)

psrs = hsim.sim_pta(timespan=11.4, cad=23, sigma=1e-7,
                   phi=phi, theta=theta, Npsrs=34)  # builds curve

freqs = np.logspace(np.log10(5e-10),np.log10(5e-7),400) # freqs to calc red noise
spectra = []
for p in psrs:
     sp = hsen.Spectrum(p, freqs=freqs)
     sp.NcalInv
     spectra.append(sp)

scGWB = hsen.GWBSensitivityCurve(spectra)
scDeter = hsen.DeterSensitivityCurve(spectra)

#### real PTA data - NANOGrav 15 year data
nanodata = np.loadtxt('/home/laura/phd_things/sensitivity_curves/sensitivity_curves_NG15yr_fullPTA.txt', delimiter=",")

nanofreqs = nanodata[:, 0]
nanostrain = nanodata[:, 1]

#################### plotting
fig, ax = plt.subplots(1, figsize=(5,3))

ax.set_xlabel(r'f [Hz]', fontsize=11, labelpad=10)
ax.set_ylabel(r'Characteristic Strain', fontsize=11, labelpad=10)
ax.tick_params(axis='both', which='major', labelsize=11)
    
#ax.set_xlim(1.0e-5, 1.0e0)
#ax.set_ylim(3.0e-22, 1.0e-15)
    
ax.loglog(f, np.sqrt(Sn*f), label=r'LISA') # plot the characteristic strain of LISA
ax.loglog(f_signal, heff, label=r'MBHB $10^{6}\mathrm{M_{\odot}}$ at $z = 3$') # source
ax.loglog(freqs,scGWB.h_c,label=r'Stochastic PTA')
ax.loglog(freqs,scDeter.h_c,label=r'Deterministic PTA')
ax.loglog(nanofreqs, nanostrain, label=r'NANOGrav 15 yr data')

plt.legend()
plt.show()
    


