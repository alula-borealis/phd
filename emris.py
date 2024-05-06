import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy.cosmology import WMAP9 as cosmo

# latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

C_SPEED = 299792458
G_CONST = 6.674*10**(-11)
M_SUN = 2*10**(30)
PC = 3.086*10**(16)

z = 3   # redshift
dc = cosmo.comoving_distance(z).si.value # comoving distance

# mass as function of schwarzschild radius
rs = 10000  # asteroid size

m_2 = (rs * (C_SPEED**2)) / (2*G_CONST) # asteroid size
m_1 = (4e6)*(M_SUN) # sag a*
chirp = ((m_1*m_2)**(3/5)) / ((m_1 + m_2)**(1/5)) # chirp mass
ecc = 0.5   # eccentricity (does this need to be function?)


chirp_z = chirp*(1 + z) # redshifted chirp
dl = dc*(1 + z) # luminosity distance

# functions of eccentricity

def fe_func(ecc):
    fe_func = (1 + (73/24)*ecc**2 + (37/96)*ecc**4)/((1 - ecc**2)**(7/2))
    return fe_func

def ge_func(ecc):
    ge_func = (304*ecc + 121*ecc**3) / ((1 - ecc**2)**(5/2))
    return ge_func

def gne_func(ecc_n, n):
    gne_func = ((n**4)/32)*((jv(n-2, n*ecc_n) - 2*ecc_n*jv(n-1, n*ecc_n) + (2/n)*jv(n, n*ecc_n) + 2*ecc_n*jv(n+1, n*ecc_n) - jv(n+2, n*ecc_n))**2 + (1-ecc_n**2)*(jv(n-2, n*ecc_n) - 2*jv(n, n*ecc_n) + jv(n+2, n*ecc_n))**2 + (4/(3*n**2))*(jv(n, n*ecc_n)**2))
    return gne_func

# frequency
forb = np.geomspace(0.0001, 0.1, 1000)

def frequency(forb, n, z):
    freq = (forb*n)/(1 + z)
    return freq

# fraction of GW power into each harmonic

def eccentricity(freq, n, z):
    ecc_n = ecc*(freq*(1 + z)/n) # eccentricity of rest frame freq
    return ecc_n

# phi sum
def phi_func(forb, n, z):
    phi_values = []
    for f in forb:
        phi = 0.0
        for n_val in n:
            freq = frequency(f, n_val, z)
            ecc_n = eccentricity(freq, n_val, z)
            phi += (gne_func(ecc_n, n_val) / (n_val**(2/3)) * fe_func(ecc_n))
        phi_values.append(2**(2/3) * phi)
    return phi_values

n = np.arange(1, 100)

def strain(ecc, mass, dl, forb):
    strain_square = (32/15)*(G_CONST**(10/3)/C_SPEED**8)*((4 - np.sqrt(1-ecc**2))/(np.sqrt(1-ecc**2)))*(mass**(10/3)/dl**2)*((2*np.pi*forb)**(4/3))
    return strain_square

phi = phi_func(forb, n, z)
'''
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.scatter(forb, phi)
plt.show()
'''

strain = np.sqrt(strain(ecc, chirp, dl, forb))
print(strain)

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.scatter(forb, strain)
plt.show()

