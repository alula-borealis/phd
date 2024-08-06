# This is for a population using a semi-analytic prescription
import numpy as np
import matplotlib.pyplot as plt
import astropy as ap
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u

# constants
m_sun = 1.989e30
G = 6.6743e-11
c = 299792458
pi = np.pi

# variables
m1 = 1
m2 = 2
masses1 = np.random.uniform(1e3, 1e7, 10)*m_sun # population
masses2 = np.random.uniform(1e3, 1e7, 10)*m_sun
total_masses = masses1 + masses1
print(total_masses)

massesc = ((masses1*masses2)**(3/5))/(total_masses**(1/5)) # chirp mass
print(massesc)
#q = 0.9 # mass ratio
# q = m2/m1
z = 0.05 # redshift to source

vsim = cosmo.comoving_volume(z).si.value   # Comoving volume
dc = cosmo.comoving_distance(z).si.value   # comoving distance (these should be SI)
print("vsim", vsim)
print("dc", dc)

fbins = 1000
freqs = np.linspace(0.0001, 1, fbins) # for population
# f = 1*u.Hz.si # for single binary
print("f", freqs)

# calculations
# masses
m_total = m1 + m2
mc = ((m1*m2)**(3/5))/(m_total**(1/5)) # chirp mass
# hardening timescale
# time it takes for binary to come close together so the freq increases
# by an e-folding
# dt/dln(fr) = fr/(dfr/dt) = tau

#### SUMMATION
## comoving number-density of sources
# nc = dN/dVc = N/V_sim
num = 1e6 # number of sources N
# vsim - comoving volume of simulation - the same as vc?
nc = num/vsim # comoving number-density of sources

fr = (1 + z)*freqs # fr is the rest-frame GW freq (twice orbital)

# luminosity distance related to comoving distance
dl = dc*(1 + z)

def calculate_strain(chirp_masses, fr, dl, vsim, z):
    total_strain = np.zeros_like(fr)

    for chirp_mass in chirp_masses:
        # Calculate tau
        tau = (5/96)*(((G*chirp_mass)/c**3)**(-5/3))*(2*pi*fr)**(-8/3)

        # Calculate hs
        hs = (8/10**(1/2))*((G*chirp_mass)**(5/3)/((c**4)*dl))*(2*pi*fr)**(2/3)

        # Calculate h2_integrand
        h2_integrand = (hs**2)*((4*pi*c*(dc**2)*(1+z))/vsim)*tau

        # Accumulate the strain
        total_strain += np.sqrt(h2_integrand)

    return total_strain

total_strain = calculate_strain(massesc, fr, dl, vsim, z)

plt.loglog(fr, total_strain)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gravitational Wave Strain')
plt.title('Population of MBHB, semi-analytic')
plt.show()