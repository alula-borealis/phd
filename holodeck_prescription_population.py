# This is for a population using a semi-analytic prescription
import numpy as np
import matplotlib.pyplot as plt
import astropy as ap
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u

# constants
m_sun = ap.constants.M_sun
G = ap.constants.G
c = ap.constants.c
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

vsim = cosmo.comoving_volume(z).si   # Comoving volume
dc = cosmo.comoving_distance(z).si   # comoving distance (these should be SI)
print("vsim", vsim)
print("dc", dc)

fbins = 100
f = np.linspace(0.0001, 1, fbins)*u.Hz.si # for population
# f = 1*u.Hz.si # for single binary
print("f", f)

# calculations
# masses
m_total = m1 + m2
mc = ((m1*m2)**(3/5))/(m_total**(1/5)) # chirp mass
# hardening timescale
# time it takes for binary to come close together so the freq increases
# by an e-folding
# dt/dln(fr) = fr/(dfr/dt) = tau

# f is the observer-frame frequency to source
fr = (1 + z)*f # fr is the rest-frame GW freq (twice orbital)

# assuming circular binary driven by GWs
tau = (5/96)*(((G*massesc)/c**3)**(-5/3))*(2*pi*fr)**(-8/3)

# total GW power emitted
power = (32/5*G*c**5)*(G*massesc)**(10/3)*(2*pi*fr)**(10/3)

# luminosity distance related to comoving distance
dl = dc*(1 + z)

# GW strain [Sesana 08]
hs = (8/10**(1/2))*((G*massesc)**(5/3)/((c**4)*dl))*(2*pi*fr)**(2/3)
print("hs", hs)

# relating number and number density
# d^2N/dz dln(fr) = (dnc/dz)(dt/dln(fr))(dz/dt)(dVc/dz)

# dt/dln(fr) is the hardening timescale
# -the longer binaries spend in an interval, the more binaries expected to find
# nothing enforces number of binaries is an integer

# dz/dt = H0(1+z)E(z) rate of evolution of the Universe

# dVc/dz = 4pi(c/Ho)(dc^2/E(z))

# hc^2 = \int(from 0 to inf) dz(dnc/dz)hs^2 4pi c dc^2 (1+z)tau
#### FOR AN INDIVIDUAL BINARY

h2_integrand = (hs**2)*((4*pi*c*(dc**2)*(1+z))/vsim)*tau
h = np.sqrt(h2_integrand)

print("strain", h)

#### SUMMATION
## comoving number-density of sources
# nc = dN/dVc = N/V_sim
num = 1e6 # number of sources N
# vsim - comoving volume of simulation - the same as vc?
nc = num/vsim # comoving number-density of sources
