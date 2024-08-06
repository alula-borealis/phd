import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as syc
import astropy.constants as ayc

# constants
G = syc.g
c = syc.c
pi = syc.pi
m_sun = ayc.M_sun
r_sun = ayc.R_sun
kpc = ayc.kpc
pc = ayc.pc

m1 = 1e5*m_sun    # mass in solar mass
m2 = 1e5*m_sun
f1 = 1e-2   # testing frequencies for LISA
f2 = 1e-4

# this part is just testing that the times make sense

#h = (3.6e-22)*((1e9*pc)/r)*(m1/(10*m_sun))*((m2/(1e6*m_sun))**(2/3))*((f/0.01)**(2/3))

T1 = (1.41e6)*(((0.01)/f1)**(8/3))*((10*m_sun)/m1)*(((1e6*m_sun)/m2)**(2/3))

T2 = (1.41e6)*(((0.01)/f2)**(8/3))*((10*m_sun)/m1)*(((1e6*m_sun)/m2)**(2/3))

final = T2-T1

print("Time = ", final)
print("T days = ", final/86400)


# plot of strain vs freq for LISA freqs at a distance of 3 Gpc

def calculate_strain(mass, freqs, r):
    strain = ((3.6e-22))*((1/r))*(mass/10)*((mass/1e6)**(2/3))*((freqs/0.01)**(2/3))

    return strain


r = 3   # distance in Gpc
masses = np.linspace(1e5*m_sun, 1e8*m_sun, 5)
freqs = np.linspace(0.000001, 10, 100)
strains = []

for mass in masses:
    strain_data = calculate_strain(mass, freqs, r)
    strains.append(strain_data)


for i, mass in enumerate(masses):
    plt.loglog(freqs, strains[i], label=f"Mass of BH = {mass} M_solar")

### this probably needs to be a loglog graph!

plt.title("Strain vs Frequency for different BH masses")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Strain")
plt.legend()
plt.show()