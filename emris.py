import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy.cosmology import WMAP9 as cosmo
import matplotlib.colors as colors

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
ecc = 0.9   # eccentricity (does this need to be function?)


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
forb = np.geomspace(0.001, 1, 1000)

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

## is the e actually a function of frequency due to the harmonics?
'''
strain = np.sqrt(strain(ecc, chirp, dl, forb))
print(strain)

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.scatter(forb, strain)
plt.show()
'''
##################################
# using new papers

# max frequency of EMRI (at ISCO)

primary_m = 4.3e6*M_SUN
secondary_lower_m = 1e-15*M_SUN
secondary_higher_m = 1e-12*M_SUN

chirp = (primary_m*secondary_higher_m)**(3/5)/(primary_m + secondary_higher_m)**(1/5)

f_isco_max = 4.4*1000*(M_SUN/primary_m)
distance = np.geomspace(8e3*PC, 1e9*PC, num=1000)
dist_pc = distance/PC

lisa_freq = np.geomspace(0.0001, f_isco_max, num=1000) #LISA band

freq_grid, r_grid = np.meshgrid(lisa_freq, distance)

strain_lower = (3.2e-22)/((r_grid)/(1e9*PC))*((secondary_lower_m/M_SUN)**(1/2))*((primary_m/(100*M_SUN))**(1/3))*((100/freq_grid)**(1/6))
strain_higher = (3.2e-22)/((r_grid)/(1e9*PC))*((secondary_higher_m/M_SUN)**(1/2))*((primary_m/(100*M_SUN))**(1/3))*((100/freq_grid)**(1/6))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

pcm = ax1.pcolormesh(lisa_freq, dist_pc, strain_lower, norm='log', cmap='viridis')

ax1.set_title(r'$\mu = 10^{-15} M_{\odot}$')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim([min(lisa_freq), f_isco_max])
ax1.set_xlabel(r"Frequency (Hz)", fontsize=14)
ax1.set_ylabel(r"$r_0$ (pc)", fontsize=14)
ax1.tick_params(labelsize=13)

bar1 = fig.colorbar(pcm, ax=ax1)
#pcm = ax.pcolor(solar_mass, mass_ratio, sep_pc,
                   #norm=colors.LogNorm(vmin=sep_pc.min(), vmax=sep_pc.max()),
                   #cmap='PuBu_r', shading='auto')

bar1.ax.tick_params(labelsize=13)
bar1.set_label(label=R'$h_{c,2}$', size=14, labelpad=10)

ax1.plot()
########

pcm2 = ax2.pcolormesh(lisa_freq, dist_pc, strain_higher, norm='log', cmap='viridis')

ax2.set_title(r'$\mu = 10^{-12} M_{\odot}$')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim([min(lisa_freq), f_isco_max])
ax2.set_xlabel(r"Frequency (Hz)", fontsize=14)
ax2.set_ylabel(r"$r_0$ (pc)", fontsize=14)
ax2.tick_params(labelsize=13)

bar2 = fig.colorbar(pcm2, ax=ax2)
#pcm = ax.pcolor(solar_mass, mass_ratio, sep_pc,
                   #norm=colors.LogNorm(vmin=sep_pc.min(), vmax=sep_pc.max()),
                   #cmap='PuBu_r', shading='auto')

bar2.ax.tick_params(labelsize=13)
bar2.set_label(label=R'$h_{c,2}$', size=14, labelpad=10)

fig.tight_layout()

plt.savefig("distance_bounds_emri_classic.png", dpi=200)

r_lower = 8e3*PC
r_higher = 1e9*PC

strain_mlow_rlow = (3.2e-22)/((r_lower)/(1e9*PC))*((secondary_lower_m/M_SUN)**(1/2))*((primary_m/(100*M_SUN))**(1/3))*((100/lisa_freq)**(1/6))
strain_mhigh_rlow = (3.2e-22)/((r_lower)/(1e9*PC))*((secondary_higher_m/M_SUN)**(1/2))*((primary_m/(100*M_SUN))**(1/3))*((100/lisa_freq)**(1/6))

strain_mhigh_rhigh = (3.2e-22)/((r_higher)/(1e9*PC))*((secondary_higher_m/M_SUN)**(1/2))*((primary_m/(100*M_SUN))**(1/3))*((100/lisa_freq)**(1/6))
strain_mlow_rhigh = (3.2e-22)/((r_higher)/(1e9*PC))*((secondary_lower_m/M_SUN)**(1/2))*((primary_m/(100*M_SUN))**(1/3))*((100/lisa_freq)**(1/6))

fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(10,4))

ax21.set_title(r'$r_0 = 10^{9}$ (pc)')
ax21.set_xscale('log')
ax21.set_yscale('log')
ax21.set_xlim([min(lisa_freq), f_isco_max])
ax21.set_xlabel(r"Frequency (Hz)", fontsize=14)
ax21.set_ylabel(r"$h_{c,2}$", fontsize=14)
ax21.plot(lisa_freq, strain_mhigh_rhigh, label=r'$\mu = 10^{-12} M_{\odot}$')
ax21.plot(lisa_freq, strain_mlow_rhigh, label=r'$\mu = 10^{-15} M_{\odot}$')

ax22.set_title(r'$r_0 = 8 \times 10^{3}$ (pc)')
ax22.set_xscale('log')
ax22.set_yscale('log')
ax22.set_xlim([min(lisa_freq), f_isco_max])
ax22.set_xlabel(r"Frequency (Hz)", fontsize=14)
ax22.set_ylabel(r"$h_{c,2}$", fontsize=14)
ax22.plot(lisa_freq, strain_mhigh_rlow, label=r'$\mu = 10^{-12} M_{\odot}$')
ax22.plot(lisa_freq, strain_mlow_rlow, label=r'$\mu = 10^{-15} M_{\odot}$')

plt.legend()
plt.savefig("mass_bounds_emris_classic.png", dpi=200)

### merger time

chirp_mass = np.geomspace(1e-7*M_SUN, 1e-5*M_SUN, num=1000)
chirp_solar = np.geomspace(1e-7, 1e-5, num=1000)
lisa_frequency = np.geomspace(0.001, 1, num=1000)

freq_grid_lisa, chirp_grid = np.meshgrid(lisa_frequency, chirp_mass)

coeff = (96/5)*(np.pi**(8/3))*(((G_CONST*chirp_grid)/C_SPEED**3)**(5/3))


t_merge = (3/8)*((freq_grid_lisa**(-8/3)))/coeff
t_years = t_merge/31556952
t_years = t_years.astype(float)

print("years: ", t_years)

fig3, ax3 = plt.subplots(figsize=(10,4))

pcm3 = ax3.pcolormesh(lisa_frequency, chirp_solar, t_years, norm='log', cmap='viridis')


ax3.set_xscale('log')
ax3.set_yscale('log')
#ax3.set_xlim([min(lisa_frequency), max(lisa_frequency)])
#ax3.set_ylim([min(chirp_mass), max(chirp_mass)])
ax3.set_ylabel(r"Chirp Mass ($M_{\odot}$)", fontsize=14)
ax3.set_xlabel(r"Frequency", fontsize=14)
ax3.tick_params(labelsize=13)

bar3 = fig.colorbar(pcm3, ax=ax3)
#pcm = ax.pcolor(solar_mass, mass_ratio, sep_pc,
                   #norm=colors.LogNorm(vmin=sep_pc.min(), vmax=sep_pc.max()),
                   #cmap='PuBu_r', shading='auto')
bar3.ax.tick_params(labelsize=13)
bar3.set_label(label=r'Merger Time (yr)', size=14, labelpad=10)

fig.tight_layout()

plt.savefig("lifetime_separation_emris.png", dpi=500)


### test for comparing to 2205.10359

fig4, ax4 = plt.subplots(figsize=(10,4))

r_mega = 1e6*PC
primary_m_classic = 10*M_SUN
secondary_classic_m = 1e-4*M_SUN

mass_c_classic = ((primary_m_classic*secondary_classic_m)**(3/5))/(primary_m_classic + secondary_classic_m)**(1/5)

strain_classic = (3.2e-22)/((r_mega)/(1e9*PC))*((secondary_classic_m/M_SUN)**(1/2))*((primary_m/(100*M_SUN))**(1/3))*((100/lisa_freq)**(1/6))
strain_h0 = (4/r_mega)*(((G_CONST*mass_c_classic)/C_SPEED**2)**(5/3))*((np.pi*lisa_freq)/C_SPEED)**(2/3)

ax4.set_title(r'$r_0 = 10^{6}$ (pc)')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlim([min(lisa_freq), f_isco_max])
ax4.set_xlabel(r"Frequency (Hz)", fontsize=14)
ax4.set_ylabel(r"$h_{c,2}$", fontsize=14)
ax4.plot(lisa_freq, strain_classic, label=r'$\mu = 10^{-4} M_{\odot}$, $M = 10 M_{\odot}$')


plt.legend()
plt.savefig("mass_bounds_emris_classic.png", dpi=200)

#### fdot dependence

fig5, ax5 = plt.subplots(figsize=(10,4))

prime_mass = 4.2e6*M_SUN
secondary_mass = np.geomspace(1e-15*M_SUN, 1e-12*M_SUN, num=1000)
secondary_mass_solar = np.geomspace(1e-15, 1e-12, num=1000)

fid_freq = 0.01

coefficient = (96/5)*(np.pi**(8/3))*(G_CONST**(5/3))*(1/C_SPEED**5)
f_dot = coefficient*(prime_mass*secondary_mass)/((prime_mass + secondary_mass)**(1/3))*(fid_freq**(11/3))

ax5.set_title(r'$f = 0.01$, $M = 4.2 \times 10^6 M_{\odot}$')
ax5.set_xscale('log')
ax5.set_yscale('log')
#ax5.set_xlim([min(lisa_freq), f_isco_max])
ax5.set_xlabel(r"PBH Mass ($M_{\odot}$)", fontsize=14)
ax5.set_ylabel(r"$\dot{f}$", fontsize=14)
ax5.plot(secondary_mass_solar, f_dot)

plt.savefig("mass_dependence_emri.png", dpi=200)

##########
plt.show()

