import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from astropy import constants as const
import matplotlib.colors as colors

# latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

C_SPEED = 299792458
G_CONST = 6.674*10**(-11)
M_SUN = 2*10**(30)
PC = 3.086*10**(16)

############ add in plots of lifetime as well as separation
### Do I need s_radius?

total_mass = np.geomspace(1e6*M_SUN, 1e10*M_SUN, num=1000)
solar_mass = np.geomspace(1e6, 1e10, num=1000)
mass_ratio = np.geomspace(0.0001, 1, num=1000)

#s_radius = (2*G_CONST*total_mass)/(C_SPEED**2) 

init_sep = (0.01)*PC
#r_crit = 3*s_radius

mass_grid, ratio_grid = np.meshgrid(total_mass, mass_ratio)

#################################

coeff_1 = (5*C_SPEED**5)/(256*G_CONST**3.0)
#lifetime = coeff*(((ratio_grid + 1)**2)/(ratio_grid*mass_grid**3))*(init_sep**4 - r_crit**4)
#life_years = lifetime/31556952

lifetime = coeff_1*(((ratio_grid + 1)**2)/(ratio_grid*mass_grid**3))*(init_sep**4)
life_years = lifetime/31556952


############################
coeff_2 = (256*G_CONST**3.0)/(5*C_SPEED**5)
hub_time = (1e8)*31556952

#sep_quad = coeff*((mass_grid**3)*(ratio_grid)*hub_time/(ratio_grid + 1)**2)
sep_quad = coeff_2*((mass_grid**3)*(ratio_grid)*hub_time/(ratio_grid + 1)**2)
sep = sep_quad**(1/4)
sep_pc = sep/PC

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

pcm = ax1.pcolormesh(solar_mass, mass_ratio, sep_pc, norm='log', cmap='viridis')

ax1.set_title(r'$\tau = 10^8$ yr')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim([min(solar_mass), max(solar_mass)])
ax1.set_ylim([min(mass_ratio), max(mass_ratio)])
ax1.set_xlabel(r"Total Mass ($M_{\odot}$)", fontsize=14)
ax1.set_ylabel(r"Mass Ratio", fontsize=14)
ax1.tick_params(labelsize=13)

bar1 = fig.colorbar(pcm, ax=ax1)
#pcm = ax.pcolor(solar_mass, mass_ratio, sep_pc,
                   #norm=colors.LogNorm(vmin=sep_pc.min(), vmax=sep_pc.max()),
                   #cmap='PuBu_r', shading='auto')

bar1.ax.tick_params(labelsize=13)
bar1.set_label(label='Separation (pc)', size=14, labelpad=10)
########

pcm2 = ax2.pcolormesh(solar_mass, mass_ratio, life_years, norm='log', cmap='viridis')

ax2.set_title(r'$a_{0} = 0.01$ pc')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim([min(solar_mass), max(solar_mass)])
ax2.set_ylim([min(mass_ratio), max(mass_ratio)])
ax2.set_xlabel(r"Total Mass ($M_{\odot}$)", fontsize=14)
ax2.set_ylabel(r"Mass Ratio", fontsize=14)
ax2.tick_params(labelsize=13)

bar2 = fig.colorbar(pcm2, ax=ax2)
#pcm = ax.pcolor(solar_mass, mass_ratio, sep_pc,
                   #norm=colors.LogNorm(vmin=sep_pc.min(), vmax=sep_pc.max()),
                   #cmap='PuBu_r', shading='auto')
bar2.ax.tick_params(labelsize=13)
bar2.set_label(label=r'Merger Time (yr)', size=14, labelpad=10)

fig.tight_layout()

plt.savefig("lifetime_separation.png", dpi=500)
plt.show()