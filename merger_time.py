import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
import matplotlib.colors as colors

C_SPEED = 299792458
G_CONST = 6.674*10**(-11)
M_SUN = 2*10**(30)
PC = 3.086*10**(16)

############ add in plots of lifetime as well as separation

total_mass = np.geomspace(1e6*M_SUN, 1e10*M_SUN, num=1000)
solar_mass = np.geomspace(1e6, 1e10, num=1000)
mass_ratio = np.geomspace(0.0001, 1, num=1000)


#s_radius = (2*G_CONST*total_mass)/(C_SPEED**2) 

init_sep = (0.01)*PC
#r_crit = 3*s_radius

mass_grid, ratio_grid = np.meshgrid(total_mass, mass_ratio)

#################################
'''
coeff = (5*C_SPEED**5)/(256*G_CONST**3.0)
#lifetime = coeff*(((ratio_grid + 1)**2)/(ratio_grid*mass_grid**3))*(init_sep**4 - r_crit**4)
#life_years = lifetime/31556952

lifetime = coeff*(((ratio_grid + 1)**2)/(ratio_grid*mass_grid**3))*(init_sep**4)
life_years = lifetime/31556952
print(life_years)
'''
############################
coeff = (256*G_CONST**3.0)/(5*C_SPEED**5)
hub_time = (1e8)*31556952

#sep_quad = coeff*((mass_grid**3)*(ratio_grid)*hub_time/(ratio_grid + 1)**2)
sep_quad = coeff*((mass_grid**3)*(ratio_grid)*hub_time/(ratio_grid + 1)**2)
sep = sep_quad**(1/4)
sep_pc = sep/PC

fig, ax = plt.subplots()

pcm = ax.pcolormesh(solar_mass, mass_ratio, sep_pc, norm='log', cmap='viridis')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([min(solar_mass), max(solar_mass)])
ax.set_ylim([min(mass_ratio), max(mass_ratio)])
ax.set_xlabel("Total Mass ($M_{\odot}$)")
ax.set_ylabel("Mass Ratio")

fig.colorbar(pcm, label='Separation $(pc)$')
#pcm = ax.pcolor(solar_mass, mass_ratio, sep_pc,
                   #norm=colors.LogNorm(vmin=sep_pc.min(), vmax=sep_pc.max()),
                   #cmap='PuBu_r', shading='auto')

plt.show()