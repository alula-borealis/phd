import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as syc
import astropy.constants as ayc
from scipy import interpolate

# Constants
pi = np.pi
G = 6.67430e-11  # gravitational constant in m^3 kg^(-1) s^(-2)
c = 299792458  # speed of light in m/s
H0 = 70e3 / (3.086e22)  # Hubble constant in s^(-1), converted from km/s/Mpc to 1/s
# constants
m_sun = 1.989e30
r_sun = ayc.R_sun
kpc = ayc.kpc
pc = ayc.pc
fm     = 3.168753575e-8   # LISA modulation frequency
YEAR   = 3.15581497632e7  # year in seconds
AU     = 1.49597870660e11 # Astronomical unit (meters)
MPC = pc*1e6


# Function to calculate the integrand
def integrand(f, M, dM, z):
    return (M**(5/3) / (dM**2 * (1 + z)**(1/3))) * f**(2/3)

# Monte Carlo simulation
def monte_carlo_sum(num_samples, frequencies):
    results = []

    for f in frequencies:
        sum_result = 0.0

        for _ in range(num_samples):
            # Sample parameters (you need to define appropriate ranges)
            M_i = np.random.uniform(low=lower_bound_M, high=upper_bound_M)
            dM_i = np.random.uniform(low=lower_bound_dM, high=upper_bound_dM)
            z_i = np.random.uniform(low=lower_bound_z, high=upper_bound_z)

            # Calculate contribution to the sum
            sum_result += integrand(f, M_i, dM_i, z_i)

        # Constants in the expression
        prefactor = (2 * pi**(2/3) / 9) * (G**(5/3) / (c**3 * H0**2))

        # Average over samples and multiply by prefactor
        result = prefactor* sum_result / num_samples

        results.append(result)

    return results

# Define the range of parameters and frequency
lower_bound_M = 10*m_sun
upper_bound_M = 100*m_sun
lower_bound_dM = 1e5
upper_bound_dM = 1e7
lower_bound_z = 0
upper_bound_z = 5

h = 1.10423e-44

# Define frequency range
frequencies = np.logspace(0.0001, 1, 100)  # Adjust the range and number of points as needed

# Number of Monte Carlo samples
num_samples = 1000  # Adjust as needed

# Perform the Monte Carlo simulation
results = monte_carlo_sum(num_samples, frequencies)

# Plot the results
plt.plot(frequencies, results)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Output')
plt.title('Monte Carlo Simulation Results')
plt.show()


###########################
