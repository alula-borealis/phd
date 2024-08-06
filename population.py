import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import G, c
from scipy.integrate import simps
from scipy.interpolate import interp1d

# Constants
Msol = 1.9891e30  # Solar mass in kg
c_speed = 299792458  # Speed of light in m/s - can probably get rid of this

# Function to calculate gravitational wave frequency and strain
def calculate_gw_frequency(m1, m2, distance):
    total_mass = m1 + m2
    chirp_mass = (m1 * m2) ** (3 / 5) / (total_mass ** (1 / 5))
    frequency = ((c_speed ** 3)/(2*G*np.pi))*(1/chirp_mass**(5/3))*(5/256)*Msol  # Hz
    strain_amplitude = (4 * chirp_mass * G * total_mass * Msol / c_speed ** 2) ** (5 / 3) / (distance * 3.086e22)  # Strain
    return frequency, strain_amplitude

# Function to simulate binary black hole mergers
def simulate_binary_mergers(num_mergers, mass_range, distance_range):
    mergers = []

    for _ in range(num_mergers):
        m1 = np.random.uniform(*mass_range)
        m2 = np.random.uniform(*mass_range)
        distance = np.random.uniform(*distance_range)

        frequency, strain = calculate_gw_frequency(m1, m2, distance)
        mergers.append({'m1': m1, 'm2': m2, 'distance': distance, 'frequency': frequency, 'strain': strain})

    return mergers

# Function to calculate gravitational wave background
def calculate_gw_background(mergers, frequency_bins):
    background_strain = np.zeros(len(frequency_bins))

    for merger in mergers:
        frequency, strain = merger['frequency'], merger['strain']
        frequency_indices = np.searchsorted(frequency_bins, frequency, side='right')
        background_strain[:frequency_indices] += strain

    return background_strain
    
# Simulation parameters
num_mergers = 1000
mass_range = (10, 50)  # Black hole mass range in solar masses
distance_range = (1e6, 1e9)  # Distance range in parsecs
frequency_bins = np.logspace(-4, 0, 100)  # Frequency bins in Hz

# Simulate binary mergers
mergers = simulate_binary_mergers(num_mergers, mass_range, distance_range)

# Calculate gravitational wave background
background_strain = calculate_gw_background(mergers, frequency_bins)

print("Freq bins: ",frequency_bins)
print("strain: ", background_strain)
# Plot the gravitational wave background
plt.loglog(frequency_bins, background_strain)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Strain Amplitude')
plt.title('Gravitational Wave Background from Binary Black Hole Mergers')
plt.show()
