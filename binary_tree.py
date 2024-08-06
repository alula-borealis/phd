import numpy as np
import matplotlib.pyplot as plt

def draw_delta_S(delta_1, delta_2, S_0, delta_S):
    """Draws a value for \delta S from the mass-weighted probability function"""
    time_step = delta_1 - delta_2 
    exponent = -(time_step ** 2) / (2 * delta_S)
    prefactor = (time_step / (delta_S ** (3/2))) / np.sqrt(2 * np.pi)
    f_FU = prefactor * np.exp(exponent)
    
    # integrate the pdf to draw a random sample
    cdf = np.cumsum(f_FU)   # gives cumulative distribution function (cdf)
    cdf /= cdf[-1]  # normalize to [0, 1]
    print(cdf) 
    
    return np.interp(np.random.rand(), cdf, delta_S)

def binary_tree(mass, z, S_0, delta_S, n_steps=10):
    """Generates a binary merger tree for a halo of given mass at z"""
    tree = [(mass, z)]
    
    for _ in range(n_steps):
        new_tree = []
        for m, z in tree:
            delta = draw_delta_S(S_0, 0, S_0, delta_S)
            S_p = S_0 + delta
            m_p = m * np.sqrt(S_p / S_0)
            new_tree.append((m_p, z + 1))
            new_tree.append((m - m_p, z + 1))
        
        tree = new_tree
    
    return tree

# Example usage
initial_mass = 1e12  # Initial halo mass in solar masses
initial_redshift = 0  # Initial redshift
initial_S_0 = 0.5  # Initial S(M)
delta_S = np.linspace(0.1, 2.0, 1000)  # ΔS values

merger_tree = binary_tree(initial_mass, initial_redshift, initial_S_0, delta_S)

# Plotting the merger tree
masses, redshifts = zip(*merger_tree)
plt.scatter(redshifts, masses, s=10)
plt.xlabel('Redshift')
plt.ylabel('Halo Mass')
plt.yscale('log')
plt.title('Binary Merger Tree of Dark Matter Halos')
plt.show()