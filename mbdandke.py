import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define constants
k = 1       # Spring constant (arbitrary unit)
k_B = 1     # Boltzmann constant (arbitrary unit)
gamma = 1   # Friction coefficient (arbitrary unit)
dt = 0.01   # Time step (arbitrary unit)
n_steps = 5000000  # Number of simulation steps
temperatures = [0.5, 1, 2, 5]  # Temperatures to compare

# Define the theoretical probability density function (PDF)
def P(x, k, k_B, T):
    normalization_factor = np.sqrt(k / (2 * np.pi * k_B * T))
    return normalization_factor * np.exp(-k * x**2 / (2 * k_B * T))

# Langevin simulation function
def langevin_simulation(T, k, gamma, dt, n_steps):
    x = np.zeros(n_steps)
    for i in range(1, n_steps):
        random_force = np.sqrt(2 * k_B * T * gamma / dt) * np.random.randn()
        x[i] = x[i-1] - (k / gamma) * x[i-1] * dt + random_force * dt
    return x

# Prepare data for plotting
x_values = np.linspace(-3, 3, 100)  # Range for x values

# Plot the comparison of final position distribution vs theoretical PDF
plt.figure(figsize=(10, 6))

for T in temperatures:
    # Theoretical PDF
    pdf_theory = P(x_values, k, k_B, T)
    
    # Langevin simulation
    x_sim = langevin_simulation(T, k, gamma, dt, n_steps)
    
    # Extract final positions from the simulation
    final_positions = x_sim[-10000:]  # Last 10,000 positions as final positions
    
    # Compute histogram for final position distribution
    pdf_sim, bin_edges = np.histogram(final_positions, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Mid-points of bins
    
    # Plot theoretical PDF
    plt.plot(x_values, pdf_theory, label=f'Theory T={T}', linestyle='--', linewidth=2)
    
    # Plot simulated PDF (final position distribution)
    plt.plot(bin_centers, pdf_sim, label=f'Simulation (Final Pos) T={T}', linewidth=2)

# Titles and labels
plt.title('Comparison of Final Position Distribution vs Theoretical PDF')
plt.xlabel('Position (arbitrary units)')
plt.ylabel('Probability Density')
plt.xlim(-3, 3)
plt.ylim(0, None)
plt.grid(True)
plt.legend()

# Show plot
plt.show()
