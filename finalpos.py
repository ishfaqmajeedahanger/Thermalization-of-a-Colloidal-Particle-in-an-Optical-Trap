import numpy as np
import matplotlib.pyplot as plt

# Define constants
k = 1       # Spring constant (arbitrary unit)
k_B = 1     # Boltzmann constant (arbitrary unit)
gamma = 1   # Friction coefficient (arbitrary unit)
dt = 0.01   # Time step (arbitrary unit)
n_steps = 500 # Number of simulation steps
T = 5       # Temperature

# Define the Boltzmann distribution (theoretical PDF)
def boltzmann_distribution(x, k, k_B, T):
    normalization_factor = np.sqrt(k / (2 * np.pi * k_B * T))
    return normalization_factor * np.exp(-k * x**2 / (2 * k_B * T))

# Langevin simulation function
def langevin_simulation(T, k, gamma, dt, n_steps):
    x = np.zeros(n_steps)
    for i in range(1, n_steps):
        random_force = np.sqrt(2 * k_B * T * gamma / dt) * np.random.randn()
        x[i] = x[i-1] - (k / gamma) * x[i-1] * dt + random_force * dt
    return x

# Simulate trajectories for T = 5
num_trajectories = 1000  # Number of trajectories to use for distribution comparison
final_positions = []

for _ in range(num_trajectories):
    x_sim = langevin_simulation(T, k, gamma, dt, n_steps)
    final_positions.append(x_sim[-1])  # Collect final position at last step

# Prepare the x values for the theoretical PDF
x_values = np.linspace(-3, 3, 100)  # Range for x values
pdf_theory = boltzmann_distribution(x_values, k, k_B, T)

# Plot the histogram of final positions (simulated PDF)
plt.figure(figsize=(10, 6))
plt.hist(final_positions, bins=50, density=True, alpha=0.6, color='b', label='Simulated PDF')

# Plot the theoretical Boltzmann distribution
plt.plot(x_values, pdf_theory, 'r--', lw=2, label='Boltzmann Distribution')

# Titles and labels
plt.title(f'Position Distribution at Final Time vs Boltzmann Distribution (T={T})')
plt.xlabel('Position (arbitrary units)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
