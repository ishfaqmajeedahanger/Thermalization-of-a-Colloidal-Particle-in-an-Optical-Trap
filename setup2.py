import numpy as np
import matplotlib.pyplot as plt

# Define constants for the specific case
k = 1       # Spring constant (arbitrary unit)
k_B = 1     # Boltzmann constant (arbitrary unit)
gamma = 1   # Friction coefficient (arbitrary unit)
dt = 0.01   # Time step (arbitrary unit)
n_steps = 500  # Number of simulation steps
T = 5       # Temperature to compare
n_particles = 1000  # Number of particles (trajectories)

# Langevin simulation function for multiple particles
def langevin_simulation(T, k, gamma, dt, n_steps, n_particles):
    x = np.zeros((n_particles, n_steps))
    for i in range(1, n_steps):
        random_force = np.sqrt(2 * k_B * T * gamma / dt) * np.random.randn(n_particles)
        x[:, i] = x[:, i-1] - (k / gamma) * x[:, i-1] * dt + random_force * dt
    return x

# Run simulation and get distributions at different steps for multiple particles
steps_to_plot = [50, 100, 200, 400, 500]
trajectories = langevin_simulation(T, k, gamma, dt, max(steps_to_plot), n_particles)

# Prepare figure
fig, axs = plt.subplots(1, 5, figsize=(18, 4))
x_values = np.linspace(-3, 3, 100)

# Define theoretical Boltzmann distribution
def boltzmann_distribution(x, k, k_B, T):
    normalization_factor = np.sqrt(k / (2 * np.pi * k_B * T))
    return normalization_factor * np.exp(-k * x**2 / (2 * k_B * T))

# Plot distribution at each step
for i, step in enumerate(steps_to_plot):
    # Get simulated distribution at this step (positions of all particles)
    x_sim = trajectories[:, step-1]  # Only use positions at the specific step
    
    # Compute histogram
    pdf_sim, bin_edges = np.histogram(x_sim, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot simulated histogram
    axs[i].plot(bin_centers, pdf_sim, label=f'Step {step}')
    
    # Plot theoretical Boltzmann distribution
    pdf_theory = boltzmann_distribution(x_values, k, k_B, T)
    axs[i].plot(x_values, pdf_theory, linestyle='--', label='Boltzmann')
    
    axs[i].set_title(f'Step {step}')
    axs[i].set_xlim(-3, 3)
    axs[i].set_ylim(0, 2)  # Set consistent y-axis limits for all subplots
    axs[i].legend()

plt.suptitle('Thermalization: Evolution of Position Distribution at Different Steps (Multiple Particles)')
plt.tight_layout()
plt.show()


