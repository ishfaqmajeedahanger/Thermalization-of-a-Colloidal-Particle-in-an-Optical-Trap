import numpy as np
import matplotlib.pyplot as plt

# Define constants for the specific case
k_x = 1.0e-6    # Spring constant along x (fN/nm)
k_y = 1.0e-6    # Spring constant along y (fN/nm)
k_z = 0.2e-6    # Spring constant along z (fN/nm)
k_B = 1         # Boltzmann constant (arbitrary unit)
gamma = 1       # Friction coefficient (arbitrary unit)
dt = 0.01       # Time step (arbitrary unit)
n_steps = 500   # Number of simulation steps
T = 300         # Temperature in Kelvin to compare

# Define the Boltzmann distribution (theoretical PDF) for each axis
def boltzmann_distribution(x, k_axis, k_B, T):
    normalization_factor = np.sqrt(k_axis / (2 * np.pi * k_B * T))
    return normalization_factor * np.exp(-k_axis * x**2 / (2 * k_B * T))

# Langevin simulation function for 3D
def langevin_simulation_3D(T, k_x, k_y, k_z, gamma, dt, n_steps):
    x, y, z = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
    for i in range(1, n_steps):
        random_force_x = np.sqrt(2 * k_B * T * gamma / dt) * np.random.randn()
        random_force_y = np.sqrt(2 * k_B * T * gamma / dt) * np.random.randn()
        random_force_z = np.sqrt(2 * k_B * T * gamma / dt) * np.random.randn()

        x[i] = x[i-1] - (k_x / gamma) * x[i-1] * dt + random_force_x * dt
        y[i] = y[i-1] - (k_y / gamma) * y[i-1] * dt + random_force_y * dt
        z[i] = z[i-1] - (k_z / gamma) * z[i-1] * dt + random_force_z * dt
    
    return x, y, z

# Simulate 3D trajectories for T = 300K
num_trajectories = 10000
steps_to_plot = [10, 50, 100, 200, 500]
positions_at_steps = {step: {'x': [], 'y': [], 'z': []} for step in steps_to_plot}

for i_ in range(num_trajectories):
    x_sim, y_sim, z_sim = langevin_simulation_3D(T, k_x, k_y, k_z, gamma, dt, n_steps)
    for step in steps_to_plot:
        positions_at_steps[step]['x'].append(x_sim[step - 1])
        positions_at_steps[step]['y'].append(y_sim[step - 1])
        positions_at_steps[step]['z'].append(z_sim[step - 1])

# Prepare the x values for the theoretical PDF with a larger range
x_values = np.linspace(-35, 35, 400)  # Expand Range for x values
pdf_theory = boltzmann_distribution(x_values, k_y, k_B, T)  

# Create a panel of 5 figures
fig, axs = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
fig.suptitle('Position Distributions at Different Steps vs Boltzmann Distribution (T=300k)', fontsize=17)

for i, step in enumerate(steps_to_plot):
    # Plot histogram of positions at each step (simulated PDF)
    axs[i].hist(positions_at_steps[step], bins=50, density=True, alpha=0.6, color='b', label='Simulated PDF')
    
    # Plot the theoretical Boltzmann distribution
    axs[i].plot(x_values, pdf_theory, 'r--', lw=2, label='Boltzmann Distribution')

    # Titles and labels
    axs[i].set_title(f'Step {step}')
    axs[i].set_xlabel('Position (arb. units)')
    axs[i].grid(True)

# Add y-axis label to the leftmost subplot
axs[0].set_ylabel('Probability Density')

# Display the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for the main title
plt.show()
