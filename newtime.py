import numpy as np
import matplotlib.pyplot as plt

# Define constants
k = 1e-6       # Spring constant (N/m)
k_B = 1.38e-23 # Boltzmann constant (J/K)
gamma = 2e-8   # Friction coefficient (Ns/m)
dt = 0.001     # Time step (s)
n_steps = 500  # Number of simulation steps
T = 300        # Temperature (K)

# Langevin simulation function
def langevin_simulation(T, k, gamma, dt, n_steps):
    x = np.zeros(n_steps)
    for i in range(1, n_steps):
        random_force = np.sqrt(2 * k_B * T * gamma / dt) * np.random.randn()
        x[i] = x[i-1] - (k / gamma) * x[i-1] * dt + random_force * dt
    return x

# Simulate trajectories
num_trajectories = 10000  # Number of trajectories
steps_to_plot = [10, 50, 100, 200, 500]  # Time steps to plot distributions
positions_at_steps = {step: [] for step in steps_to_plot}

for _ in range(num_trajectories):
    x_sim = langevin_simulation(T, k, gamma, dt, n_steps)
    for step in steps_to_plot:
        positions_at_steps[step].append(x_sim[step - 1])  # Collect positions at specified time steps

# Plot the distribution of positions at selected time steps
fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
fig.suptitle('Position Distributions at Different Times (T=300K)', fontsize=17)

for i, step in enumerate(steps_to_plot):
    time = step * dt  # Calculate time in seconds
    axs[i].hist(positions_at_steps[step], bins=50, density=True, alpha=0.7, color='b', label=f'Time = {time:.3f} s')
    
    # Titles and labels
    axs[i].set_title(f'Step {step} ({time:.3f} s)')
    axs[i].set_xlabel('Position (m)')
    axs[i].grid(True)

# Add y-axis label to the leftmost subplot
axs[0].set_ylabel('Probability Density')                                                                    

# Display the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for the main title
plt.show()
