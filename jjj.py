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

# Choose two time points to plot
time_points = [100, 500]  # Steps corresponding to time points 0.1s and 0.5s
num_sample_trajectories = 1000  # Number of trajectories to plot

# Plot a few sample trajectories at the two time points
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

for i, step in enumerate(time_points):
    for _ in range(num_sample_trajectories):
        x_sim = langevin_simulation(T, k, gamma, dt, n_steps)
        axs[i].plot(np.arange(step) * dt, x_sim[:step], alpha=0.1, color='b')  # Plot up to the chosen step

    # Labels and title for each subplot
    axs[i].set_xlabel('Time (s)')
    axs[i].set_ylabel('Position (m)')
    axs[i].set_title(f'Trajectories up to Step {step} (Time = {step * dt:.3f} s)')
    axs[i].grid(True)

# Display the plot
plt.tight_layout()
plt.show()
