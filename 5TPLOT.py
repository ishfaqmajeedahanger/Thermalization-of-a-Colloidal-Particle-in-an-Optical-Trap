import numpy as np
import matplotlib.pyplot as plt

# Define constants for the specific case
k = 1       # Spring constant (arbitrary unit)
k_B = 1     # Boltzmann constant (arbitrary unit)
gamma = 1   # Friction coefficient (arbitrary unit)
dt = 0.01   # Time step (arbitrary unit)
n_steps = 500 # Number of simulation steps
T = 5       # Temperature to compare

# Langevin simulation function
def langevin_simulation(T, k, gamma, dt, n_steps):
    x = np.zeros(n_steps)
    for i in range(1, n_steps):
        random_force = np.sqrt(2 * k_B * T * gamma / dt) * np.random.randn()
        x[i] = x[i-1] - (k / gamma) * x[i-1] * dt + random_force * dt
    return x

# Simulate trajectories for T = 5
num_trajectories = 10  # Number of trajectories to visualize
trajectories = []

plt.figure(figsize=(10, 6))

for _ in range(num_trajectories):
    # Generate a new trajectory for each experiment
    x_sim = langevin_simulation(T, k, gamma, dt, n_steps)
    trajectories.append(x_sim)
    plt.plot(np.arange(500) * dt, x_sim[:500], lw=1, alpha=0.3)  # Plot only the first 500 steps

# Titles and labels
plt.title(f'Trajectories of a Brownian Particle in Harmonic Potential (T={T}) - First 500 Steps')
plt.xlabel('Time (arbitrary units)')
plt.ylabel('Position (arbitrary units)')
plt.grid(True)

# Show plot
plt.show()


