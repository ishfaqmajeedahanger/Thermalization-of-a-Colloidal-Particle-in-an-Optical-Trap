import numpy as np
import matplotlib.pyplot as plt

# Define constants
k = 1e-6       # Spring constant (N/m)
k_B = 1.38e-23 # Boltzmann constant (J/K)
gamma = 2e-8   # Friction coefficient (Ns/m)
dt = 0.0001    # Time step (s)
n_steps = 500  # Number of simulation steps
T_start = 300  # Starting temperature (K)
T_end = 1000 # Ending temperature (K)

# Temperature function (linear increase from T_start to T_end)
def temperature_function(step, n_steps, T_start, T_end):
    # Make sure to correctly scale the temperature to the last step
    return T_start + (T_end - T_start) * (step) / (n_steps - 1)

# Langevin simulation function with time-dependent temperature
def langevin_simulation(T_start, T_end, k, gamma, dt, n_steps):
    x = np.zeros(n_steps)
    for i in range(1, n_steps):
        T = temperature_function(i, n_steps, T_start, T_end)  # Get the temperature at the current step
        random_force = np.sqrt(2 * k_B * T * gamma / dt) * np.random.randn()
        x[i] = x[i-1] - (k / gamma) * x[i-1] * dt + random_force * dt / gamma
    return x

# Simulate trajectories
num_trajectories = 10000  # Number of trajectories
steps_to_plot = [5, 30, 50, 100, 490]  # Steps to compare: early, mid, late steps
positions_at_steps = {step: [] for step in steps_to_plot}

for _ in range(num_trajectories):
    x_sim = langevin_simulation(T_start, T_end, k, gamma, dt, n_steps)
    for step in steps_to_plot:
        positions_at_steps[step].append(x_sim[step - 1])  # Collect position at specified steps

# Collect all position data for the final step (490)
position_values = np.array([pos for pos in positions_at_steps[490]])
min_pos, max_pos = position_values.min(), position_values.max()

# Define the Boltzmann distribution with normalization
def boltzmann_distribution(x, k, k_B, T):
    normalization_factor = np.sqrt(k / (2 * np.pi * k_B * T))
    exp_part = np.exp(-k * x**2 / (2 * k_B * T))
    return normalization_factor * exp_part

# Plot the simulated and theoretical distributions at various steps
fig, axs = plt.subplots(1, len(steps_to_plot), figsize=(20, 4), sharey=True)
fig.suptitle('Position Distributions at Different Steps vs Boltzmann Distribution', fontsize=17)

for i, step in enumerate(steps_to_plot):
    T_current = temperature_function(step, n_steps, T_start, T_end)  # Get temperature at current step
    
    # Prepare the x values for the theoretical PDF based on the current temperature
    x_values = np.linspace(min_pos, max_pos, 100)
    pdf_theory = boltzmann_distribution(x_values, k, k_B, T_current)
    normalization_factor = np.trapz(pdf_theory, x_values)
    pdf_theory /= normalization_factor  # Normalize the theoretical PDF
    
    # Plot histogram of positions at each step (simulated PDF)
    axs[i].hist(positions_at_steps[step], bins=50, density=True, alpha=0.6, color='b', label=f'Simulated PDF (Step {step})')
    
    # Plot the normalized theoretical Boltzmann distribution
    axs[i].plot(x_values, pdf_theory, 'r--', lw=2, label=f'Boltzmann Dist (T={T_current: 1f}K)')
    
    # Titles and labels
    axs[i].set_title(f'Step {step} (T={T_current: 1f}K)')
    axs[i].set_xlabel('Position (m)')
    axs[i].grid(True)

# Add y-axis label to the leftmost subplot
axs[0].set_ylabel('Probability Density')

# Display the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for the main title
plt.legend()
plt.show()

# Plot a few sample trajectories without cluttering the legend
num_sample_trajectories = 1000  # Number of trajectories to plot
fig, ax = plt.subplots(figsize=(10, 6))

# Simulate and plot sample trajectories
for _ in range(num_sample_trajectories):
    x_sim = langevin_simulation(T_start, T_end, k, gamma, dt, n_steps)
    ax.plot(np.arange(n_steps) * dt, x_sim)

# Labels and title
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (m)')
ax.set_title('Sample Langevin Trajectories (Heating from 300K to 1000K)')
ax.grid(True)
plt.show()







