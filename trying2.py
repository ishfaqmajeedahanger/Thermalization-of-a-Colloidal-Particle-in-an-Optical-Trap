import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mitcker

# Define constants
k = 1e-6       # Spring constant (N/m)
k_B = 1.38e-23 # Boltzmann constant (J/K)
gamma = 2e-8   # Friction coefficient (Ns/m)
dt = 0.0001    # Time step (s)
pre_equilibration_steps = 1000  # Number of steps at initial 150 K

# Temperatures
T_start = 150  # Starting temperature (K)
T_max = 450    # Maximum heating temperature (K)
T_end = 200    # Ending temperature after cooling (K)

# Simulation steps for each phase
n_steps_heating = 1000  # Number of steps for heating phase
n_steps_cooling = 1000  # Number of steps for cooling phase
total_steps = pre_equilibration_steps + n_steps_heating + n_steps_cooling

# Langevin simulation function with heating and cooling
def langevin_simulation_heating_cooling(T_start, T_max, T_end, k, gamma, dt, n_steps_heating, n_steps_cooling, pre_equilibration_steps, initial_position):
    x = np.zeros(total_steps)
    x[0] = initial_position  # Start at rest

    # Pre-equilibration phase at T_start
    for i in range(1, pre_equilibration_steps):
        random_force = np.sqrt(2 * k_B * T_start * gamma / dt) * np.random.randn()
        x[i] = x[i-1] - (k / gamma) * x[i-1] * dt + random_force * dt / gamma

    # Heating phase from T_start to T_max
    for i in range(pre_equilibration_steps, pre_equilibration_steps + n_steps_heating):
        T_current = T_start + (T_max - T_start) * (i - pre_equilibration_steps) / n_steps_heating
        random_force = np.sqrt(2 * k_B * T_current * gamma / dt) * np.random.randn()
        x[i] = x[i-1] - (k / gamma) * x[i-1] * dt + random_force * dt / gamma

    # Cooling phase from T_max to T_end
    for i in range(pre_equilibration_steps + n_steps_heating, total_steps):
        T_current = T_max + (T_end - T_max) * (i - pre_equilibration_steps - n_steps_heating) / n_steps_cooling
        random_force = np.sqrt(2 * k_B * T_current * gamma / dt) * np.random.randn()
        x[i] = x[i-1] - (k / gamma) * x[i-1] * dt + random_force * dt / gamma

    return x

# Boltzmann distribution function
def boltzmann_distribution(x, k, k_B, T):
    normalization_factor = np.sqrt(k / (2 * np.pi * k_B * T))
    exp_part = np.exp(-k * x**2 / (2 * k_B * T))
    return normalization_factor * exp_part

# Simulate trajectories
num_trajectories = 25000  # Number of trajectories
positions_at_steps = {step: [] for step in [940, 980, 1000, 1940, 1980, 2000, 2940, 2980, 2995]}
temperatures_at_steps = {
    940: T_start, 980: T_start, 1000: T_start, 1940: T_max, 1980: T_max, 2000: T_max,
    2940: T_end, 2980: T_end, 2995: T_end
}

# Run the simulation and collect position data
for _ in range(num_trajectories):
    initial_position = 0
    x_sim = langevin_simulation_heating_cooling(T_start, T_max, T_end, k, gamma, dt, n_steps_heating, n_steps_cooling, pre_equilibration_steps, initial_position)
    for step in positions_at_steps.keys():
        if step < total_steps:  # Ensure we don't exceed the array bounds
            positions_at_steps[step].append(x_sim[step])

# Plot a histogram of the positions for each selected step along with the Boltzmann distribution
fig, axs = plt.subplots(1, len(positions_at_steps), figsize=(18, 4), sharey=True)
fig.suptitle('Position Distributions at Different Steps with Boltzmann Distribution (Heating from 150 to 450 and Cooling back to 200)', fontsize=14)

# Plot histograms for each selected step
for i, (step, positions) in enumerate(positions_at_steps.items()):
    # Determine the temperature for the Boltzmann distribution at this step
    T_current = temperatures_at_steps[step]
    
    # Plot histogram of simulated positions
    axs[i].hist(positions, bins=80, density=True, color='b', alpha=0.8, label=f'Simulated PDF (Step {step})')
    
    # Prepare x values for theoretical Boltzmann distribution
    x_values = np.linspace(min(positions), max(positions), 100) if len(positions) > 0 else np.array([0])
    pdf_theory = boltzmann_distribution(x_values, k, k_B, T_current)
    axs[i].plot(x_values, pdf_theory, 'r--', lw=1.5, label=f'Boltzmann PDF at {T_current} K')
    
    # Set title to show only the step
    axs[i].set_title(f'Step {step}', fontsize=8)
    axs[i].set_xlabel('Position (m)', fontsize=8, labelpad=10)
    axs[i].tick_params(axis='both', which='major', labelsize=5)
    axs[i].grid(True)

# Ensure scientific notation and control font size
formatter = mitcker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))
axs[i].xaxis.set_major_formatter(formatter)
axs[i].xaxis.get_offset_text().set_fontsize(5)

# Set a shared ylabel and adjust layout for better spacing
axs[0].set_ylabel('Probability Density', fontsize=7)
plt.tight_layout(rect=[0, 0, 1, 0.85])
plt.legend(fontsize=8, loc='upper right')
plt.show()

# Plot sample trajectories with marked histogram steps
fig, ax = plt.subplots(figsize=(14, 8))

# Plot several sample trajectories
for _ in range(300):
    x_sim = langevin_simulation_heating_cooling(T_start, T_max, T_end, k, gamma, dt, n_steps_heating, n_steps_cooling, pre_equilibration_steps, initial_position=0)
    ax.plot(np.arange(total_steps) * dt, x_sim, lw=0.4, alpha=0.3)

# Add vertical lines at phase transition points
ax.axvline(pre_equilibration_steps * dt, color='gray', linestyle='--', linewidth=1.2, label="End of Pre-Equilibration")
ax.axvline((pre_equilibration_steps + n_steps_heating) * dt, color='gray', linestyle='--', linewidth=1.2, label="End of Heating Phase")

# Add thick vertical lines at histogram steps
histogram_steps = [940, 980, 1000, 1940, 1980, 2000, 2940, 2980, 2995]
for step in histogram_steps:
    ax.axvline(step * dt, color='purple', linestyle='-', linewidth=2.0, label=f'Histogram Step {step * dt:.3f} s')

# Adjust labels and title
ax.set_xlabel('Time (s)', fontsize=10)
ax.set_ylabel('Position (m)', fontsize=10)
ax.set_title('Sample Langevin Trajectories with Heating and Cooling Phases', fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=5)
ax.grid(True)

# Add legend, showing only unique labels for the histogram steps
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys(), fontsize=6, loc='upper right')
plt.show()







