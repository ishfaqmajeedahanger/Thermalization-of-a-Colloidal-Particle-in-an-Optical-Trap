import numpy as np
import matplotlib.pyplot as plt

# Define constants
k = 1e-6       # Spring constant (N/m)
k_B = 1.38e-23 # Boltzmann constant (J/K)
gamma = 2e-8   # Friction coefficient (Ns/m)
dt = 0.0001    # Time step (s)
n_steps = 500  # Number of simulation steps (heating phase)
T_start = 300  # Starting temperature (K)
T_end = 500    # Ending temperature (K)
pre_equilibration_steps = 10  # Number of steps at 300 K before heating

# Total simulation steps including pre-equilibration
total_steps = pre_equilibration_steps + n_steps

# Langevin simulation function with gradual heating 
def langevin_simulation_with_gradual_heating(T_start, T_end, k, gamma, dt, n_steps, pre_equilibration_steps, initial_position):
    x = np.zeros(total_steps)
    x[0] = initial_position  # Start at a thermal state
    
    # Pre-equilibration phase at T_start
    for i in range(1, pre_equilibration_steps):
        random_force = np.sqrt(2 * k_B * T_start * gamma / dt) * np.random.randn()
        x[i] = x[i-1] - (k / gamma) * x[i-1] * dt + random_force * dt/gamma
    
    # Gradual Heating phase
    for i in range(pre_equilibration_steps, total_steps):
        T_current = T_start + (T_end - T_start) * (i - pre_equilibration_steps) / n_steps
        random_force = np.sqrt(2 * k_B * T_current * gamma / dt) * np.random.randn() 
        x[i] = x[i-1] - (k / gamma) * x[i-1] * dt + random_force * dt/gamma
    
    return x

# Simulate trajectories and calculate average potential energy
num_trajectories = 25000  # Number of trajectories
all_trajectories = np.zeros((num_trajectories, total_steps))  # Store all trajectories

# Run the simulation for each trajectory
for j in range(num_trajectories):
    initial_position = 0  # Start each trajectory from zero position
    all_trajectories[j] = langevin_simulation_with_gradual_heating(T_start, T_end, k, gamma, dt, n_steps, pre_equilibration_steps, initial_position)

# Calculate average potential energy at each time step in units of k_B
average_potential_energy_kB = np.zeros(total_steps)

for i in range(total_steps):
    potential_energy = 0.5 * k * all_trajectories[:, i]**2  # Potential energy for each trajectory at step i
    average_potential_energy_kB[i] = np.mean(potential_energy) / k_B  # Average and normalize by k_B

# Plotting the average potential energy in units of k_B as a function of time
plt.figure(figsize=(10, 6))
time_values = np.arange(total_steps) * dt  # Time values for x-axis
plt.plot(time_values, average_potential_energy_kB, color='b', lw=2)
plt.axhline(150, color='r', linestyle='--', lw=2, label='Theoretical Avg Energy at 300k (150 k_B)')
plt.axhline(250, color='g', linestyle='--', lw=2, label='Theoretical Avg Energy at 500k (250 k_B)')
plt.title('Average Potential Energy (in units of k_B) as a Function of Time', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Average Potential Energy (k_B)', fontsize=12)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plotting a few sample trajectories to show behavior over time
num_sample_trajectories = 1000  # Number of trajectories to plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot sample trajectories
for i in range(num_sample_trajectories):
    ax.plot(np.arange(total_steps) * dt, all_trajectories[i], alpha=0.1)

# Add labels and title
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (m)')
ax.set_title('Sample Langevin Trajectories (Heating from 300K to 500K)')
ax.grid(True)
plt.tight_layout()
plt.show()

# Plot the position distributions at selected steps
steps_to_plot = [20, 120, 290, 360, 440, 495]  # Specific steps to plot
positions_at_steps = {step: all_trajectories[:, step] for step in steps_to_plot}

fig, axs = plt.subplots(1, len(steps_to_plot), figsize=(20, 4), sharey=True)
fig.suptitle('Position Distributions at Different Steps (Heating from 300K to 500K)', fontsize=17)

for i, step in enumerate(steps_to_plot):
    axs[i].hist(positions_at_steps[step], bins=90, density=True, alpha=0.8, color='b', label=f'Step {step}')
    
    # Plot theoretical Boltzmann distribution at starting and ending temperatures
    x_values = np.linspace(min(positions_at_steps[step]), max(positions_at_steps[step]), 100)
    
    # Boltzmann distributions at T_start and T_end
    def boltzmann_distribution(x, T):
        return np.sqrt(k / (2 * np.pi * k_B * T)) * np.exp(-k * x**2 / (2 * k_B * T))
    
    axs[i].plot(x_values, boltzmann_distribution(x_values, T_start), 'r--', lw=2, label='T_start (300K)' if i == 0 else "")
    axs[i].plot(x_values, boltzmann_distribution(x_values, T_end), 'g--', lw=2, label='T_end (500K)' if i == 0 else "")
    
    axs[i].set_title(f'Step {step}')
    axs[i].set_xlabel('Position (m)')
    axs[i].grid(True)

# Add y-axis label and legend
axs[0].set_ylabel('Probability Density')
plt.legend()
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
