import numpy as np
import matplotlib.pyplot as plt

# Define constants
k = 1e-6       # Spring constant (N/m)
k_B = 1.38e-23 # Boltzmann constant (J/K)
gamma = 2e-8   # Friction coefficient (Ns/m)
dt = 0.0001    # Time step (s)
pre_equilibration_steps = 3000  # Pre-equilibration steps at initial 150 K

# Temperatures
T_start = 150  # Starting temperature (K)
T_max = 450    # Maximum temperature (K)
T_end = 150    # End temperature after cooling (K)

# Simulation steps for each phase
n_steps_heating = 3000   # Number of steps for heating phase
n_steps_cooling = 3000   # Number of steps for cooling phase
n_steps_cycle = pre_equilibration_steps + n_steps_heating + n_steps_cooling
total_steps = n_steps_cycle * 3  # Total steps for three cycles

# Langevin simulation function
def langevin_simulation_multi_cycles(T_start, T_max, T_end, k, gamma, dt, n_steps_heating, n_steps_cooling, pre_equilibration_steps, initial_position):
    x = np.zeros(total_steps)
    x[0] = initial_position  # Start at rest

    # Define temperature sequence for each cycle
    temperatures = [T_start] * pre_equilibration_steps + \
                   np.linspace(T_start, T_max, n_steps_heating).tolist() + \
                   np.linspace(T_max, T_end, n_steps_cooling).tolist()

    for cycle in range(3):
        offset = cycle * n_steps_cycle
        for i in range(1, n_steps_cycle):
            T_current = temperatures[i]
            random_force = np.sqrt(2 * k_B * T_current * gamma / dt) * np.random.randn()
            x[offset + i] = x[offset + i - 1] - (k / gamma) * x[offset + i - 1] * dt + random_force * dt / gamma

    return x

# Simulate trajectories and calculate energies
num_trajectories = 10  # Reduce number of trajectories for clearer plotting
all_trajectories = []
all_energies = []

for _ in range(num_trajectories):
    initial_position = 0
    x_sim = langevin_simulation_multi_cycles(T_start, T_max, T_end, k, gamma, dt, n_steps_heating, n_steps_cooling, pre_equilibration_steps, initial_position)
    all_trajectories.append(x_sim)
    # Calculate potential energy for each position
    energy = 0.5 * k * x_sim**2
    all_energies.append(energy)

# Plot Trajectory and Energy over time for each trajectory
time_array = np.arange(total_steps) * dt  # Time array for x-axis

fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Sample Trajectories and Energies of Brownian Particle Across Three Cycles', fontsize=14)

# Plot position trajectory for each sample trajectory
for i, trajectory in enumerate(all_trajectories):
    axs[0].plot(time_array, trajectory, alpha=0.6, label=f'Trajectory {i+1}')
axs[0].set_ylabel('Position (m)')
axs[0].set_title('Sample Trajectories of Brownian Particle')
axs[0].legend(loc='upper right')
axs[0].grid()

# Plot energy trajectory for each sample trajectory
for i, energy in enumerate(all_energies):
    axs[1].plot(time_array, energy, alpha=0.6, label=f'Trajectory {i+1}')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Energy (J)')
axs[1].set_title('Potential Energy of Brownian Particle as Function of Time')
axs[1].legend(loc='upper right')
axs[1].grid()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()













