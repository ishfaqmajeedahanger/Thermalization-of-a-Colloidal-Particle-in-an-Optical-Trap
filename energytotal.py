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
total_steps = n_steps_cycle * 10  # Total steps for ten cycles

# Langevin simulation function
def langevin_simulation_multi_cycles(T_start, T_max, T_end, k, gamma, dt, n_steps_heating, n_steps_cooling, pre_equilibration_steps, initial_position):
    x = np.zeros(total_steps)
    x[0] = initial_position  # Start at rest

    # Define temperature sequence for each cycle
    temperatures = [T_start] * pre_equilibration_steps + \
                   np.linspace(T_start, T_max, n_steps_heating).tolist() + \
                   np.linspace(T_max, T_end, n_steps_cooling).tolist()

    for cycle in range(10):  # Simulate 10 cycles
        offset = cycle * n_steps_cycle
        
        # Carry over the last position from the previous cycle except for the first cycle 
        if cycle > 0:
            x[offset] = x[offset - n_steps_cycle]  # Start from the end of the previous cycle
        
        for i in range(1, n_steps_cycle):
            T_current = temperatures[i]
            random_force = np.sqrt(2 * k_B * T_current * gamma / dt) * np.random.randn()
            x[offset + i] = x[offset + i - 1] - (k / gamma) * x[offset + i - 1] * dt + random_force * dt / gamma

    return x

# Simulate trajectories and store positions
num_trajectories = 100  # Simulate 100 trajectories
all_trajectories = np.zeros((num_trajectories, total_steps))  # Initialize an array to store trajectories

for j in range(num_trajectories):
    initial_position = 0 if j == 0 else all_trajectories[j-1, -1]  # First trajectory starts from 0, others from last position
    all_trajectories[j] = langevin_simulation_multi_cycles(T_start, T_max, T_end, k, gamma, dt, n_steps_heating, n_steps_cooling, pre_equilibration_steps, initial_position)

# Calculate average potential energy as a function of time
average_potential_energy = np.zeros(total_steps)

for i in range(total_steps):
    potential_energy = 0.5 * k * all_trajectories[:, i]**2  # Calculate potential energy for each trajectory
    average_potential_energy[i] = np.mean(potential_energy)  # Average over all trajectories

# Plotting the average potential energy as a function of time
plt.figure(figsize=(12, 6))
time_values = np.arange(total_steps) * dt  # Time values for x-axis
plt.plot(time_values, average_potential_energy, color='b', lw=2)
plt.title('Average Potential Energy as a Function of Time', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Average Potential Energy (J)', fontsize=12)
plt.grid()
plt.tight_layout()
plt.show()




