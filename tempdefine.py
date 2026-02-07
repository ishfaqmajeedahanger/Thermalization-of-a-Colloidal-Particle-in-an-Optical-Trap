import numpy as np
import matplotlib.pyplot as plt

# Define constants
k = 1e-6       # Spring constant (N/m)
k_B = 1.38e-23 # Boltzmann constant (J/K)
gamma = 2e-8   # Friction coefficient (Ns/m)
dt = 0.0001    # Time step (s)
n_steps = 500  # Number of simulation steps
T_start = 300  # Starting temperature (K)
T_end = 1000   # Ending temperature (K)

# Temperature function (linear increase from T_start to T_end)
def temperature_function(step, n_steps, T_start, T_end):
    return T_start + (T_end - T_start) * step / n_steps

# Langevin simulation function with time-dependent temperature
def langevin_simulation(T_start, T_end, k, gamma, dt, n_steps):
    x = np.zeros(n_steps)
    for i in range(1, n_steps):
        T = temperature_function(i, n_steps, T_start, T_end)  # Get the temperature at the current step
        random_force = np.sqrt(2 * k_B * T * gamma / dt) * np.random.randn()
        x[i] = x[i-1] - (k / gamma) * x[i-1] * dt + random_force * dt/gamma
    return x

# Simulate trajectories
num_trajectories = 10000  # Number of trajectories
steps_to_plot = [5, 30, 50, 100, 490]  # Steps to compare
positions_at_steps = {step: [] for step in steps_to_plot}

for _ in range(num_trajectories):
    x_sim = langevin_simulation(T_start, T_end, k, gamma, dt, n_steps)
    for step in steps_to_plot:
        positions_at_steps[step].append(x_sim[step - 1])  # Collect position at specified steps

# Collect all position data
position_values = np.array([pos for step in positions_at_steps.values() for pos in step])
min_pos, max_pos = position_values.min(), position_values.max()
print("Position Range:", min_pos, max_pos)  # Debug range of positions

# Define the Boltzmann distribution with normalization
def boltzmann_distribution(x, k, k_B, T):
    normalization_factor = np.sqrt(k / (2 * np.pi * k_B * T))
    exp_part = np.exp(-k * x**2 / (2 * k_B * T))
    return normalization_factor * exp_part

# Calculate the theoretical Boltzmann distribution at T_start (300K)
x_values = np.linspace(min_pos, max_pos, 100)  # Define x range based on simulated positions
pdf_theory_initial = boltzmann_distribution(x_values, k, k_B, T_start)  # Boltzmann PDF for T = 300K

# Normalize the theoretical PDF
normalization_factor_initial = np.trapz(pdf_theory_initial, x_values)
pdf_theory_initial /= normalization_factor_initial

# Plot the histogram of positions at the first step (e.g., step 5)
plt.hist(positions_at_steps[5], bins=50, density=True, alpha=0.6, color='b', label='Simulated PDF (Step 5)')

# Plot the normalized theoretical Boltzmann distribution for T = 300K
plt.plot(x_values, pdf_theory_initial, 'r--', lw=2, label='Boltzmann Distribution (T=300K)')

# Labels and title
plt.xlabel('Position (m)')
plt.ylabel('Probability Density')
plt.title('Position Distribution at Step 5 (Initial Temperature = 300K)')
plt.legend()
plt.grid(True)
plt.show()

# Plot a few sample trajectories
num_sample_trajectories = 1000  # Number of trajectories to plot
fig, ax = plt.subplots(figsize=(10, 6))

# Simulate and plot sample trajectories
for _ in range(num_sample_trajectories):
    x_sim = langevin_simulation(T_start, T_end, k, gamma, dt, n_steps)
    ax.plot(np.arange(n_steps) * dt, x_sim, label=f'Trajectory {_+1}')  # Time on x-axis, position on y-axis

# Labels and title
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (m)')
ax.set_title('Sample Langevin Trajectories (T=300K to 1000K)')
ax.grid(True)
plt.legend()
plt.show()

