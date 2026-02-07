import numpy as np
import matplotlib.pyplot as plt

# Define constants
k = 1e-6       # Spring constant (N/m)
k_B = 1.38e-23 # Boltzmann constant (J/K)
gamma = 2e-8   # Friction coefficient (Ns/m)
dt = 0.0001    # Time step (s)
n_steps = 500 # Number of simulation steps (heating phase)
T_start = 300  # Starting temperature (K)
T_end = 500    # Ending temperature (K) - updated to 500 K for this thermalization
pre_equilibration_steps = 10 # Number of steps at 300 K before heating

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

# Simulate trajectories
num_trajectories = 25000  # Number of trajectories
steps_to_plot = [20, 120, 290, 360, 440, 495]  # Steps to compare: early, mid, late steps
positions_at_steps = {step: [] for step in steps_to_plot}

# Run the simulation
for _ in range(num_trajectories):
    initial_position = 0 
    x_sim = langevin_simulation_with_gradual_heating(T_start, T_end, k, gamma, dt, n_steps, pre_equilibration_steps, initial_position)
    for step in steps_to_plot:
        positions_at_steps[step].append(x_sim[step ])  # Collect position at specified steps

# Collect all position data for the final step (495)
position_values = np.array([pos for pos in positions_at_steps[495]])
min_pos, max_pos = position_values.min(), position_values.max()

# Define the Boltzmann distribution with normalization
def boltzmann_distribution(x, k, k_B, T):
    normalization_factor = np.sqrt(k / (2 * np.pi * k_B * T))
    exp_part = np.exp(-k * x**2 / (2 * k_B * T))
    return normalization_factor * exp_part

# Plot the simulated and theoretical distributions at various steps
fig, axs = plt.subplots(1, len(steps_to_plot), figsize=(20, 4), sharey=True)
fig.suptitle('Position Distributions at Different Steps when heating from 300k to 500k vs Boltzmann Distribution', fontsize=17)

for i, step in enumerate(steps_to_plot):
    
    # Prepare the x values for the theoretical PDF based on the current temperature
    x_values = np.linspace(min_pos, max_pos, 100)
    pdf_theory_start = boltzmann_distribution(x_values, k, k_B, T_start)  
    normalization_factor = np.trapz(pdf_theory_start, x_values)
    pdf_theory_start /= normalization_factor  # Normalize the theoretical PDF
    axs[i].plot(x_values, pdf_theory_start, 'r--', lw=2) 
    
    pdf_theory_end = boltzmann_distribution(x_values, k, k_B, T_end)  
    normalization_factor = np.trapz(pdf_theory_end, x_values)
    pdf_theory_end /= normalization_factor  # Normalize the theoretical PDF
    axs[i].plot(x_values, pdf_theory_end, 'b--', lw=2) 

    # Plot histogram of positions at each step (simulated PDF)
    axs[i].hist(positions_at_steps[step], bins=90, density=True, alpha=0.8, color='b', label=f'Simulated PDF (Step {step})')
    
    # Set title for each subplot with the respective step
    axs[i].set_title(f'Step {step}')  # Add this line to label each subplot
    axs[i].set_xlabel('Position (m)')
    axs[i].grid(True)
   
    # Titles and labels
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
    initial_position = np.zeros(1)  # Start at rest
    x_sim = langevin_simulation_with_gradual_heating(T_start, T_end, k, gamma, dt, n_steps, pre_equilibration_steps, initial_position)
    ax.plot(np.arange(total_steps) * dt, x_sim)

# Adding vertical lines corresponding to the steps_to_plot
for step in steps_to_plot:
    ax.axvline(step * dt, color='gray', linestyle='--', linewidth=1.0, label=f'Step {step} ({step * dt:.2f}s)' if step == steps_to_plot[0] else "")

# Labels and title
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (m)')
ax.set_title('Sample Langevin Trajectories (Heating from 300K to 500K)')
ax.grid(True)
plt.show()
plt.legend()
plt.show() 