import numpy as np
import matplotlib.pyplot as plt

# Define constants
k = 1e-6       # Spring constant (N/m)
k_B = 1.38e-23 # Boltzmann constant (J/K)
gamma = 2e-8   # Friction coefficient (Ns/m)
dt = 0.0001     # Time step (s)
n_steps = 500  # Number of simulation steps
T = 300        # Temperature (K)

# Langevin simulation function
def langevin_simulation(T, k, gamma, dt, n_steps):
    x = np.zeros(n_steps)
    for i in range(1, n_steps):
        random_force = np.sqrt(2 * k_B * T * gamma / dt) * np.random.randn()
        x[i] = x[i-1] - (k / gamma) * x[i-1] * dt + random_force * dt/gamma
    return x

# Simulate trajectories
num_trajectories = 10000  # Number of trajectories
steps_to_plot = [5, 30, 50, 100, 490]  # Steps to compare
positions_at_steps = {step: [] for step in steps_to_plot}

for _ in range(num_trajectories):
    x_sim = langevin_simulation(T, k, gamma, dt, n_steps)
    for step in steps_to_plot:
        positions_at_steps[step].append(x_sim[step - 1])  # Collect position at specified steps

# Collect all position data
position_values = np.array([pos for step in positions_at_steps.values() for pos in step])
min_pos, max_pos = position_values.min(), position_values.max()
print("Position Range:", min_pos, max_pos)  # Debug range of positions

# Prepare the x values for the theoretical PDF based on min and max positions
x_values = np.linspace(min_pos, max_pos, 100)

# Define the Boltzmann distribution with normalization
def boltzmann_distribution(x, k, k_B, T):
    normalization_factor = np.sqrt(k / (2 * np.pi * k_B * T))
    exp_part = np.exp(-k * x**2 / (2 * k_B * T))
    return normalization_factor * exp_part

# Calculate the theoretical Boltzmann distribution
pdf_theory = boltzmann_distribution(x_values, k, k_B, T)

# Normalize the theoretical PDF
normalization_factor = np.trapz(pdf_theory, x_values)
pdf_theory /= normalization_factor  # Normalize the theoretical PDF
print("Normalized PDF (First 10 values):", pdf_theory[:10])  # Debug normalized PDF

# Create a panel of 5 figures
fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
fig.suptitle('Position Distributions at Different Steps vs Boltzmann Distribution (T=300K)', fontsize=17)

for i, step in enumerate(steps_to_plot):
    # Plot histogram of positions at each step (simulated PDF)
    axs[i].hist(positions_at_steps[step], bins=50, density=True, alpha=0.6, color='b', label='Simulated PDF')
    
    # Plot the normalized theoretical Boltzmann distribution
    axs[i].plot(x_values, pdf_theory, 'r--', lw=2, label='Boltzmann Distribution')

    # Titles and labels
    axs[i].set_title(f'Step {step}')
    axs[i].set_xlabel('Position (m)')
    axs[i].grid(True)

# Add y-axis label to the leftmost subplot
axs[0].set_ylabel('Probability Density')

# Display the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for the main title
plt.legend()
plt.show()

# Plot a few sample trajectories
num_sample_trajectories = 1000  # Number of trajectories to plot
fig, ax = plt.subplots(figsize=(10, 6))

# Simulate and plot sample trajectories
for _ in range(num_sample_trajectories):
    x_sim = langevin_simulation(T, k, gamma, dt, n_steps)
    ax.plot(np.arange(n_steps) * dt, x_sim, label=f'Trajectory {_+1}')  # Time on x-axis, position on y-axis

# Labels and title
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (m)')
ax.set_title('Sample Langevin Trajectories (T=300K)')
ax.grid(True)
plt.legend()
plt.show()

