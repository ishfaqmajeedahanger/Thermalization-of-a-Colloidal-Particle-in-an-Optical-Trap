import numpy as np
import matplotlib.pyplot as plt

# Constants
k = 1e-6       # Spring constant (N/m)
k_B = 1.38e-23 # Boltzmann constant (J/K)
gamma = 2e-8   # Friction coefficient (Ns/m)
dt = 0.0001    # Time step (s)
T_start = 300  # Starting temperature (K)
T_end = 500    # Ending temperature (K)
n_steps = 500  # Number of heating steps

# Temperature rate of change per step
dT_dt = (T_end - T_start) / (n_steps * dt)  # Rate of temperature change over time

# Calculate entropy production rate over time
entropy_production_rates = []
temperatures = np.linspace(T_start, T_end, n_steps)
for T in temperatures:
    mean_square_velocity = k_B * T / gamma  # <v^2> = k_B * T / m (since m = gamma / dt)
    entropy_production_rate = mean_square_velocity * (dT_dt / T) * (gamma / k_B)
    entropy_production_rates.append(entropy_production_rate)

# Plotting entropy production rate over time
time_points = np.arange(0, n_steps * dt, dt)
plt.figure(figsize=(8, 5))
plt.plot(time_points, entropy_production_rates, label='Entropy Production Rate')
plt.xlabel('Time (s)')
plt.ylabel('Entropy Production Rate (W/K)')
plt.title('Entropy Production Rate during Gradual Heating')
plt.grid(True)
plt.legend()
plt.show()



