import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_trajectories = 10000  # Number of trajectories
time_steps = np.linspace(0, 1, 1000)  # Time from 0 to 1 second
cold_water_diffusion = 0.01  # Diffusion coefficient for cold water
normal_water_diffusion = 0.05  # Diffusion coefficient for normal water
hot_water_diffusion = 0.1  # Diffusion coefficient for hot water

# Initialize arrays to store trajectories
cold_water_trajectories = np.zeros((num_trajectories, len(time_steps)))
normal_water_trajectories = np.zeros((num_trajectories, len(time_steps)))
hot_water_trajectories = np.zeros((num_trajectories, len(time_steps)))

# Simulate trajectories
for i in range(num_trajectories):
    # Cold water
    cold_water_trajectories[i] = np.sqrt(2 * cold_water_diffusion * time_steps) * np.random.randn(len(time_steps))

    # Normal water
    normal_water_trajectories[i] = np.sqrt(2 * normal_water_diffusion * time_steps) * np.random.randn(len(time_steps))

    # Hot water
    hot_water_trajectories[i] = np.sqrt(2 * hot_water_diffusion * time_steps) * np.random.randn(len(time_steps))

# Plotting the results
plt.figure(figsize=(15, 5))

# Cold Water
plt.subplot(1, 3, 1)
for j in range(0, num_trajectories, 1000):  # Plot every 1000th trajectory
    plt.plot(time_steps, cold_water_trajectories[j], color='blue', alpha=0.1)
plt.title('Ink Diffusion in Cold Water')
plt.xlabel('Time (s)')
plt.ylabel('Position (arbitrary units)')
plt.xlim(0, 1)
plt.ylim(-0.5, 0.5)

# Normal Water
plt.subplot(1, 3, 2)
for j in range(0, num_trajectories, 1000):
    plt.plot(time_steps, normal_water_trajectories[j], color='orange', alpha=0.1)
plt.title('Ink Diffusion in Normal Water')
plt.xlabel('Time (s)')
plt.xlim(0, 1)
plt.ylim(-0.5, 0.5)

# Hot Water
plt.subplot(1, 3, 3)
for j in range(0, num_trajectories, 1000):
    plt.plot(time_steps, hot_water_trajectories[j], color='red', alpha=0.1)
plt.title('Ink Diffusion in Hot Water')
plt.xlabel('Time (s)')
plt.xlim(0, 1)
plt.ylim(-0.5, 0.5)

# Adjust layout
plt.tight_layout()
plt.show()
