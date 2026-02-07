import numpy as np
import matplotlib.pyplot as plt

def trapped_1D(N, Dt, x1, R, T, eta, kx):
    kB = 1.38e-23  # Boltzmann constant [J/K]
    gamma = 6 * np.pi * R * eta  # friction coefficient
    D = kB * T / gamma  # diffusion coefficient

    x = np.zeros(N)
    x[0] = x1  # initial condition for x

    for i in range(1, N):
        # Deterministic step
        x[i] = x[i-1] - (kx * Dt / gamma) * x[i-1]

        # Diffusive step (random thermal noise)
        x[i] += np.sqrt(2 * D * Dt) * np.random.randn()

    return x

def plot_multiple_trajectories_1D(num_experiments, N, Dt, R, T, eta, kx):
    plt.figure(figsize=(10, 6))

    time_limit = int(0/ Dt)  # Corresponds to 0.04 seconds

    for _ in range(num_experiments):
        # Generate a new 1D trajectory for each experiment
        x = trapped_1D(N, Dt, 0.0, R, T, eta, kx)
        plt.plot(np.arange(time_limit) * Dt, x[:time_limit] * 1e9, lw=1, alpha=0.3)  # Convert meters to nanometers

    plt.title('1000 Trajectories of a Brownian Particle in Optical Trap (X-direction)')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (nm)')
    plt.xticks([0])  # Set ticks at 0 seconds
    plt.xlim(0)  # Set x-axis limit to 0s
    plt.grid(True)
    plt.show()

# Parameters for the simulation
N = int(1 / 0.001)  # Number of steps for 1 second with timestep 0.001s
Dt = 0.001  # Time step in seconds
R = 1e-6  # Particle radius in meters
T = 300  # Temperature in Kelvin
eta = 0.001  # Fluid viscosity in Pa.s
kx = 1.0e-6  # Trap stiffness in fN/nm (you can adjust this value as needed)

# Number of experiments (trajectories)
num_experiments = 1000

# Plot the results of 1000 experiments
plot_multiple_trajectories_1D(num_experiments, N, Dt, R, T, eta, kx)
