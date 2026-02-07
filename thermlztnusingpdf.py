import numpy as np
import matplotlib.pyplot as plt

# Function to simulate 1D Brownian motion of a trapped particle
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

# Function to calculate velocity from position (finite difference)
def calculate_velocity(x, Dt):
    v = np.diff(x) / Dt  # Velocity as change in position over time step
    return v

# Function to compute kinetic energy from velocity
def kinetic_energy(v, mass):
    return 0.5 * mass * v**2

# Function to simulate and plot kinetic energy distribution
def simulate_thermalization_distribution(N, Dt, R, T, eta, kx, time_steps):
    kB = 1.38e-23  # Boltzmann constant [J/K]
    m = 4/3 * np.pi * R**3 * 2650  # Mass of the particle (assuming density of silica ~ 2650 kg/m³)

    plt.figure(figsize=(8, 5))
    
    for step in time_steps:
        x = trapped_1D(step, Dt, 0.0, R, T, eta, kx)
        v = calculate_velocity(x, Dt)
        E = kinetic_energy(v, m)
        
        # Create histogram of kinetic energy
        plt.hist(E, bins=30, density=True, alpha=0.5, label=f'Time = {step * Dt:.3f}s')

    plt.title('Kinetic Energy Distribution at Different Time Steps')
    plt.xlabel('Kinetic Energy (J)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()

# Parameters
N = int(1 / 0.001)  # Number of steps for 1 second with timestep 0.001s
Dt = 0.001  # Time step in seconds
R = 1e-6  # Particle radius in meters
T = 300  # Temperature in Kelvin
eta = 0.001  # Fluid viscosity in Pa.s
kx = 1.0e-6  # Trap stiffness in fN/nm (adjust as needed)

# Define time steps (in number of iterations)
time_steps = [int(0.1 / Dt), int(0.5 / Dt), int(1.0 / Dt)]  # 0.1s, 0.5s, 1.0s

# Run the simulation and plot
simulate_thermalization_distribution(N, Dt, R, T, eta, kx, time_steps)

