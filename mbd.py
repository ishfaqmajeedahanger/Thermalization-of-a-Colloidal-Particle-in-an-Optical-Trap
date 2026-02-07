import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell

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

# Function to simulate multiple trajectories and plot the kinetic energy distribution
def simulate_thermalization(num_experiments, N, Dt, R, T, eta, kx):
    kB = 1.38e-23  # Boltzmann constant [J/K]
    m = 4/3 * np.pi * R**3 * 2650  # Mass of the particle (assuming density of silica ~ 2650 kg/m³)

    energies = []

    for i_ in range(num_experiments):
        # Simulate trajectory
        x = trapped_1D(N, Dt, 0.0, R, T, eta, kx)
        
        # Calculate velocity and kinetic energy
        v = calculate_velocity(x, Dt)
        E = kinetic_energy(v, m)
        
        # Store the kinetic energy at the final time step
        energies.append(E[-1])

    # Convert energies to numpy array
    energies = np.array(energies)

    # Create bins for histogram
    energy_bins = np.linspace(0, np.max(energies), 50)  # Increased number of bins

    # Simulated PDF (histogram of kinetic energy)
    plt.figure(figsize=(10, 6))
    hist, bins, _ = plt.hist(energies, bins=energy_bins, density=True, alpha=0.6, color='b', label='Simulated PDF')

    # Maxwell-Boltzmann theoretical distribution for kinetic energy
    v_theory = np.linspace(0, 2 * np.sqrt(kB * T / m), 1000)  # Adjust velocity range for theory
    E_theory = 0.5 * m * v_theory**2  # Kinetic energy corresponding to these velocities
    
    # Maxwell-Boltzmann PDF in terms of energy
    mb_dist = (2 * np.sqrt(E_theory / (np.pi * kB * T)) *
               np.exp(-E_theory / (kB * T)) / (kB * T))  
    
    # Normalize the Maxwell-Boltzmann distribution to match the histogram
    mb_dist /= np.trapz(mb_dist, E_theory)  # Normalize the integral to 1
    mb_dist *= np.trapz(hist, bins[: -1])  # Scale to match the area under the histogram

    # Plot Maxwell-Boltzmann PDF
    plt.plot(E_theory, mb_dist, 'r-', lw=2, label='Maxwell-Boltzmann Distribution')

    # Optional: Set y-axis to logarithmic scale for better visibility
    # plt.yscale('log')

    plt.title('Thermalization: Kinetic Energy Distribution vs Maxwell-Boltzmann')
    plt.xlabel('Kinetic Energy (J)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, np.max(energies))  # Match x-axis to the range of energies
    plt.ylim(bottom=0)  # Set lower limit of y-axis
    plt.show()

# Simulation parameters
N = int(1 / 0.001)  # Number of steps for 1 second with timestep 0.001s
Dt = 0.001  # Time step in seconds
R = 1e-6  # Particle radius in meters
T = 300  # Temperature in Kelvin
eta = 0.001  # Fluid viscosity in Pa.s
kx = 1.0e-6  # Trap stiffness in fN/nm (adjust as needed)

# Number of experiments (trajectories)
num_experiments = 1000

# Simulate thermalization and compare with Maxwell-Boltzmann distribution
simulate_thermalization(num_experiments, N, Dt, R, T, eta, kx)










