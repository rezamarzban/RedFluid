Below is a Python code implementation using only NumPy (and optionally SciPy, though SciPy is not strictly necessary here) to simulate the deflagration to detonation transition (DDT) of a methane-air mixture in a shockwave tube with obstacle blockages. The simulation employs a 1D finite volume method with the Rusanov flux scheme to handle shocks, includes combustion via a reaction progress variable, and models obstacles as reductions in the tube's cross-sectional area.
Problem Overview
The goal is to simulate a shockwave tube where a high-pressure region (driver section) generates a shock wave that propagates into a low-pressure region (driven section) containing a methane-air mixture. Obstacles in the tube enhance turbulence, potentially accelerating the flame from deflagration (subsonic combustion) to detonation (supersonic combustion). The key components include:
Governing Equations: Euler equations for compressible flow (mass, momentum, energy) plus a reaction progress equation.
Obstacles: Modeled as reductions in cross-sectional area 
A(x)
.
Combustion: Simplified single-step reaction with an Arrhenius reaction rate.
Numerical Method: Finite volume method with Rusanov flux for shock capturing.
Code Implementation
python
import numpy as np
import matplotlib.pyplot as plt  # For visualization (optional)

### Physical and Simulation Parameters
L = 10.0          # Tube length (m)
N = 1000          # Number of grid cells
dx = L / N        # Grid spacing (m)
x = np.linspace(0.5 * dx, L - 0.5 * dx, N)  # Cell centers (m)

# Cross-sectional area A(x): 1.0 normally, 0.5 at obstacles
A = np.ones(N)
A[np.arange(N) % 100 < 10] = 0.5  # Obstacles every 1 m, width 0.1 m

# Gas properties
R = 287.0         # Specific gas constant (J/kg·K)
gamma = 1.4       # Specific heat ratio
cv = R / (gamma - 1)  # Specific heat at constant volume (J/kg·K)
q0 = 2.8e6        # Heat release of methane-air (J/kg)
A_reaction = 1e10 # Pre-exponential factor (s^-1, tune for flame speed)
Ea = 1.5e5        # Activation energy (J/mol)
R_gas = 8.314     # Universal gas constant (J/mol·K)

# Initial conditions
p_high = 10 * 1.01325e5  # Driver pressure (Pa)
p_low = 1 * 1.01325e5    # Driven pressure (Pa)
T0 = 300.0               # Initial temperature (K)
rho_high = p_high / (R * T0)  # High-pressure density (kg/m^3)
rho_low = p_low / (R * T0)    # Low-pressure density (kg/m^3)

# Reaction zone: combustion only for x > 5 m
react_zone = (x > 5.0).astype(float)

### Initialize Conserved Variables
# U = [rho*A, m=rho*u*A, rho*E*A, rho*c*A]
U = np.zeros((4, N))
for i in range(N):
    rho = rho_high if x[i] < 5.0 else rho_low
    U[0, i] = rho * A[i]        # Mass: rho*A
    U[1, i] = 0.0               # Momentum: m = rho*u*A (u=0 initially)
    E = cv * T0 + q0            # Energy: E = cv*T + (1-c)*q0 + 0.5*u^2
    U[2, i] = rho * E * A[i]    # rho*E*A
    U[3, i] = 0.0               # rho*c*A (c=0 initially)

### Helper Functions

def get_primitives(U, A):
    """Convert conserved variables to primitive variables."""
    rho_A = U[0]
    m = U[1]
    rho_E_A = U[2]
    rho_c_A = U[3]
    rho = rho_A / A
    u = m / rho_A
    E = rho_E_A / rho_A
    c = rho_c_A / rho_A
    T = (E - (1 - c) * q0 - 0.5 * u**2) / cv
    p = rho * R * T
    return rho, u, p, c, T

def compute_flux(rho, u, p, E, c, A):
    """Compute the flux vector F."""
    return np.array([
        rho * u * A,            # Mass flux
        (rho * u**2 + p) * A,   # Momentum flux
        (rho * E * u + p * u) * A,  # Energy flux
        rho * c * u * A         # Reaction progress flux
    ])

def rusanov_flux(U_L, U_R, A_L, A_R):
    """Compute numerical flux at cell interfaces using Rusanov scheme."""
    rho_L, u_L, p_L, c_L, T_L = get_primitives(U_L, A_L)
    rho_R, u_R, p_R, c_R, T_R = get_primitives(U_R, A_R)
    A_interface = 0.5 * (A_L + A_R)
    E_L = U_L[2] / U_L[0]  # E = (rho*E*A) / (rho*A)
    E_R = U_R[2] / U_R[0]
    F_L = compute_flux(rho_L, u_L, p_L, E_L, c_L, A_interface)
    F_R = compute_flux(rho_R, u_R, p_R, E_R, c_R, A_interface)
    a_L = np.sqrt(gamma * p_L / rho_L)  # Speed of sound
    a_R = np.sqrt(gamma * p_R / rho_R)
    lambda_max = max(abs(u_L) + a_L, abs(u_R) + a_R)
    F = 0.5 * (F_L + F_R) - 0.5 * lambda_max * (U_R - U_L)
    return F

### Time-Stepping Loop
t = 0.0
t_end = 0.01  # Simulation end time (s)
CFL = 0.5     # Courant number for stability
while t < t_end:
    # Compute primitive variables
    rho, u, p, c, T = get_primitives(U, A)
    
    # Compute time step based on CFL condition
    a = np.sqrt(gamma * p / rho)
    lambda_max = np.max(np.abs(u) + a)
    dt = CFL * dx / lambda_max
    if t + dt > t_end:
        dt = t_end - t
    
    # Compute fluxes at cell interfaces
    F = np.zeros((4, N + 1))
    for i in range(1, N):
        U_L = U[:, i-1]
        U_R = U[:, i]
        A_L = A[i-1]
        A_R = A[i]
        F[:, i] = rusanov_flux(U_L, U_R, A_L, A_R)
    # Simple boundary conditions: extrapolate fluxes
    F[:, 0] = F[:, 1]
    F[:, N] = F[:, N-1]
    
    # Compute source terms
    S = np.zeros((4, N))
    # Momentum source due to area change
    for i in range(1, N-1):
        A_ip = 0.5 * (A[i] + A[i+1])
        A_im = 0.5 * (A[i] + A[i-1])
        S[1, i] = p[i] * (A_ip - A_im) / dx
    # Reaction source term
    dot_omega = A_reaction * (1 - c) * np.exp(-Ea / (R_gas * T)) * react_zone
    S[3, :] = rho * A * dot_omega
    
    # Update conserved variables
    for i in range(N):
        U[:, i] = U[:, i] - dt / dx * (F[:, i+1] - F[:, i]) + dt * S[:, i]
    
    t += dt

### Visualization (Optional)
rho, u, p, c, T = get_primitives(U, A)
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(x, p / 1e5)
plt.xlabel('x (m)')
plt.ylabel('Pressure (bar)')
plt.title('Pressure')
plt.subplot(2, 2, 2)
plt.plot(x, u)
plt.xlabel('x (m)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity')
plt.subplot(2, 2, 3)
plt.plot(x, T)
plt.xlabel('x (m)')
plt.ylabel('Temperature (K)')
plt.title('Temperature')
plt.subplot(2, 2, 4)
plt.plot(x, c)
plt.xlabel('x (m)')
plt.ylabel('Reaction Progress')
plt.title('Reaction Progress')
plt.tight_layout()
plt.show()
Explanation of the Code
1. Domain and Grid
Length: 10 meters, discretized into 1000 cells (
\Delta x = 0.01 \, \text{m}
).
Obstacles: Cross-sectional area 
A(x) = 0.5
 for the first 10 cells of every 100 cells (every 1 m, width 0.1 m), 
A(x) = 1.0
 elsewhere.
2. Physical Parameters
Initial Conditions: High pressure (10 atm) for 
x < 5 \, \text{m}
, low pressure (1 atm) for 
x > 5 \, \text{m}
, both at 300 K, zero velocity, unburnt (
c = 0
).
Gas Properties: Ideal gas with 
\gamma = 1.4
, R = 287 \, \text{J/kg·K}, c_v = 717.5 \, \text{J/kg·K}.
Combustion: Heat release 
q_0 = 2.8 \times 10^6 \, \text{J/kg}
, reaction rate 
\dot{\omega} = A (1 - c) \exp(-E_a / (R T))
, with 
A = 10^{10} \, \text{s}^{-1}
 (needs tuning), 
E_a = 1.5 \times 10^5 \, \text{J/mol}
.
3. Governing Equations
The conserved variables are:
U = [\rho A, m = \rho u A, \rho E A, \rho c A]
Flux: 
F = [\rho u A, (\rho u^2 + p) A, (\rho E + p) u A, \rho c u A]
Source: 
S = [0, p \frac{dA}{dx}, 0, \rho A \dot{\omega}]
Where:
E = e + \frac{1}{2} u^2 + (1 - c) q_0
, 
e = c_v T
, 
p = \rho R T
.
4. Numerical Method
Finite Volume Update: 
U_i^{n+1} = U_i^n - \frac{\Delta t}{\Delta x} (F_{i+1/2} - F_{i-1/2}) + \Delta t S_i
.
Rusanov Flux: 
F_{i+1/2} = \frac{1}{2} (F_L + F_R) - \frac{1}{2} \lambda_{\text{max}} (U_R - U_L)
, with 
\lambda_{\text{max}} = \max(|u| + a)
, 
a = \sqrt{\gamma p / \rho}
.
Time Step: 
\Delta t = \text{CFL} \cdot \Delta x / \lambda_{\text{max}}
, CFL = 0.5.
5. Boundary Conditions
Simple extrapolation: fluxes at boundaries mirror adjacent interior fluxes.
6. Visualization
Plots pressure, velocity, temperature, and reaction progress at the final time (0.01 s).
Notes and Tuning
Reaction Rate: 
A_{\text{reaction}} = 10^{10} \, \text{s}^{-1}
 is a placeholder. Adjust it to match the laminar flame speed of methane-air (~0.4 m/s) or to trigger DDT appropriately.
Stability: The explicit method may struggle with stiff reaction terms; a smaller 
\Delta t
 or implicit solver might be needed for robustness.
DDT Detection: Monitor pressure peaks or flame front acceleration to confirm transition to detonation.
Obstacles: The periodic blockages (blockage ratio 0.5) enhance flame acceleration; adjust spacing or size for different effects.
This code provides a foundational simulation that can be refined based on specific experimental data or desired outcomes. Run it and visualize the results to observe the shock propagation and potential DDT in the methane-air mixture.
