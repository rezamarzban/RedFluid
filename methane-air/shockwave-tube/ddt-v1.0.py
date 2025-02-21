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
