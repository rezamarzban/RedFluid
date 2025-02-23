import numpy as np
from numba import jit
import matplotlib.pyplot as plt

# Physical constants
R_universal = 8.314  # Universal gas constant (J/mol·K)
M_reactants = 0.029  # Molar mass of reactants (kg/mol, approx. for air)
cv = 717.0  # Specific heat at constant volume (J/kg·K)
q0 = 2.5e6  # Chemical energy release (J/kg)
Ea = 1e5  # Activation energy (J/mol)
A_rate = 1e10  # Pre-exponential factor for reaction rate (1/s)
gamma = 1.4  # Ratio of specific heats

# Critical properties for air (approximated as N2-dominated)
T_c = 126.2  # Critical temperature (K)
p_c = 3.39e6  # Critical pressure (Pa)
omega = 0.037  # Acentric factor

# Simulation parameters
L = 10.0  # Domain length (m)
N = 1000  # Number of grid points
dx = L / (N - 1)  # Grid spacing
t_max = 1e-3  # Maximum simulation time (s)
CFL = 0.5  # CFL number for stability

# Compute Peng-Robinson parameters a and b
def compute_a(T):
    """Temperature-dependent Peng-Robinson parameter a."""
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1 + kappa * (1 - np.sqrt(T / T_c)))**2
    return 0.45724 * (R_universal * T_c)**2 / p_c * alpha

def compute_b():
    """Constant Peng-Robinson parameter b."""
    return 0.07780 * R_universal * T_c / p_c

# Precompute b (constant for pure component)
b = compute_b()

# Initial conditions
x = np.linspace(0, L, N)
A = np.ones(N)  # Cross-sectional area (m^2), constant for simplicity
U = np.zeros((4, N))  # Conservative variables: [rho*A, rho*u*A, rho*E*A, rho*c*A]
U[0] = 1.2 * A  # Initial density (kg/m^3)
U[1] = 0.0  # Initial momentum (at rest)
U[2] = U[0] * (cv * 300 + q0)  # Total energy (J/m^3), T=300K + chemical energy
U[3] = 0.0  # Reaction progress (0 = unburnt)
U[3, :N//10] = 1.0  # Ignite left 10% of the domain

@jit(nopython=True)
def get_primitives(U, A, q0, cv):
    """Convert conservative to primitive variables."""
    rho = np.maximum(U[0] / A, 1e-10)  # Density (kg/m^3)
    u = np.clip(U[1] / U[0], -1e4, 1e4)  # Velocity (m/s)
    E = U[2] / U[0]  # Specific total energy (J/kg)
    c = np.clip(U[3] / U[0], 0, 1)  # Reaction progress
    
    e_kinetic = 0.5 * u * u
    e_chemical = (1 - c) * q0
    e_thermal = E - e_kinetic - e_chemical
    T = np.maximum(e_thermal / cv, 100.0)  # Temperature (K)
    
    return rho, u, c, T

@jit(nopython=True)
def compute_pressure(rho, T, a, b):
    """Peng-Robinson EOS for pressure."""
    v = 1 / rho  # Specific volume
    v_safe = np.maximum(v, b * 1.01)  # Avoid division by zero
    term1 = R_universal * T / (v_safe - b)
    term2 = a / (v_safe * (v_safe + b) + b * (v_safe - b))
    P = term1 - term2
    return np.clip(P, 1e-6, 1e10)  # Prevent unphysical pressures

@jit(nopython=True)
def reaction_rate(rho, c, T, Ea, A_rate, R_universal):
    """Arrhenius reaction rate."""
    exponent = -Ea / (R_universal * T)
    safe_exp = np.minimum(exponent, 100.0)  # Prevent overflow
    return np.where(c < 1.0, rho * A_rate * np.exp(safe_exp) * (1 - c), 0.0)

@jit(nopython=True)
def rusanov_flux(U, A, p, rho, u, c, T, cv, q0, gamma):
    """Compute fluxes using Rusanov (local Lax-Friedrichs) scheme."""
    N = U.shape[1]
    F = np.zeros((4, N-1))  # Fluxes at N-1 interfaces
    
    for i in range(N-1):
        # Left state (cell i)
        rho_L = rho[i]
        u_L = u[i]
        p_L = p[i]
        c_L = c[i]
        T_L = T[i]
        A_L = A[i]
        
        # Right state (cell i+1)
        rho_R = rho[i+1]
        u_R = u[i+1]
        p_R = p[i+1]
        c_R = c[i+1]
        T_R = T[i+1]
        A_R = A[i+1]
        
        # Physical flux from left state
        F_L = np.array([
            rho_L * u_L * A_L,                                    # Mass flux
            (rho_L * u_L * u_L + p_L) * A_L,                     # Momentum flux
            (rho_L * (cv * T_L + (1 - c_L) * q0 + 0.5 * u_L * u_L) + p_L) * u_L * A_L,  # Energy flux
            rho_L * u_L * c_L * A_L                              # Species flux
        ])
        
        # Physical flux from right state
        F_R = np.array([
            rho_R * u_R * A_R,
            (rho_R * u_R * u_R + p_R) * A_R,
            (rho_R * (cv * T_R + (1 - c_R) * q0 + 0.5 * u_R * u_R) + p_R) * u_R * A_R,
            rho_R * u_R * c_R * A_R
        ])
        
        # Maximum wave speed for dissipation
        c_s_L = np.sqrt(gamma * p_L / rho_L)  # Sound speed left
        c_s_R = np.sqrt(gamma * p_R / rho_R)  # Sound speed right
        s_max = max(np.abs(u_L) + c_s_L, np.abs(u_R) + c_s_R)
        
        # Rusanov flux: average of fluxes + dissipation
        F[:, i] = 0.5 * (F_L + F_R) - 0.5 * s_max * (U[:, i+1] - U[:, i])
    
    return F

def adaptive_dt(U, A, dx, CFL, gamma):
    """Compute adaptive time step based on CFL condition."""
    rho, u, _, T = get_primitives(U, A, q0, cv)
    c_s = np.sqrt(gamma * (R_universal * T) / M_reactants)  # Speed of sound
    max_speed = np.max(np.abs(u) + c_s)
    return CFL * dx / max_speed if max_speed > 0 else 1e-6

# Main simulation loop
t = 0.0
dt = 1e-6  # Initial time step
while t < t_max:
    # Get primitive variables
    rho, u, c, T = get_primitives(U, A, q0, cv)
    
    # Compute a for each cell based on T
    a = compute_a(T)
    
    # Compute pressure using Peng-Robinson EOS
    p = compute_pressure(rho, T, a, b)
    
    # Compute fluxes at interfaces
    F = rusanov_flux(U, A, p, rho, u, c, T, cv, q0, gamma)
    
    # Compute flux differences for interior cells
    dFdx = np.zeros((4, N))
    dFdx[:, 1:-1] = (F[:, 1:] - F[:, :-1]) / dx
    
    # Compute source term (reaction)
    S = np.zeros_like(U)
    S[3] = reaction_rate(rho, c, T, Ea, A_rate, R_universal) * A
    
    # Update conservative variables
    U -= dt * dFdx
    U += dt * S
    
    # Apply reflective boundary conditions
    U[:, 0] = U[:, 1]
    U[:, -1] = U[:, -2]
    
    # Update time step
    dt = adaptive_dt(U, A, dx, CFL, gamma)
    t += dt
    
    # Print progress occasionally
    if t % 1e-4 < dt:
        print(f"t = {t:.6f}, dt = {dt:.6e}")

print("Simulation complete!")

# Extract final state for plotting
rho_final, u_final, c_final, T_final = get_primitives(U, A, q0, cv)
a_final = compute_a(T_final)
p_final = compute_pressure(rho_final, T_final, a_final, b)

# Plotting results
plt.figure(figsize=(12, 10))

plt.subplot(2, 3, 1)
plt.plot(x, rho_final, label='Density (kg/m³)')
plt.xlabel('Position (m)')
plt.ylabel('Density')
plt.title('Density')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(x, u_final, label='Velocity (m/s)')
plt.xlabel('Position (m)')
plt.ylabel('Velocity')
plt.title('Velocity')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(x, p_final / 1e5, label='Pressure (bar)')  # Pa to bar
plt.xlabel('Position (m)')
plt.ylabel('Pressure')
plt.title('Pressure')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(x, c_final, label='Reaction Progress')
plt.xlabel('Position (m)')
plt.ylabel('Reaction Progress')
plt.title('Reaction Progress')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(x, T_final, label='Temperature (K)')
plt.xlabel('Position (m)')
plt.ylabel('Temperature')
plt.title('Temperature')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
