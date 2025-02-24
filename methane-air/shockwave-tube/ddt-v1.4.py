import numpy as np
from numba import jit
import matplotlib.pyplot as plt

# =============================================================================
# Physical Constants and Reaction Parameters
# =============================================================================
R_universal = 8.314          # Universal gas constant [J/(mol*K)]
M_reactants = 0.029          # Molar mass of reactants [kg/mol]
cv = 717.0                   # Specific heat at constant volume [J/(kg*K)]
q0 = 2.5e6                   # Heat release from main reaction [J/kg]
gamma = 1.4                  # Adiabatic index

# Two-step chemical kinetics parameters (reduced rates for stability)
Ea1 = 5e4                    # Activation energy for induction [J/mol]
A_rate1 = 1e7                # Reduced pre-exponential factor for induction [1/s]
Ea2 = 8e4                    # Activation energy for main reaction [J/mol]
A_rate2 = 1e9                # Reduced pre-exponential factor for main reaction [1/s]

# =============================================================================
# Transport Properties (viscosity, conduction, species diffusion)
# =============================================================================
mu = 1.8e-5                  # Dynamic viscosity [Pa*s]
Pr = 0.7                     # Prandtl number (dimensionless)
kappa = mu * cv / Pr         # Thermal conductivity [W/(m*K)]
D = 2e-5                     # Species diffusivity [m²/s]

# =============================================================================
# Peng-Robinson EOS Critical Parameters (for an air-like mixture)
# =============================================================================
T_c = 126.2                  # Critical temperature [K]
p_c = 3.39e6                 # Critical pressure [Pa]
omega = 0.037                # Acentric factor

# =============================================================================
# Simulation Parameters and Grid
# =============================================================================
L = 10.0                     # Domain length [m]
N = 4000                     # Increased number of grid points for better resolution
dx = L / (N - 1)             # Grid spacing [m]
t_max = 0.01                 # Maximum simulation time [s]
CFL = 0.1                    # Reduced CFL number for stability
obstacle_width = 0.1         # Obstacle width [m]

x = np.linspace(0, L, N)

# Define variable cross-sectional area with obstacles
obstacle_positions = np.arange(0.5, L, 0.5)  # Obstacles every 0.5 m
blockage_ratios = [0.5 if i % 2 == 0 else 0.7 for i in range(len(obstacle_positions))]
A = np.ones(N)
for pos, br in zip(obstacle_positions, blockage_ratios):
    mask = (x >= pos - obstacle_width/2) & (x <= pos + obstacle_width/2)
    A[mask] = 1 - br

# Compute gradient of area for geometric source terms
dAdx = np.gradient(A, dx)

# =============================================================================
# Peng-Robinson EOS Functions
# =============================================================================
def compute_a(T):
    """Compute the temperature-dependent Peng–Robinson parameter a."""
    kappa_PR = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1 + kappa_PR * (1 - np.sqrt(T / T_c)))**2
    return 0.45724 * (R_universal * T_c)**2 / p_c * alpha

def compute_b():
    """Compute the constant Peng–Robinson parameter b."""
    return 0.07780 * R_universal * T_c / p_c

b = compute_b()

# =============================================================================
# Initial Conditions
# =============================================================================
# Conservative state vector U: [mass, momentum, total energy, induction progress, reaction progress]
U = np.zeros((5, N))
rho_init = 1.2               # Initial density [kg/m³]
U[0] = rho_init * A          # Mass (rho*A)
U[1] = 0.0                   # Momentum (initially at rest)
U[2] = rho_init * A * (cv * 300)  # Energy (T = 300 K, no reaction heat)
U[3] = 0.0                   # Induction variable c1
U[4] = 0.0                   # Main reaction variable c2

# Trigger ignition with a larger, less intense hot spot
hot_spot = (x < 0.5)         # Hot spot size increased to 0.5 m
T_hot = 1000.0               # Hot spot temperature reduced to 1000 K
E_hot = cv * T_hot           # Energy in hot spot
U[2, hot_spot] = rho_init * A[hot_spot] * E_hot

# =============================================================================
# Numba-Accelerated Functions
# =============================================================================
@jit(nopython=True)
def get_primitives(U, A, cv, q0):
    """Convert conservative variables to primitive variables with physical constraints."""
    rho = np.maximum(U[0] / A, 0.01)  # Minimum density
    u = np.clip(U[1] / U[0], -1e4, 1e4)  # Limit velocity
    E = U[2] / U[0]
    c1 = np.clip(U[3] / U[0], 0.0, 1.0)
    c2 = np.clip(U[4] / U[0], 0.0, 1.0)
    e_kin = 0.5 * u * u
    e_chem = c2 * q0
    e_th = E - e_kin - e_chem
    T = np.maximum(e_th / cv, 300.0)  # Minimum temperature
    return rho, u, c1, c2, T

@jit(nopython=True)
def compute_pressure(rho, T, a, b):
    """Compute pressure using the Peng–Robinson EOS with minimum pressure."""
    v = 1.0 / rho
    v_safe = np.maximum(v, b * 1.01)
    term1 = R_universal * T / (v_safe - b)
    term2 = a / (v_safe * (v_safe + b) + b * (v_safe - b))
    P = np.maximum(term1 - term2, 1e-6)  # Minimum pressure
    return P

@jit(nopython=True)
def reaction_rates(rho, c1, c2, T, Ea1, A_rate1, Ea2, A_rate2, R_universal):
    """Compute two-step reaction rates."""
    w1 = A_rate1 * rho * np.exp(-Ea1 / (R_universal * T)) * (1 - c1)
    w2 = A_rate2 * rho * np.exp(-Ea2 / (R_universal * T)) * c1 * (1 - c2)
    return w1, w2

@jit(nopython=True)
def llf_flux(U, A, p, rho, u, c1, c2, T, cv, q0, gamma):
    """Compute numerical convective fluxes using the Local Lax-Friedrichs scheme."""
    N = U.shape[1]
    F = np.zeros((5, N - 1))
    for i in range(N - 1):
        # Left state
        rho_L, u_L, p_L = rho[i], u[i], p[i]
        c1_L, c2_L, T_L = c1[i], c2[i], T[i]
        A_L = A[i]
        E_L = cv * T_L + c2_L * q0 + 0.5 * u_L * u_L

        # Right state
        rho_R, u_R, p_R = rho[i + 1], u[i + 1], p[i + 1]
        c1_R, c2_R, T_R = c1[i + 1], c2[i + 1], T[i + 1]
        A_R = A[i + 1]
        E_R = cv * T_R + c2_R * q0 + 0.5 * u_R * u_R

        # Flux vectors
        F_L = np.array([
            rho_L * u_L * A_L,
            (rho_L * u_L * u_L + p_L) * A_L,
            (rho_L * E_L + p_L) * u_L * A_L,
            rho_L * u_L * c1_L * A_L,
            rho_L * u_L * c2_L * A_L
        ])
        F_R = np.array([
            rho_R * u_R * A_R,
            (rho_R * u_R * u_R + p_R) * A_R,
            (rho_R * E_R + p_R) * u_R * A_R,
            rho_R * u_R * c1_R * A_R,
            rho_R * u_R * c2_R * A_R
        ])

        # Maximum wave speed at the interface
        c_s_L = np.sqrt(gamma * p_L / rho_L)
        c_s_R = np.sqrt(gamma * p_R / rho_R)
        s_max = max(np.abs(u_L) + c_s_L, np.abs(u_R) + c_s_R)

        # LLF flux: average of fluxes plus dissipation
        F[:, i] = 0.5 * (F_L + F_R) - 0.5 * s_max * (U[:, i + 1] - U[:, i])
    return F

def adaptive_dt(U, A, dx, CFL, gamma):
    """Compute adaptive time step based on maximum speed."""
    rho, u, _, _, T = get_primitives(U, A, cv, q0)
    c_s = np.sqrt(gamma * (R_universal * T) / M_reactants)
    max_speed = np.max(np.abs(u) + c_s)
    return CFL * dx / max_speed if max_speed > 0 else 1e-6

def compute_diffusive_flux(rho, u, T, c1, c2, A, dx, mu, kappa, D):
    """Compute diffusive flux derivatives."""
    du_dx = np.gradient(u, dx)
    dT_dx = np.gradient(T, dx)
    dc1_dx = np.gradient(c1, dx)
    dc2_dx = np.gradient(c2, dx)
    
    tau = mu * du_dx
    dtauA_dx = np.gradient(tau * A, dx)
    q_flux = -kappa * dT_dx
    dqA_dx = np.gradient(q_flux * A, dx)
    J_c1 = -rho * D * dc1_dx
    J_c2 = -rho * D * dc2_dx
    dJc1A_dx = np.gradient(J_c1 * A, dx)
    dJc2A_dx = np.gradient(J_c2 * A, dx)
    
    return (np.zeros_like(rho), dtauA_dx, dqA_dx + np.gradient(u * tau * A, dx),
            dJc1A_dx, dJc2A_dx)

# =============================================================================
# Main Simulation Loop
# =============================================================================
t = 0.0
dt = 1e-6
while t < t_max:
    rho, u, c1, c2, T = get_primitives(U, A, cv, q0)
    a = compute_a(T)
    p = compute_pressure(rho, T, a, b)
    
    # Convection and diffusion using LLF flux
    F = llf_flux(U, A, p, rho, u, c1, c2, T, cv, q0, gamma)
    dFdx = np.zeros(U.shape)
    dFdx[:, 1:-1] = (F[:, 1:] - F[:, :-1]) / dx
    
    diff_mass, diff_momentum, diff_energy, diff_c1, diff_c2 = compute_diffusive_flux(
        rho, u, T, c1, c2, A, dx, mu, kappa, D)
    dDiffdx = np.array([diff_mass, diff_momentum, diff_energy, diff_c1, diff_c2])

    # Reaction source terms
    w1, w2 = reaction_rates(rho, c1, c2, T, Ea1, A_rate1, Ea2, A_rate2, R_universal)
    S = np.zeros(U.shape)
    S[3] = w1 * A
    S[4] = w2 * A
    S[2] = q0 * w2 * A
    S[1] = p * dAdx

    # Update state
    U = U - dt * (dFdx + dDiffdx) + dt * S
    
    # Reflective boundary conditions
    U[:, 0] = U[:, 1]
    U[1, 0] = -U[1, 1]  # Reflect velocity
    U[:, -1] = U[:, -2]
    U[1, -1] = -U[1, -2]  # Reflect velocity
    
    # Time step update
    dt = adaptive_dt(U, A, dx, CFL, gamma)
    t += dt
    if t % 1e-4 < dt:
        print(f"t = {t:.6f}, dt = {dt:.6e}")

print("Simulation complete!")

# =============================================================================
# Postprocessing and Plotting
# =============================================================================
rho_final, u_final, c1_final, c2_final, T_final = get_primitives(U, A, cv, q0)
a_final = compute_a(T_final)
p_final = compute_pressure(rho_final, T_final, a_final, b)

plt.figure(figsize=(12, 10))

plt.subplot(2, 3, 1)
plt.plot(x, rho_final, label='Density')
plt.xlabel('Position (m)')
plt.ylabel('Density (kg/m³)')
plt.title('Density')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(x, u_final, label='Velocity')
plt.xlabel('Position (m)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(x, p_final / 1e5, label='Pressure (bar)')
plt.xlabel('Position (m)')
plt.ylabel('Pressure (bar)')
plt.title('Pressure')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(x, c1_final, label='Induction Progress (c1)')
plt.xlabel('Position (m)')
plt.ylabel('c1')
plt.title('Induction Progress')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(x, c2_final, label='Main Reaction Progress (c2)')
plt.xlabel('Position (m)')
plt.ylabel('c2')
plt.title('Main Reaction Progress')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(x, T_final, label='Temperature (K)')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.title('Temperature')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
