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
gamma = 1.4               

# Two-step chemical kinetics parameters
# Induction step (no significant heat release)
Ea1 = 5e4                    # Activation energy for induction [J/mol]
A_rate1 = 1e8                # Pre-exponential factor for induction [1/s]
# Main exothermic reaction (with heat release q0)
Ea2 = 8e4                    # Activation energy for main reaction [J/mol]
A_rate2 = 1e10               # Pre-exponential factor for main reaction [1/s]

# =============================================================================
# Transport Properties (viscosity, conduction, species diffusion)
# =============================================================================
mu = 1.8e-5                # Dynamic viscosity [Pa*s]
Pr = 0.7                   # Prandtl number (dimensionless)
kappa = mu * cv / Pr       # Thermal conductivity [W/(m*K)]
D = 2e-5                   # Species diffusivity [m²/s]

# =============================================================================
# Peng-Robinson EOS Critical Parameters (for an air-like mixture)
# =============================================================================
T_c = 126.2                # Critical temperature [K]
p_c = 3.39e6               # Critical pressure [Pa]
omega = 0.037              # Acentric factor

# =============================================================================
# Simulation Parameters and Grid
# =============================================================================
L = 10.0                   # Domain length [m]
N = 1000                   # Number of grid points
dx = L / (N - 1)           # Grid spacing [m]
t_max = 1e-3               # Maximum simulation time [s]
CFL = 0.5                  # CFL number for stability
blockage_ratio = 0.5       # 50% blockage in obstacles

x = np.linspace(0, L, N)

# Define variable cross-sectional area with obstacles
A = np.ones(N)
obstacle_positions = np.arange(1, 10, 1)  # Obstacles every 1 m
obstacle_width = 0.1                     # Obstacle width [m]
for pos in obstacle_positions:
    mask = (x >= pos - obstacle_width/2) & (x <= pos + obstacle_width/2)
    A[mask] = 1 - blockage_ratio

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
# The conservative state vector U has 5 components:
# U[0] = mass (rho*A)
# U[1] = momentum (rho*u*A)
# U[2] = total energy (rho*E*A)
# U[3] = induction progress (c1, 0: uninduced, 1: fully induced)
# U[4] = main reaction progress (c2, 0: unburned, 1: fully burned)
U = np.zeros((5, N))
rho_init = 1.2
U[0] = rho_init * A            # Mass
U[1] = 0.0                     # Momentum (initially at rest)
U[2] = rho_init * A * (cv * 300) # Energy (T = 300 K initially, no reaction heat)
U[3] = 0.0                     # Induction variable c1
U[4] = 0.0                     # Main reaction variable c2

# Trigger ignition with a hot spot in the left 10% of the domain:
hot_spot = slice(0, N // 10)
U[2, hot_spot] = rho_init * A[hot_spot] * (cv * 600)  # locally higher temperature

# =============================================================================
# Numba-Accelerated Functions
# =============================================================================
@jit(nopython=True)
def get_primitives(U, A, cv, q0):
    """
    Convert conservative variables to primitive variables.
    Returns density (rho), velocity (u), induction progress (c1),
    main reaction progress (c2), and temperature (T).
    """
    rho = np.maximum(U[0] / A, 1e-10)
    u = np.clip(U[1] / U[0], -1e4, 1e4)
    E = U[2] / U[0]
    c1 = np.clip(U[3] / U[0], 0.0, 1.0)
    c2 = np.clip(U[4] / U[0], 0.0, 1.0)
    
    e_kin = 0.5 * u * u
    # Only the main reaction contributes to heat release
    e_chem = c2 * q0  
    e_th = E - e_kin - e_chem
    T = np.maximum(e_th / cv, 100.0)
    return rho, u, c1, c2, T

@jit(nopython=True)
def compute_pressure(rho, T, a, b):
    """
    Compute pressure using the Peng–Robinson EOS.
    """
    v = 1.0 / rho
    v_safe = np.maximum(v, b * 1.01)
    term1 = R_universal * T / (v_safe - b)
    term2 = a / (v_safe * (v_safe + b) + b * (v_safe - b))
    P = term1 - term2
    return np.clip(P, 1e-6, 1e10)

@jit(nopython=True)
def reaction_rates(rho, c1, c2, T, Ea1, A_rate1, Ea2, A_rate2, R_universal):
    """
    Compute two-step reaction rates.
      - w1: induction rate (increases c1, no heat release)
      - w2: main reaction rate (increases c2 and releases heat)
    """
    w1 = A_rate1 * rho * np.exp(-Ea1 / (R_universal * T)) * (1 - c1)
    w2 = A_rate2 * rho * np.exp(-Ea2 / (R_universal * T)) * c1 * (1 - c2)
    return w1, w2

@jit(nopython=True)
def rusanov_flux(U, A, p, rho, u, c1, c2, T, cv, q0, gamma):
    """
    Compute numerical convective fluxes at cell interfaces using the Rusanov scheme.
    """
    N = U.shape[1]
    F = np.zeros((5, N - 1))
    for i in range(N - 1):
        # Left state (cell i)
        rho_L = rho[i]
        u_L = u[i]
        p_L = p[i]
        c1_L = c1[i]
        c2_L = c2[i]
        T_L = T[i]
        A_L = A[i]
        E_L = cv * T_L + c2_L * q0 + 0.5 * u_L * u_L

        # Right state (cell i+1)
        rho_R = rho[i + 1]
        u_R = u[i + 1]
        p_R = p[i + 1]
        c1_R = c1[i + 1]
        c2_R = c2[i + 1]
        T_R = T[i + 1]
        A_R = A[i + 1]
        E_R = cv * T_R + c2_R * q0 + 0.5 * u_R * u_R

        # Flux vectors for left and right states
        F_L = np.zeros(5)
        F_L[0] = rho_L * u_L * A_L
        F_L[1] = (rho_L * u_L * u_L + p_L) * A_L
        F_L[2] = (rho_L * E_L + p_L) * u_L * A_L
        F_L[3] = rho_L * u_L * c1_L * A_L
        F_L[4] = rho_L * u_L * c2_L * A_L

        F_R = np.zeros(5)
        F_R[0] = rho_R * u_R * A_R
        F_R[1] = (rho_R * u_R * u_R + p_R) * A_R
        F_R[2] = (rho_R * E_R + p_R) * u_R * A_R
        F_R[3] = rho_R * u_R * c1_R * A_R
        F_R[4] = rho_R * u_R * c2_R * A_R

        # Estimate maximum wave speed at the interface
        c_s_L = np.sqrt(gamma * p_L / rho_L)
        c_s_R = np.sqrt(gamma * p_R / rho_R)
        s_max = max(np.abs(u_L) + c_s_L, np.abs(u_R) + c_s_R)

        # Rusanov flux: average of fluxes plus dissipation
        for k in range(5):
            F[k, i] = 0.5 * (F_L[k] + F_R[k]) - 0.5 * s_max * (U[k, i + 1] - U[k, i])
    return F

def adaptive_dt(U, A, dx, CFL, gamma):
    """Compute adaptive time step based on local maximum speed."""
    rho, u, c1, c2, T = get_primitives(U, A, cv, q0)
    c_s = np.sqrt(gamma * (R_universal * T) / M_reactants)
    max_speed = np.max(np.abs(u) + c_s)
    return CFL * dx / max_speed if max_speed > 0 else 1e-6

# =============================================================================
# Diffusive Fluxes (Viscosity, Thermal Conduction, Species Diffusion)
# =============================================================================
def compute_diffusive_flux(rho, u, T, c1, c2, A, dx, mu, kappa, D):
    """
    Compute diffusive flux derivatives for each conservation equation.
    Uses central differences (via np.gradient) for simplicity.
    """
    # Compute gradients
    du_dx = np.gradient(u, dx)
    dT_dx = np.gradient(T, dx)
    dc1_dx = np.gradient(c1, dx)
    dc2_dx = np.gradient(c2, dx)
    
    # Viscous stress for momentum: tau = mu * du/dx
    tau = mu * du_dx
    dtauA_dx = np.gradient(tau * A, dx)
    
    # Conduction flux for energy: q = -kappa * dT/dx
    q_flux = -kappa * dT_dx
    dqA_dx = np.gradient(q_flux * A, dx)
    
    # Species diffusion fluxes: J = -rho * D * d(c)/dx
    J_c1 = -rho * D * dc1_dx
    J_c2 = -rho * D * dc2_dx
    dJc1A_dx = np.gradient(J_c1 * A, dx)
    dJc2A_dx = np.gradient(J_c2 * A, dx)
    
    # Assemble diffusive contributions for each conservation equation:
    diff_mass     = np.zeros_like(rho)             # Mass diffusion is neglected
    diff_momentum = dtauA_dx                       # Viscous term for momentum
    # Energy diffusion: conduction + work done by viscosity (u*tau)
    diff_energy   = dqA_dx + np.gradient(u * tau * A, dx)
    diff_c1       = dJc1A_dx                      # Species diffusion for induction variable
    diff_c2       = dJc2A_dx                      # Species diffusion for main reaction variable
    
    return diff_mass, diff_momentum, diff_energy, diff_c1, diff_c2

# =============================================================================
# Main Simulation Loop (Operator-Split: Convection+Diffusion, then Reaction)
# =============================================================================
t = 0.0
dt = 1e-6
while t < t_max:
    # Get primitive variables
    rho, u, c1, c2, T = get_primitives(U, A, cv, q0)
    
    # Compute EOS parameter a and pressure field
    a = compute_a(T)
    p = compute_pressure(rho, T, a, b)
    
    # -------------------------
    # Convection and Diffusion
    # -------------------------
    F = rusanov_flux(U, A, p, rho, u, c1, c2, T, cv, q0, gamma)
    dFdx = np.zeros(U.shape)
    dFdx[:, 1:-1] = (F[:, 1:] - F[:, :-1]) / dx

    diff_mass, diff_momentum, diff_energy, diff_c1, diff_c2 = compute_diffusive_flux(
        rho, u, T, c1, c2, A, dx, mu, kappa, D)
    dDiffdx = np.zeros(U.shape)
    dDiffdx[0] = diff_mass
    dDiffdx[1] = diff_momentum
    dDiffdx[2] = diff_energy
    dDiffdx[3] = diff_c1
    dDiffdx[4] = diff_c2

    # -------------------------
    # Reaction Source Terms
    # -------------------------
    w1, w2 = reaction_rates(rho, c1, c2, T, Ea1, A_rate1, Ea2, A_rate2, R_universal)
    S = np.zeros(U.shape)
    S[3] = w1 * A         # Induction progress source
    S[4] = w2 * A         # Main reaction progress source
    S[2] = q0 * w2 * A    # Energy source (heat release)
    
    # Geometric source term for momentum due to varying area
    S[1] = p * dAdx

    # Update conservative variables with convective, diffusive, and reaction effects
    U = U - dt * (dFdx + dDiffdx) + dt * S

    # Apply reflective boundary conditions
    U[:, 0] = U[:, 1]
    U[:, -1] = U[:, -2]
    
    # Update time step and time
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
