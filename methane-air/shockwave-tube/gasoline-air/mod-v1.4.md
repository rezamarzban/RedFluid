```python
# Constants
M_reactants = 0.0304  # kg/mol, molar mass of gasoline-air mixture
q0 = 2.72e6          # J/kg, heat release for gasoline-air mixture
rho_init = 1.26      # kg/m³, initial density at 1 atm, 300 K
cv = 717.0           # J/(kg·K), kept as air-like (optional: adjust to 720.0)
R_universal = 8.314  # J/(mol·K)

# Modified compute_pressure function
def compute_pressure(rho, T, a, b, M_reactants):
    v_mol = M_reactants / rho  # molar volume, m³/mol
    v_safe = np.maximum(v_mol, b * 1.01)
    term1 = R_universal * T / (v_safe - b)
    term2 = a / (v_safe * (v_safe + b) + b * (v_safe - b))
    P = np.maximum(term1 - term2, 1e-6)
    return P

# In the main simulation loop, update the pressure calculation:
# Replace: p = compute_pressure(rho, T, a, b)
# With:    p = compute_pressure(rho, T, a, b, M_reactants)
```
