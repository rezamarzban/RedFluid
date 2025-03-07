```python
# Constants
M_reactants = 0.0186  # kg/mol, molar mass of H₂-air mixture (stoichiometric)
q0 = 3.38e6          # J/kg, heat release for H₂-air mixture
rho_init = 0.92      # kg/m³, initial density at 1 atm, 300 K
cv = 717.0           # J/(kg·K), kept as air-like (unchanged for simplicity)
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
