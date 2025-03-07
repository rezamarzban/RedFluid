### Changelog for `ddt-v1.0.py`
- Implemented a 1D detonation tube simulation using the Euler equations with a reacting source term.  
- Defined physical parameters, including gas properties and tube geometry with obstacles.  
- Initialized conserved variables (`rho*A`, `rho*u*A`, `rho*E*A`, `rho*c*A`) based on initial conditions.  
- Implemented Rusanov flux scheme for numerical flux computation.  
- Added source terms for momentum (due to area change) and reaction kinetics.  
- Used a CFL-based adaptive time-stepping method.  
- Included optional visualization for pressure, velocity, temperature, and reaction progress.


  ### **Version History**  

#### **v1.0 → v1.1**  
- **Added** Peng-Robinson EOS for more accurate pressure calculations.  
- **Implemented** adaptive time-stepping based on the CFL condition.  
- **Optimized** reaction rate calculations using Numba for speed.  
- **Improved** numerical stability with `np.clip()` and `np.maximum()`.  
- **Fixed** floating-point instability in the Arrhenius reaction rate.  
- **Enhanced** visualization with density, velocity, pressure, temperature, and reaction progress plots.


#### **v1.1 → v1.2**  
- **Added** spatially varying cross-sectional area with obstacles (blockage effects).  
- **Implemented** adaptive time-stepping based on local wave speeds.  
- **Enhanced** reaction progress tracking with better ignition modeling.  
- **Refined** Peng-Robinson EOS with temperature-dependent **a** parameter.  
- **Optimized** Rusanov flux computation for numerical stability.  
- **Improved** boundary conditions to maintain physical consistency.  
- **Fixed** minor numerical instabilities in reaction rate calculations.



#### **v1.2 → v1.3**  
1. **Peng–Robinson EOS Implementation**  
   - Added functions `compute_a(T)` and `compute_b()` to calculate Peng–Robinson equation of state (EOS) parameters.  
   - Updated the pressure calculation to use the Peng–Robinson EOS in `compute_pressure()`.  

2. **Geometric Effects (Variable Cross-Section)**  
   - Introduced a blockage model where obstacles reduce the cross-sectional area (`A`).  
   - Added `dAdx` to account for the geometric source term in momentum conservation.  

3. **Improved Chemistry Model**  
   - Expanded reaction kinetics to include a two-step model (induction + exothermic reaction).  
   - Modified `reaction_rates()` to include separate rates for induction and heat release.  

4. **Rusanov Flux Implementation for Convective Terms**  
   - Replaced simple upwind fluxes with the **Rusanov (local Lax-Friedrichs) scheme** for better numerical stability.  
   - Updated `rusanov_flux()` to account for wave speeds in flux computation.  

5. **Adaptive Time Stepping**  
   - Introduced `adaptive_dt()` to dynamically adjust time step size based on the CFL condition and local wave speeds.  

6. **Improved Diffusive Flux Handling**  
   - Added `compute_diffusive_flux()` to compute viscosity, thermal conduction, and species diffusion effects.  

7. **Boundary Condition Updates**  
   - Implemented **reflective boundary conditions** to handle shock reflections correctly.  

8. **Postprocessing Enhancements**  
   - Improved plotting with better visualization of density, velocity, pressure, induction progress, reaction progress, and temperature.  

These improvements enhance the physical accuracy, numerical stability, and computational efficiency of the detonation simulation.


#### **v1.3 → v1.4**  
1. **Improved Grid Resolution**  
   - Increased the number of grid points to **4000** for higher spatial accuracy.  

2. **Modified Chemical Kinetics for Stability**  
   - Reduced reaction rate parameters (`A_rate1`, `A_rate2`) to prevent numerical instability.  
   - Lowered ignition hot spot temperature to **1000 K** to smooth out initiation.  

3. **Refined Obstacle-Induced Blockage Model**  
   - Implemented alternating blockage ratios **(50% and 70%)** for obstacles.  
   - Adjusted `dAdx` calculation to better account for sudden area changes.  

4. **Peng–Robinson EOS Enhancements**  
   - Improved the **pressure calculation** with a safer volume approximation **(v_safe)** to prevent division by zero.  
   - Precomputed `b` for efficiency.  

5. **Adaptive Time Stepping for Stability**  
   - Implemented `adaptive_dt()` to dynamically adjust **time step size** based on the local wave speed.  

6. **Local Lax-Friedrichs (LLF) Convective Flux**  
   - Replaced simple upwind fluxes with **LLF flux scheme**, improving numerical robustness.  

7. **Enhanced Diffusive Flux Handling**  
   - Added viscosity, thermal conduction, and species diffusion effects.  

8. **Improved Boundary Conditions**  
   - Applied **reflective boundary conditions** to properly handle wave reflections at domain boundaries.  

9. **Detailed Postprocessing and Visualization**  
   - Plots now include **density, velocity, pressure, induction progress, reaction progress, and temperature**.  

These updates significantly improve **accuracy, numerical stability, and physical realism** in detonation wave simulations.
