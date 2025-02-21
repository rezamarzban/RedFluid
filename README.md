# RedFluid
RedFluid: PP, LEL &amp; UEL and DDT in Password Protected Python Script

### Key Points
- The equations for simulating deflagration to detonation transition (DDT) in a shockwave tube include mass, momentum, energy, and reaction progress conservation, along with an equation of state and reaction rate.
- These are typically one-dimensional (1D) compressible flow equations with combustion source terms, often using a progress variable for simplicity.
- Reaction rates are modeled using Arrhenius kinetics, and simulations require numerical methods to handle shocks and discontinuities.

### Introduction to DDT Simulation
Deflagration to detonation transition (DDT) is the process where a subsonic flame accelerates to become a supersonic detonation, studied in shockwave tubes for safety and propulsion research. These tubes create controlled conditions to observe this transition, making them ideal for studying explosive phenomena.

### Governing Equations
The core equations for modeling DDT in shockwave tubes are the 1D compressible flow equations, enhanced with combustion source terms. These include:

- **Mass Conservation**: Ensures the total mass of the gas remains constant as it moves and reacts.
- **Momentum Conservation**: Accounts for the gas's motion and pressure changes, often assuming inviscid flow for simplicity.
- **Energy Conservation**: Incorporates heat release from combustion, crucial for flame acceleration, with a source term for the reaction.
- **Reaction Progress Variable**: Tracks the combustion process, using a progress variable to simplify detailed chemistry.
- **Equation of State**: Relates pressure, density, and temperature, typically for an ideal gas.
- **Reaction Rate**: Models the temperature-dependent reaction rate, often using Arrhenius kinetics.

### Surprising Detail: Simplified Combustion Modeling
A notable aspect is the use of a progress variable instead of detailed species tracking, simplifying calculations while still capturing the essential DDT dynamics. This approach, using Arrhenius kinetics, makes simulations more manageable for complex transitions.

---

### Survey Note: Detailed Analysis of DDT Equations in Shockwave Tubes

This section provides a comprehensive examination of the mathematical models and equations used to describe deflagration to detonation transition (DDT) in shockwave tubes, drawing from extensive research and theoretical frameworks. It encompasses all relevant details from the initial exploration, including experimental insights, numerical simulations, and theoretical criteria, ensuring a thorough understanding for advanced readers.

#### Background and Context
DDT is a critical phenomenon in combustion science, where a subsonic deflagration transitions to a supersonic detonation, characterized by significant pressure and velocity increases. Shockwave tubes, used to replicate and study blast waves, are instrumental in investigating DDT under controlled conditions. These tubes typically feature a high-pressure section and a low-pressure section separated by a diaphragm, with the rupture initiating a shock wave that interacts with a combustible mixture, potentially leading to DDT.

The study of DDT is vital for applications in explosion prevention (e.g., fuel pipelines, mine tunnels) and propulsion systems (e.g., pulse detonation engines). Research has shown that DDT can occur through various mechanisms, including shock-flame interactions, turbulence, and hot spot formation, often requiring numerical simulations to capture the transient dynamics.

#### Governing Equations for DDT
The mathematical modeling of DDT in shockwave tubes primarily involves solving the compressible flow equations with combustion source terms. For a one-dimensional setup, which is common in shockwave tube simulations, the equations are as follows:

1. **Mass Conservation:**
   $$\frac{\partial \rho}{\partial t} + \frac{\partial (\rho u)}{\partial x} = 0$$
   This equation ensures the conservation of mass, accounting for density ($$\rho$$) changes due to flow and combustion.

2. **Momentum Conservation:**
   $$\frac{\partial (\rho u)}{\partial t} + \frac{\partial (\rho u^2 + p)}{\partial x} = 0$$
   This tracks the momentum, considering velocity ($$u$$) and pressure ($$p$$) variations, essential for capturing shock wave propagation. For inviscid flow, viscous terms are neglected, suitable for high Reynolds number flows typical in shock tubes. For more accuracy, viscous terms can be included as:
   $$\frac{\partial (\rho u)}{\partial t} + \frac{\partial (\rho u^2 + p)}{\partial x} = \frac{\partial}{\partial x} \left( \mu \frac{\partial u}{\partial x} \right)$$
   where $$\mu$$ is the viscosity, though often omitted for simplicity in DDT simulations.

3. **Energy Conservation:**
   $$\frac{\partial (\rho E)}{\partial t} + \frac{\partial (\rho u E + p u)}{\partial x} = \rho q \omega$$
   Here, $$E = e + \frac{u^2}{2}$$ is the specific total energy, with $$e$$ as the specific internal energy. The term $$\rho q \omega$$ represents the heat release from combustion, driving the transition to detonation. For an ideal gas, $$e = c_v T$$, where $$c_v$$ is the specific heat at constant volume, and $$T$$ is temperature. For viscous flow, additional terms for heat conduction can be included:
   $$\frac{\partial (\rho E)}{\partial t} + \frac{\partial (\rho u E + p u)}{\partial x} = \frac{\partial}{\partial x} \left( k \frac{\partial T}{\partial x} \right) + \rho q \omega$$
   where $$k$$ is the thermal conductivity, though often neglected for inviscid models.

4. **Reaction Progress Variable:**
   For detailed chemistry, species conservation equations can be used, but for simplicity, especially in DDT simulations, a progress variable $$c$$ (where $$c = 0$$ for unburnt and $$c = 1$$ for fully burnt) is often employed:
   $$\frac{\partial (\rho c)}{\partial t} + \frac{\partial (\rho u c)}{\partial x} = \rho \omega$$
   The heat release $$q \omega$$ is linked to the reaction rate $$\omega$$, where $$q$$ is the total heat release per unit mass.

5. **Equation of State:**
   Assuming an ideal gas, this relates pressure, density, and temperature:
   $$p = \rho R T$$
   where $$R$$ is the gas constant, and $$T$$ is temperature.

6. **Reaction Rate:**
   The reaction rate $$\omega$$ is commonly modeled using Arrhenius kinetics:
   $$\omega = A (1 - c) \exp\left(-\frac{E_a}{R T}\right)$$
   where $$A$$ is the pre-exponential factor, and $$E_a$$ is the activation energy, capturing the temperature-dependent reaction rate crucial for DDT. In some models, a dual-source term is used, with one part for deflagration and another for autoignition, as seen in simulations using ddtFoam ([Numerical Simulations of DDT Limits in Hydrogen-Air Mixtures in Obstacle Laden Channel](https://www.mdpi.com/1996-1073/14/1/24)).

#### Numerical Simulations and Specific Models
Numerical simulations are essential for solving these equations, given the transient and nonlinear nature of DDT. Several papers highlight the use of high-order numerical algorithms to solve the multidimensional, fully compressible, reactive Navier-Stokes equations, often coupled with single-step or multi-step chemical models ([Enhanced DDT mechanism from shock-flame interactions in thin channels](https://www.sciencedirect.com/science/article/abs/pii/S1540748920306271)). For instance, the material point method has been used with heat balance equations and equations of state like JWL and Virial, alongside the Lee-Tarver equation, to model DDT in explosives under complex conditions ([Numerical simulation of the deflagration to detonation transition behavior in explosives based on the material point method](https://www.sciencedirect.com/science/article/pii/S0010218021006635)).

In some cases, the Euler equations are used for comparison, neglecting viscosity, while others include boundary layer effects and turbulence, which are critical for DDT initiation. The choice between single-step and multi-step chemical schemes impacts the simulation outcomes, with multi-step models being more restrictive for hot spot ignition leading to detonation ([Influence of kinetics on DDT simulations](https://www.sciencedirect.com/science/article/abs/pii/S001021801830484X)).

#### Initial and Boundary Conditions
In a shockwave tube, initial conditions typically involve a high-pressure gas on one side and a low-pressure combustible mixture on the other, separated by a diaphragm. Upon rupture, a shock wave travels into the low-pressure section, and a rarefaction wave into the high-pressure section. This setup is designed to study how the shock wave interacts with the mixture, potentially igniting it and leading to DDT. The boundary conditions at the tube ends (e.g., closed or open) also influence the dynamics, with closed ends promoting pressure buildup and shock formation.

#### Criteria and Conditions for DDT
While the governing equations are fundamental, certain criteria help predict DDT. For example, the critical expansion ratio for flame acceleration is given by $$\sigma_{cr} = \sigma^* \cdot (1 - \alpha)^{-1}$$, where $$\sigma^* = e^x$$, $$x = E_a / RT_u$$, and for H₂-air, $$x \approx 25$$, so $$\sigma^* \approx 3.7$$, requiring $$\sigma > \sigma_{cr}$$ with run-up distances of 20–40 tube diameters ([Transition to Detonation - an overview](https://www.sciencedirect.com/topics/engineering/transition-to-detonation)). Another criterion is $$D/\lambda > 7$$, where $$D$$ is the characteristic geometrical size, and $$\lambda$$ is the detonation cell size, used for predicting DDT in confined systems.

#### Experimental and Theoretical Insights
Experimental studies, such as those in smooth tubes with hydrogen-oxygen mixtures, show that DDT run-up distance depends inversely on initial pressure, with turbulent boundary layers playing a role in controlling scales of motion ([DDT in a smooth tube filled with a hydrogen–oxygen mixture](https://link.springer.com/article/10.1007/s00193-005-0265-6)). Theoretical models, like the Zeldovich-von Neumann-Döring (ZND) for detonations, are steady-state, but DDT requires transient models, often involving numerical simulations to capture shock-flame interactions and hot spot formation.

#### Challenges and Detailed Considerations
A notable detail is the use of a progress variable instead of detailed species tracking, simplifying computations while still capturing DDT dynamics. This is effective for shockwave tube simulations, despite the complexity of combustion. Another challenge is the stochastic nature of ignition events and the impact of nonequilibrium, shock-driven turbulence, which remains an unresolved area ([Origins of the deflagration-to-detonation transition in gas-phase combustion](https://www.sciencedirect.com/science/article/abs/pii/S0010218006001817)).

For viscous effects, the Navier-Stokes equations include additional terms for viscosity and heat conduction, which are crucial for capturing boundary layer effects and flame structure, though often neglected in inviscid models for simplicity. The choice of reaction rate model, such as Arrhenius kinetics, is critical, with parameters like activation energy $$E_a$$ and pre-exponential factor $$A$$ needing calibration for specific mixtures.

#### Table of Key Equations and Parameters

| **Equation Type**         | **Equation**                                                               | **Description**                                      |
|---------------------------|---------------------------------------------------------------------------|-----------------------------------------------------|
| Mass Conservation         | $$\frac{\partial \rho}{\partial t} + \frac{\partial (\rho u)}{\partial x} = 0$$ | Ensures mass conservation during flow and reaction. |
| Momentum Conservation     | $$\frac{\partial (\rho u)}{\partial t} + \frac{\partial (\rho u^2 + p)}{\partial x} = 0$$ | Tracks momentum changes due to pressure and velocity, inviscid. |
| Energy Conservation       | $$\frac{\partial (\rho E)}{\partial t} + \frac{\partial (\rho u E + p u)}{\partial x} = \rho q \omega$$ | Includes heat release $$\rho q \omega$$ from combustion, inviscid. |
| Reaction Progress Variable| $$\frac{\partial (\rho c)}{\partial t} + \frac{\partial (\rho u c)}{\partial x} = \rho \omega$$ | Tracks reaction progress, $$\omega$$ is reaction rate. |
| Equation of State         | $$p = \rho R T$$                                                        | Relates pressure, density, and temperature.         |
| Reaction Rate (Arrhenius) | $$\omega = A (1 - c) \exp\left(-\frac{E_a}{R T}\right)$$                | Models temperature-dependent reaction rate.         |

For viscous flow, additional terms include:
- Momentum: $$\frac{\partial}{\partial x} \left( \mu \frac{\partial u}{\partial x} \right)$$
- Energy: $$\frac{\partial}{\partial x} \left( k \frac{\partial T}{\partial x} \right)$$

#### Conclusion
The equations for DDT in shockwave tubes are rooted in compressible flow dynamics with combustion, solved numerically to capture the transient transition. These models, supported by experimental validations and theoretical criteria, provide a robust framework for understanding and predicting DDT, essential for safety and engineering applications.

### Key Citations
- [Deflagration-to-detonation transition in gases in tubes with cavities](https://link.springer.com/article/10.1007/s10891-010-0448-6)
- [Deflagration and detonation induced by shock wave focusing at different Mach numbers](https://www.sciencedirect.com/science/article/pii/S1000936123002157)
- [Shock transition to detonation in channels with obstacles](https://www.sciencedirect.com/science/article/abs/pii/S1540748916302188)
- [Shock tube](https://en.wikipedia.org/wiki/Shock_tube)
- [Transition to Detonation - an overview](https://www.sciencedirect.com/topics/engineering/transition-to-detonation)
- [Deflagration to detonation transition](https://en.wikipedia.org/wiki/Deflagration_to_detonation_transition)
- [Detonation Wave - an overview](https://www.sciencedirect.com/topics/physics-and-astronomy/detonation-wave)
- [Fast Deflagration-to-Detonation Transition in Helical Tubes](https://www.mdpi.com/2227-9717/11/6/1719)
- [Shock wave](https://en.wikipedia.org/wiki/Shock_wave)
- [Normal Shock Wave Equations](https://www.grc.nasa.gov/www/k-12/airplane/normal.html)
- [Enhanced DDT mechanism from shock-flame interactions in thin channels](https://www.sciencedirect.com/science/article/abs/pii/S1540748920306271)
- [Characteristics of a 1D FDTD Simulation of Shockwave Formation in Ferrite Loaded Non-Linear Transmission Lines](https://www.osti.gov/biblio/1255027)
- [Studying Shock Wave Phenomena with a Shock Tube Application](https://www.comsol.com/blogs/studying-shock-wave-phenomena-with-a-shock-tube-application)
- [Numerical simulation of the deflagration to detonation transition behavior in explosives based on the material point method](https://www.sciencedirect.com/science/article/pii/S0010218021006635)
- [Numerical simulation of deflagration-to-detonation transition: the role of shock–flame interactions in turbulent flames](https://www.sciencedirect.com/science/article/abs/pii/S0010218098000765)
- [Normal Shock Wave Equations](https://www.grc.nasa.gov/www/BGH/normal.html)
- [Experimental and theoretical observations on DDT in smooth narrow channels](https://www.sciencedirect.com/science/article/abs/pii/S1540748920306325)
- [Evaluation of numerical schemes for capturing shock waves in modeling proppant transport in fractures](https://link.springer.com/article/10.1007/s12182-017-0194-x)
- [Ucf](https://stars.library.ucf.edu/cgi/viewcontent.cgi?httpsredir=1&article=5208&context=etd)
- [Numerical Simulations of DDT Limits in Hydrogen-Air Mixtures in Obstacle Laden Channel](https://www.mdpi.com/1996-1073/14/1/24)
- [Numerical Simulation of Gas-Phase DDT](https://ui.adsabs.harvard.edu/abs/1998APS..DFD..GL07O/abstract)
- [Finite-difference time-domain method](https://en.wikipedia.org/wiki/Finite-difference_time-domain_method)
- [Fault simulation model for i DDT testing: An investigation](https://www.researchgate.net/publication/4074601_Fault_simulation_model_for_i_DDT_testing_An_investigation)
- [Computer simulation of DDT distribution in Palos Verdes shelf sediments](https://pennstate.pure.elsevier.com/en/publications/computer-simulation-of-ddt-distribution-in-palos-verdes-shelf-sed)
- [Computer simulation of derivative TPD](https://www.sciencedirect.com/science/article/abs/pii/0040603195027092)
- [Numerical simulation and validation of flame acceleration and DDT in hydrogen air mixtures](https://www.sciencedirect.com/science/article/abs/pii/S0360319918322638)
