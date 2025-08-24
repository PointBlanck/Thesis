"""
Module containing all the code relevant to the integration of the stellar motion system.
Enhanced for numerical precision and robustness in initial conditions.
"""

import numpy as np
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass
import warnings
from statistics import mean
import scipy.integrate as scpint
import scipy.optimize as scpopt
import matplotlib.pyplot as plt
import datetime as dt
import potentials as ptns

# Configuration and validation classes
@dataclass
class IntegrationConfig:
    """Configuration parameters for integration with validation."""
    rtol: float = 1e-10
    atol: float = 1e-12
    method: str = "DOP853"  # High-order method for better precision
    max_step: Optional[float] = None
    first_step: Optional[float] = None
    
    def __post_init__(self):
        if self.rtol <= 0 or self.atol <= 0:
            raise ValueError("Tolerances must be positive")
        if self.rtol < 1e-15 or self.atol < 1e-15:
            warnings.warn("Very tight tolerances may cause numerical instability")

@dataclass
class PhysicalParameters:
    """Container for physical parameters with validation."""
    rc: float
    energy: float
    kappac: float
    omegac: float
    pphic: float
    period: float
    
    def __post_init__(self):
        if self.rc <= 0:
            raise ValueError("Circular radius must be positive")
        if self.omegac <= 0:
            raise ValueError("Angular velocity must be positive")
        if self.kappac <= 0:
            raise ValueError("Epicyclic frequency must be positive")

class InitialConditionValidator:
    """Validates and adjusts initial conditions for numerical stability."""
    
    def __init__(self, params: PhysicalParameters, tolerance: float = 1e-10):
        self.params = params
        self.tolerance = tolerance
    
    def validate_radial_coordinate(self, r0: float) -> float:
        """Ensure radial coordinate is physical and numerically stable."""
        r_min = max(1e-6, 0.001 * self.params.rc)  # Avoid singularities
        r_max = 100 * self.params.rc  # Reasonable upper bound
        
        if r0 <= 0:
            raise ValueError(f"Radial coordinate must be positive, got {r0}")
        
        if r0 < r_min:
            warnings.warn(f"r0={r0} too small, adjusting to {r_min}")
            return r_min
        
        if r0 > r_max:
            warnings.warn(f"r0={r0} too large, adjusting to {r_max}")
            return r_max
        
        return r0
    
    def compute_angular_momentum(self, r0: float, phi0: float, pr0: float, 
                               energy: float) -> float:
        """
        Compute angular momentum ensuring energy conservation and real values.
        """
        try:
            # Compute potential energy at initial position
            V0 = ptns.total_potential(r0, phi0)
            
            # Angular momentum from energy conservation
            # E = (1/2)(pr^2 + pphi^2/r^2) + V(r,phi)
            # pphi^2 = 2*r^2*(E - V - pr^2/2)
            
            discriminant = 2 * r0**2 * (energy - V0 - 0.5 * pr0**2)
            
            if discriminant < 0:
                raise ValueError(f"Negative discriminant: {discriminant}. "
                               f"Initial conditions violate energy conservation.")
            
            # Choose sign to match physical expectation (typically positive for prograde)
            pphi0_magnitude = np.sqrt(discriminant)
            
            # Add spiral pattern contribution if present
            if hasattr(ptns, 'Omg_sp'):
                pphi0 = r0**2 * ptns.Omg_sp + pphi0_magnitude
            else:
                pphi0 = pphi0_magnitude
            
            return pphi0
            
        except Exception as e:
            raise ValueError(f"Failed to compute angular momentum: {e}")
    
    def validate_initial_conditions(self, ksi_init: float, pksi_init: float, 
                                  rc: float) -> Tuple[float, float, float, float]:
        """
        Validate and potentially adjust initial conditions for numerical stability.
        
        Returns: (r0, phi0, pr0, pphi0)
        """
        # Compute physical parameters
        r0 = rc - ksi_init
        r0 = self.validate_radial_coordinate(r0)
        
        phi0 = np.pi / 2.0  # Standard initial angular position
        pr0 = -pksi_init
        
        # Validate momentum isn't too extreme
        v_escape_approx = np.sqrt(2 * abs(ptns.total_potential(r0, phi0)))
        if abs(pr0) > 10 * v_escape_approx:
            warnings.warn(f"Very large radial momentum: {pr0}, "
                         f"escape velocity ~{v_escape_approx}")
        
        # Compute angular momentum
        pphi0 = self.compute_angular_momentum(r0, phi0, pr0, self.params.energy)
        
        return r0, phi0, pr0, pphi0

# Enhanced system definitions
def system(t: float, y: np.ndarray) -> np.ndarray:
    """
    Enhanced system definition with error checking.
    
    Args:
        t: Time (unused in autonomous system)
        y: State vector [r, phi, pr, pphi]
    
    Returns:
        Derivative vector [dr/dt, dphi/dt, dpr/dt, dpphi/dt]
    """
    if len(y) != 4:
        raise ValueError(f"State vector must have 4 components, got {len(y)}")
    
    r, phi, pr, pphi = y
    
    # Check for physical validity
    if r <= 0:
        raise ValueError(f"Radial coordinate must be positive: r={r}")
    
    try:
        drdt = ptns.hamiltonian_dpr(r, phi, pr, pphi)
        dphidt = ptns.hamiltonian_dpphi(r, phi, pr, pphi)
        dprdt = -ptns.hamiltonian_dr(r, phi, pr, pphi)
        dpphidt = -ptns.hamiltonian_dphi(r, phi, pr, pphi)
        
        # Check for numerical issues
        derivatives = np.array([drdt, dphidt, dprdt, dpphidt])
        if not np.all(np.isfinite(derivatives)):
            raise ValueError(f"Non-finite derivatives at r={r}, phi={phi}")
        
        return derivatives
        
    except Exception as e:
        raise RuntimeError(f"Error computing derivatives: {e}")

def system1(y: np.ndarray, energy: float, phi0: float, pphi0: float) -> List[float]:
    """
    Reduced system for finding periodic orbits.
    
    Args:
        y: State vector [r, pr]
        energy: Total energy
        phi0: Fixed angular coordinate
        pphi0: Fixed angular momentum
    
    Returns:
        [dr/dt, dpr/dt]
    """
    if len(y) != 2:
        raise ValueError(f"State vector must have 2 components, got {len(y)}")
    
    r0, pr0 = y
    
    if r0 <= 0:
        return [1e10, 1e10]  # Large values to push away from r=0
    
    try:
        drdt = ptns.hamiltonian_dpr(r0, phi0, pr0, pphi0)
        dprdt = -ptns.hamiltonian_dr(r0, phi0, pr0, pphi0)
        return [drdt, dprdt]
    except:
        return [1e10, 1e10]

# Event detection
def poincare_event(t: float, y: np.ndarray) -> float:
    """Enhanced event function for Poincaré sections."""
    return np.cos(y[1])  # Zero when phi = π/2 + nπ

poincare_event.direction = -1
poincare_event.terminal = False

class StellarMotionIntegrator:
    """Main class for stellar motion integration with enhanced robustness."""
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        self.lines = []
        
        # Set up plotting
        plt.ion()
        self.fig, self.ax = plt.subplots(layout='constrained', figsize=(10, 8))
        self.ax.set_xlabel("ξ")
        self.ax.set_ylabel("$P_ξ$")
        self.ax.set_box_aspect(1)
        self.ax.grid(True, alpha=0.3)
    
    def compute_physical_parameters(self, rc: float) -> PhysicalParameters:
        """Compute and validate physical parameters."""
        try:
            kappac = ptns.epicyclic_frequency(rc)
            omegac = ptns.angular_velocity(rc)
            pphic = omegac * rc**2
            energy = ptns.energy_cyclic(rc, pphic)
            period = 2 * np.pi / omegac
            
            return PhysicalParameters(
                rc=rc, energy=energy, kappac=kappac, 
                omegac=omegac, pphic=pphic, period=period
            )
        except Exception as e:
            raise ValueError(f"Failed to compute physical parameters: {e}")
    
    def integrate(self, ksi_init: float, pksi_init: float, 
                  steps: int, rc: float, 
                  custom_tol: Optional[float] = None) -> dict:
        """
        Enhanced integration with comprehensive error handling and validation.
        
        Args:
            ksi_init: Initial radial displacement from circular orbit
            pksi_init: Initial radial momentum
            steps: Number of orbital periods to integrate
            rc: Circular orbit radius
            custom_tol: Optional custom tolerance
        
        Returns:
            Dictionary with integration results and metadata
        """
        # Compute physical parameters
        params = self.compute_physical_parameters(rc)
        
        # Validate initial conditions
        validator = InitialConditionValidator(params)
        r0, phi0, pr0, pphi0 = validator.validate_initial_conditions(
            ksi_init, pksi_init, rc
        )
        
        y0 = [r0, phi0, pr0, pphi0]
        
        # Print diagnostic information
        print(f"=== Integration Parameters ===")
        print(f"Epicyclic frequency: {params.kappac:.6e}")
        print(f"Angular velocity: {params.omegac:.6e}")
        print(f"Period: {params.period:.6e}")
        print(f"Angular momentum (circular): {params.pphic:.6e}")
        print(f"Energy: {params.energy:.6e}")
        print(f"Integration steps: {steps}")
        print(f"Initial conditions: r0={r0:.6e}, phi0={phi0:.6e}, "
              f"pr0={pr0:.6e}, pphi0={pphi0:.6e}")
        
        # Verify energy conservation at initial conditions
        initial_energy = (0.5 * (pr0**2 + pphi0**2 / r0**2) + 
                         ptns.total_potential(r0, phi0))
        energy_error = abs(initial_energy - params.energy) / abs(params.energy)
        print(f"Initial energy error: {energy_error:.6e}")
        
        if energy_error > 1e-6:
            warnings.warn(f"Large initial energy error: {energy_error}")
        
        # Set up integration
        tol = custom_tol or self.config.rtol
        t_span = (0, steps * params.period)
        
        # Adaptive step size control
        max_step = params.period / 1000  # At least 1000 points per period
        
        print(f"=== Integration Setup ===")
        print(f"Time span: {t_span}")
        print(f"Tolerance: rtol={tol}, atol={self.config.atol}")
        print(f"Method: {self.config.method}")
        print(f"Max step: {max_step}")
        
        # Integrate with timing
        time1 = dt.datetime.now()
        try:
            sol = scpint.solve_ivp(
                system, t_span, y0, 
                rtol=tol, atol=self.config.atol,
                method=self.config.method,
                vectorized=False,
                events=poincare_event,
                max_step=max_step,
                dense_output=True
            )
            time2 = dt.datetime.now()
            
        except Exception as e:
            raise RuntimeError(f"Integration failed: {e}")
        
        # Analyze results
        print(f"=== Integration Results ===")
        print(f"Status: {sol.status} - {sol.message}")
        print(f"Execution time: {time2 - time1}")
        print(f"Function evaluations: {sol.nfev}")
        print(f"Solution points: {len(sol.t)}")
        
        if hasattr(sol, 'y_events') and len(sol.y_events[0]) > 1:
            events = sol.y_events[0][1:]  # Skip first event
            ksi_events = rc - events[:, 0]
            pksi_events = -events[:, 2]
            
            print(f"Poincaré crossings: {len(events)}")
            
            # Plot results
            scatter = self.ax.scatter(ksi_events, pksi_events, 
                                    c='black', s=10, alpha=0.7)
            self.lines.append(scatter)
            
            # Return comprehensive results
            return {
                'success': sol.status == 0,
                'solution': sol,
                'events': events,
                'ksi_events': ksi_events,
                'pksi_events': pksi_events,
                'parameters': params,
                'execution_time': time2 - time1,
                'energy_error': energy_error
            }
        else:
            warnings.warn("No Poincaré crossings found")
            return {
                'success': sol.status == 0,
                'solution': sol,
                'events': None,
                'parameters': params,
                'execution_time': time2 - time1,
                'energy_error': energy_error
            }
    
    def find_periodic_orbit(self, ksi_init: float, pksi_init: float, 
                           rc: float, custom_tol: Optional[float] = None) -> dict:
        """
        Enhanced periodic orbit finder with better convergence.
        """
        params = self.compute_physical_parameters(rc)
        validator = InitialConditionValidator(params)
        
        # Initial guess
        r0 = rc - ksi_init
        r0 = validator.validate_radial_coordinate(r0)
        pr0 = -pksi_init
        phi0 = np.pi / 2
        
        try:
            pphi0 = validator.compute_angular_momentum(r0, phi0, pr0, params.energy)
            y0 = [r0, pr0]
            
            # Multiple solution methods for robustness
            methods = ['broyden2', 'hybr', 'lm']
            
            for method in methods:
                try:
                    sol = scpopt.root(
                        system1, y0, 
                        args=(params.energy, phi0, pphi0),
                        tol=custom_tol or 1e-10,
                        method=method
                    )
                    
                    if sol.success:
                        ksi_final = rc - sol.x[0]
                        pksi_final = -sol.x[1]
                        
                        print(f"Periodic orbit found using {method}:")
                        print(f"ξ = {ksi_final:.6e}, P_ξ = {pksi_final:.6e}")
                        print(f"Residual: {np.linalg.norm(sol.fun):.6e}")
                        
                        # Plot result
                        self.ax.scatter(ksi_final, pksi_final, 
                                      c='red', s=50, marker='*')
                        
                        return {
                            'success': True,
                            'ksi': ksi_final,
                            'pksi': pksi_final,
                            'residual': np.linalg.norm(sol.fun),
                            'method': method,
                            'solution': sol
                        }
                        
                except Exception as e:
                    print(f"Method {method} failed: {e}")
                    continue
            
            raise RuntimeError("All root-finding methods failed")
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def clear_plot(self):
        """Clear previous integration results from plot."""
        for line in self.lines:
            line.remove()
        self.lines.clear()
        self.ax.figure.canvas.draw()

# Convenience functions for backward compatibility
def integrate(ksi_init: float, pksi_init: float, tol: float, 
              steps: int, rc: float):
    """Backward compatible integration function."""
    integrator = StellarMotionIntegrator()
    return integrator.integrate(ksi_init, pksi_init, steps, rc, tol)

def periodic(ksi_init: float, pksi_init: float, tol: float, 
             steps: int, rc: float):
    """Backward compatible periodic orbit finder."""
    integrator = StellarMotionIntegrator()
    return integrator.find_periodic_orbit(ksi_init, pksi_init, rc, tol)
