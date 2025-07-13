"""
Module containing all the code relevant to the integration of the system.
"""

import sympy as smp
import numpy as np
import scipy.integrate as scp
import matplotlib.pyplot as plt
import potentials as ptns

lines = []

# Define the system to be integrated.
def system(t, y):
    """
    The system to be integrated for the calculation of the stellar motion inside the galaxy.
    """
    r, phi, pr, pphi = y
    drdt = ptns.hamiltonian_dpr(r, phi, pr, pphi)
    dphidt = ptns.hamiltonian_dpphi(r, phi, pr ,pphi)
    dprdt = -ptns.hamiltonian_dr(r, phi, pr, pphi)
    dpphidt = -ptns.hamiltonian_dphi(r, phi, pr, pphi)
    return [drdt, dphidt, dprdt, dpphidt]

# Define the function that counts events
def event(t, y):
    """
    A function that becomes 0 when phi = pi/2
    """
    return np.cos(y[1])

# Integrate
event.direction = -1
plt.ion()
fig, ax = plt.subplots(layout='constrained')
def integrate(ksi_init, pksi_init, tol, steps, rc):
    kappac = ptns.epicyclic_frequency(rc)
    omegac = ptns.angular_velocity(rc)
    pphic = ptns.angular_velocity(rc)*rc**2
    energy = ptns.energy_cyclic(rc, pphic)

    # Print important quantities
    print(kappac)
    print((2*np.pi)/omegac)
    print(pphic)
    print(energy)
    
    # Define initial conditions
    r0 = rc - ksi_init
    phi0 = np.pi/2.0
    pr0 = -pksi_init
    pphi0 = (r0**2)*ptns.Omg_sp + r0*np.sqrt((r0**2)*(ptns.Omg_sp**2) - (pr0**2) - 2*ptns.total_potential(r0, phi0) + 2*energy)
    y0 = [r0, phi0, pr0, pphi0]
    print(y0)
    period = 2*np.pi/(omegac)
    t_span = (0, steps*period)
    sol = scp.solve_ivp(system, t_span, y0, events=event, rtol=tol, method="Radau")
    lines.append(ax.scatter(rc - sol.y_events[0][1:,0], -sol.y_events[0][1:,2], c='black', s=10))
#ax.set_xlabel("r")
#ax.set_ylabel("$P_ξ$")
#ax.set_box_aspect(1)
#plt.show()