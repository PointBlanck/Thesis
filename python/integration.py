"""
Module containing all the code relevant to the integration of the system.
"""
# See the module of ivp_solve and debug it and maybe add print statements to track progress more easily.
# Look for different integrating package.
# Check performance against Mathematica and make it show which integration method it is using

import numpy as np
import scipy.integrate as scpint
import scipy.optimize as scpopt
import matplotlib.pyplot as plt
import datetime as dt
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
    return np.array([drdt, dphidt, dprdt, dpphidt])

def system1(y, energy, phi0, pphi0):
    """
    The system to be integrated for the calculation of the stellar motion inside the galaxy.
    """
    r0, pr0= y
    drdt = ptns.hamiltonian_dpr(r0, phi0, pr0, pphi0)
    dprdt = -ptns.hamiltonian_dr(r0, phi0, pr0, pphi0)
    return [drdt, dprdt]



# Define the function that counts events
def event(t, y):
    """
    A function that becomes 0 when phi = pi/2
    """
    return np.cos(y[1])

# Integrate
event.direction = -1
plt.ion()
fig, ax = plt.subplots(1,2,layout='constrained')
def integrate(ksi_init, pksi_init, tol, steps, rc):
    """ Integrates and stores a single Poincare instance. """
    kappac = ptns.epicyclic_frequency(rc)
    omegac = ptns.angular_velocity(rc)
    pphic = ptns.angular_velocity(rc)*rc**2
    energy = ptns.energy_cyclic(rc, pphic)
    
    # Define initial conditions
    r0 = rc - ksi_init
    phi0 = np.pi/2.0
    pr0 = -pksi_init
    pphi0 = (r0**2)*ptns.Omg_sp + r0*np.sqrt((r0**2)*(ptns.Omg_sp**2) - (pr0**2) - 2*ptns.total_potential(r0, phi0) + 2*energy)
    y0 = [r0, phi0, pr0, pphi0]
    
    # Print important quantities
    print('Epicyclic frequency:', kappac)
    print('Angular velocity:', omegac)
    print('Period:', (2*np.pi)/omegac)
    print("P_φc", pphic)
    print('Energy of cyclic movement:', energy)
    print('Number of periods:', steps)
    print('Initial conditions:', y0)

    # Integrate
    period = 2*np.pi/(omegac)
    t_span = (0, steps*period)
    time1 = dt.datetime.now()
    sol = scpint.solve_ivp(system, t_span, y0, rtol=tol, atol=tol, vectorized=True, events=event, method="Radau")
    time2 = dt.datetime.now()
    print("Integration Status Code:", sol.status)
    print(sol.message)
    print("Execution Time:", time2 - time1)
    lines.append(ax[0].scatter(rc - sol.y_events[0][1:,0], -sol.y_events[0][1:,2], c='black', s=10))
    ax[1].plot(sol.y[0]*np.cos(sol.y[1]), sol.y[0]*np.sin(sol.y[1]))

def periodic(ksi_init, pksi_init, tol, steps, rc):
    """
    Function that uses the NR method to find one of the
    stable orbits of the system that is near the initial guess.
    """
    # Important quantities
    kappac = ptns.epicyclic_frequency(rc)
    omegac = ptns.angular_velocity(rc)
    pphic = ptns.angular_velocity(rc)*rc**2
    energy = ptns.energy_cyclic(rc, pphic)

    # Initial conditions
    r0 = rc - ksi_init
    pr0 = -pksi_init
    phi0 = np.pi/2
    pphi0 = (r0**2)*ptns.Omg_sp + r0*np.sqrt((r0**2)*(ptns.Omg_sp**2) - (pr0**2) - 2*ptns.total_potential(r0, phi0) + 2*energy)
    y0 = [r0, pr0]
    sol = scpopt.root(system1, y0, args=(energy, phi0, pphi0), tol=tol, method='broyden2')
    print(rc - sol.x[0], -sol.x[1])
    ax.scatter(rc - sol.x[0], -sol.x[1])
"""
def orbit(ksi_init, pksi_init, tol, steps, rc):
    # Info
    kappac = ptns.epicyclic_frequency(rc)
    omegac = ptns.angular_velocity(rc)
    pphic = ptns.angular_velocity(rc)*rc**2
    energy = ptns.energy_cyclic(rc, pphic)

    for r0 in np.linspace(0.01, rc, steps):
    
        # Define initial conditions
        phi0 = np.pi/2.0
        pr0 = -pksi_init
        pphi0 = (r0**2)*ptns.Omg_sp + r0*np.sqrt((r0**2)*(ptns.Omg_sp**2) - (pr0**2) - 2*ptns.total_potential(r0, phi0) + 2*energy)
        y0 = [r0, phi0, pr0, pphi0]
        # Integrate
        period = 2*np.pi/(omegac)
        t_span = (0, 1.5*period)
        sol = scp.solve_ivp(system, t_span, y0, rtol=tol, events=event, method="Radau")
        r = sol.y[0,:]
        phi = sol.y[1,:]
        x = r*np.cos(phi)
        y = r*np.sin(phi)
        ax.plot(x, y)
    
    # Print important quantities
    print('Epicyclic frequency:', kappac)
    print('Angular velocity:', omegac)
    print('Period:', (2*np.pi)/omegac)
    print('P_phic:', pphic)
    print('Energy of cyclic movement:', energy)
    print('Number of periods:', steps)
    print('Initial conditions:', y0)"""

    
    
ax[0].set_xlabel("ξ")
ax[0].set_ylabel("$P_ξ$")
ax[0].set_box_aspect(1)
ax[1].set_box_aspect(1)

