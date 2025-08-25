"""
Module containing all the code relevant to the integration of the system.
"""
# See the module of ivp_solve and debug it and maybe add print statements to track progress more easily.
# Look for different integrating package.
# Check performance against Mathematica and make it show which integration method it is using

import numpy as np
import scipy.integrate as scpint
import scipy.optimize as scpopt
from scipy.linalg import solve, det, inv
import matplotlib.pyplot as plt
import numdifftools as nd
from functools import partial
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

# Define the function that counts events
def event(t, y):
    """
    A function that becomes 0 when phi = pi/2
    """
    return np.cos(y[1])

def F(x, t_span, phi0, pphi0, tol):
    """ Encapsulation of integrating function for NR method. """
    y0 = [x[0], phi0, x[1], pphi0]
    sol = scpint.solve_ivp(system, t_span, y0, rtol=tol, atol=tol, vectorized=True, events=event, method="Radau")
    return np.array([sol.y_events[0][1][0], sol.y_events[0][1][2]])

# Integrate
plt.ion()
fig, ax = plt.subplots(layout='constrained')
def integrate(ksi_init, pksi_init, tol, steps, rc):
    """ Integrates and stores a single Poincare instance. """
    event.direction = -1
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
    print("P_φc:", pphic)
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
    lines.append(ax.scatter(rc - sol.y_events[0][1:,0], -sol.y_events[0][1:,2], c='black', s=10))

def periodic(ksi_init, pksi_init, accuracy, a, rc):
    """
    Function that uses the NR method to find one of the
    stable orbits of the system that is near the initial guess.
    """
    # Initialization
    event.direction = -1
    tol=1e-6
    iter = 0
    accurate = False
    # Important quantities
    kappac = ptns.epicyclic_frequency(rc)
    omegac = ptns.angular_velocity(rc)
    pphic = ptns.angular_velocity(rc)*rc**2
    energy = ptns.energy_cyclic(rc, pphic)

    # Initial conditions
    # Transform into natural variables
    t_span = (0, 10*(2*np.pi)/omegac)
    r0 = rc - ksi_init
    pr0 = -pksi_init
    phi0 = np.pi/2
    pphi0 = (r0**2)*ptns.Omg_sp + r0*np.sqrt((r0**2)*(ptns.Omg_sp**2) - (pr0**2) - 2*ptns.total_potential(r0, phi0) + 2*energy)
    y0 = [r0, phi0, pr0, pphi0]
    print(y0)

    # Print important quantities
    print('Epicyclic frequency:', kappac)
    print('Angular velocity:', omegac)
    print('Period:', (2*np.pi)/omegac)
    print("P_φc:", pphic)
    print('Energy of cyclic movement:', energy)
    print('Initial conditions:', y0)
    print("Calculating periodic orbit...")

    while not accurate:
        iter += 1
        # Calculate Jacobian
        F_wrapped = partial(F, t_span=t_span, phi0=y0[1], pphi0=y0[3], tol=tol)
        jacob = nd.Jacobian(F_wrapped, step=1.5e-8)([y0[0], y0[2]]) - np.eye(2)
        if det(jacob) != 0:
            Gx = F_wrapped([y0[0],y0[2]]) - np.array([y0[0], y0[2]])
            x = -solve(jacob,Gx.T)
            r0 += a*x[0]
            pr0 += a*x[1]
            print(f"Iteration {iter}:",rc - r0, -pr0, "det(J)=", det(jacob))
            if abs(x[0]) <= accuracy and abs(x[1]) <= accuracy:
                print("Fixed Point:", rc - r0, -pr0, "Accuracy:", x[0], x[1])
                ax.scatter(rc - r0, -pr0, c='red', s=30)
                accurate = True
            pphi0 = (r0**2)*ptns.Omg_sp + r0*np.sqrt((r0**2)*(ptns.Omg_sp**2) - (pr0**2) - 2*ptns.total_potential(r0, phi0) + 2*energy)
            y0 = [r0, phi0, pr0, pphi0]
        else:
            print("Error with the determinant")

ax.set_xlabel("ξ")
ax.set_ylabel("$P_ξ$")
ax.set_box_aspect(1)

