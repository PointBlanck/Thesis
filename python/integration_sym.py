"""
Module containing code used for integrating the dynamical galactic system.
Specifically designed to maintain certain quaulitative functions.
"""

# Import necessary modules
import scipy.integrate as scp
import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
import potentials_sym as ptnsm

# Define some parameters
r_c = 2.75 # Radius from center of galaxy in kpc
period = 2*np.pi/(ptnsm.omega(r_c))

# Define the hamiltonian and its partial derivatives.
r, phi, pr, pphi = smp.symbols('r phi pr pphi' , real=True)
H_ax = pr**2/2.0 + (pphi**2)/(2*r**2) - ptnsm.Omega*pphi + ptnsm.V_tot
H = H_ax + ptnsm.V_sp
dH_dpr = smp.diff(H, pr)
dH_dpphi = smp.diff(H, pphi)
dH_dr = smp.diff(H, r)
dH_dphi = smp.diff(H, phi)

# Lambdify the functions
hamiltonian = smp.lambdify((r,phi,pr,pphi), H, 'numpy')
hamiltonian_dr = smp.lambdify((r, phi, pr, pphi), dH_dr, 'numpy')
hamiltonian_dphi = smp.lambdify((r, phi, pr, pphi), dH_dphi, 'numpy')
hamiltonian_dpr = smp.lambdify((r, phi, pr, pphi), dH_dpr, 'numpy')
hamiltonian_dpphi = smp.lambdify((r, phi, pr, pphi), dH_dpphi, 'numpy')

# Calculate the initial values
ksi0 = 0.1
pksi0 = 50.0
r0 = r_c - ksi0
phi0 = np.pi/2.0
pr0 = -pksi0
pphic = (r_c**2) * ptnsm.omega(r_c)
H_j = (pphic**2)/(2.0*(r_c**2)) + ptnsm.total_potential(r_c) - ptnsm.Omega*pphic
print(H_j)
pphi0 = (r0**2)*ptnsm.Omega + r0*np.sqrt((r0**2)*(ptnsm.Omega**2) - (pr0**2)-2.0*(ptnsm.total_potential(r0) + ptnsm.spiral_potential(r0, phi0)) + 2.0*H_j)
y0 = [r0, phi0, pr0, pphi0]
t_span = (0,50*period)

# Define system to be integrated.
def system(t, y):
    r, phi, pr, pphi = y
    drdt = hamiltonian_dpr(r, phi, pr, phi) 
    dphidt = hamiltonian_dpphi(r, phi, pr, pphi)
    dprdt = -hamiltonian_dr(r, phi, pr, pphi)
    dpphidt = -hamiltonian_dphi(r, phi, pr, pphi)
    return [drdt, dphidt, dprdt, dpphidt]

# Define a function that checks if the body has performed a section of the hyperplane.
def event(t, y):
    """
    Event function for keeping track of the poincare section points.
    """
    return np.cos(y[1])


# Solve the system
event.direction = -1



fig, ax = plt.subplots(1,2,layout='constrained')
for ksi0 in [0.3, 0.5, 0.7, 0.9, 1.0]:
    y0[0] = r_c - ksi0
    sol = scp.solve_ivp(system, t_span, y0, events=event, rtol=3e-14, atol=1e-15, method='Radau')
    ksi = r_c - sol.y_events[0][:,0]
    pksi = -sol.y_events[0][:,2]
    ax[0].scatter(ksi, pksi, s=20, c='black')
ax[0].set_title('Poincare')
ax[1].plot(np.arange(0, sol.y_events[0][:, 0].size), np.abs(H.subs(r,y0[0]).subs(phi,y0[1]).subs(pr,y0[2]).subs(pphi,y0[3]).evalf() - hamiltonian(sol.y_events[0][:,0],sol.y_events[0][:,1],sol.y_events[0][:,2],sol.y_events[0][:,3])), label='Energy Difference')
plt.show()

#??????????????????????????????????????????????????????????????????????????????
# Pretty good for now! Just slow.
#??????????????????????????????????????????????????????????????????????????????