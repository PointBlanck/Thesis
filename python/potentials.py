"""
Module containing the galactic potentials and certain other
symbollicaly calculated and defined quantities
"""

# Import necessary modules
import numpy as np
import sympy as smp
import matplotlib.pyplot as plt

# Global constants
Omg_sp = 20
G = 4.302e-6
# Bulge constants
M_b = 5.0e10 # M_sol
r_b = 1.9 # kpc
# Disk constants
M_d = 8.56e10 # M_sol
a_d = 5.3 # kpc
b_d = 0.25 # kpc
# Spiral constants
h_z = 0.18 # kpc
r_0 = 8.0 # kpc
C = 8/(3*np.pi)
r_sp = 3.0 # kpc
rho_0 = 5.0e7
pitch_angle = -13.0
a = pitch_angle*np.pi/(180.0)

# Define the potentials symbollicaly.
# Define symbols.
r, phi, pr, pphi = smp.symbols('r phi pr pphi', real=True)
# Define bulge potential.
V_b = (-G*M_b)/(smp.sqrt(r_b**2 + r**2))
# Define disk potential.
V_d = (-G*M_d)/smp.sqrt(r**2 + (a_d + b_d)**2)
# Define spiral potential in steps.
cutoffr = (((2/np.pi)*smp.atan(r - 6.0) + 1) - 0.1)/1.9
kappa = 2/(r*smp.Abs(smp.sin(a)))
khz = kappa*h_z
beta = (1.0 + khz + 0.3*(khz)**2)/(1.0 + 0.3*khz)
g = 2*(phi - smp.log(r/r_0)/(smp.tan(a)))
cosgnew1 = (C/(kappa*beta))*smp.cos(g)
V_sp = -4.0*np.pi*G*h_z*rho_0*cutoffr*smp.exp(-(r - r_0)/r_sp)*cosgnew1
# Axisymmetric potential
V_ax = V_d + V_b
# Total potential
V_tot = V_ax + V_sp
# Axisymmetric Hamiltonian
H_ax = (pr**2)/2.0 + (pphi**2)/(2.0*r**2) - Omg_sp*pphi + V_ax
# Hamiltonian
H = H_ax + V_sp
# Calculate derivatives
dVax = smp.diff(V_ax, r)
ddVax = smp.diff(dVax, r)
dVsp_dr = smp.diff(V_sp, r)
dVsp_dphi = smp.diff(V_sp, phi)
# Movement quantities
epic = smp.sqrt(ddVax + (3/r)*dVax)
omega = smp.sqrt((1/r)*dVax)


# Lambdification
# Lambdify potentials
disk_potential = smp.lambdify(r, V_d, 'numpy')
bulge_potential = smp.lambdify(r, V_b, 'numpy')
spiral_potential = smp.lambdify((r, phi), V_sp, 'numpy')
# Lambdify axisymmetric hamiltonian
axisymmetric_hamiltonian = smp.lambdify((r,pr,phi), H_ax, 'numpy')
# Lambdify hamiltonian
hamiltonian = smp.lambdify((r,phi,pr,pphi), H, 'numpy')
# Lambdify Vax derivatives
axisymmetric_potential_d1 = smp.lambdify(r, dVax, 'numpy')
axisymmetric_potential_d2 = smp.lambdify(r, ddVax, 'numpy')
# Lambdify spiral potential derivatives
spiral_potential_dr1 = smp.lambdify((r,phi), dVsp_dr, 'numpy')
spiral_potential_dphi1 = smp.lambdify((r,phi), dVsp_dphi, 'numpy')
# Lambdify epicyclic and omega.
epicyclic_frequency = smp.lambdify(r, epic, 'numpy')
angular_velocity = smp.lambdify(r, omega, 'numpy')

# Test potentials
print("V_b =",V_b)
print("V_d =",V_d)
print("V_sp =",V_sp)
print("H_ax =", H_ax)
print("H =", H)
print("dVax/dr =", dVax)
print("d2Vax/dr2 =", ddVax)
print("k =", epic)
print("omgs =", omega)

# Plot potentials
ar = np.linspace(0.02, 20, 1000)
fi = np.linspace(0.01, 2*np.pi, 1000)
fig, ax = plt.subplots(2,2,layout='constrained')
ax[0][0].plot(ar, angular_velocity(ar), label='Ω')
ax[0][0].plot(ar, angular_velocity(ar) - epicyclic_frequency(ar)/2.0, label='Ω - κ/2')
ax[0][0].plot(ar, angular_velocity(ar) + epicyclic_frequency(ar)/2.0, label='Ω + κ/2')
ax[0][0].plot(ar, angular_velocity(ar) - epicyclic_frequency(ar)/4.0, label='Ω - κ/4')
ax[0][0].plot(ar, 15*np.ones(len(ar)))
ax[0][0].set_xlabel("r")
ax[0][0].set_ylabel("Ω")
ax[0][0].legend()
ax[0][0].set_ylim(0, 80)
ax[0][0].set_xlim(0, 20)
ax[0][0].grid(True)
plt.show()

