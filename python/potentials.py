"""
Module containing the galactic potentials and certain other
symbollicaly calculated and defined quantities
"""

# Import necessary modules
import numpy as np
import sympy as smp
import matplotlib.pyplot as plt

# Global constants
Omg_sp = 15
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
# Energy of cyclic movement
en = (pphi**2)/(2*r**2) + V_ax - Omg_sp*pphi
# Axisymmetric Hamiltonian
H_ax = (pr**2)/2.0 + (pphi**2)/(2.0*r**2) - Omg_sp*pphi + V_ax
# Hamiltonian
H = H_ax + V_sp
# Calculate derivatives
dVax = smp.diff(V_ax, r)
ddVax = smp.diff(dVax, r)
dVsp_dr = smp.diff(V_sp, r)
dVsp_dphi = smp.diff(V_sp, phi)
ddVsp_drdr = smp.diff(dVsp_dr, r)
ddVsp_dphidphi = smp.diff(dVsp_dphi, phi)
delta = ddVsp_drdr + (1/r)*dVsp_dr + 1/(r**2)*ddVsp_dphidphi
dH_dr = smp.diff(H, r)
dH_dphi = smp.diff(H, phi)
dH_dpr = smp.diff(H, pr)
dH_dpphi = smp.diff(H, pphi)
# Movement quantities
epic = smp.sqrt(ddVax + (3/r)*dVax)
omega = smp.sqrt((1/r)*dVax)


# Lambdification
# Lambdify potentials
disk_potential = smp.lambdify(r, V_d, 'numpy')
bulge_potential = smp.lambdify(r, V_b, 'numpy')
spiral_potential = smp.lambdify((r, phi), V_sp, 'numpy')
# Axisymmetric potential
axisymmetric_potential = smp.lambdify(r, V_ax, 'numpy')
# Total potential
total_potential = smp.lambdify((r,phi), V_tot, 'numpy')
# Energy of cyclic movement
energy_cyclic = smp.lambdify((r, pphi), en, 'numpy')
# Lambdify axisymmetric hamiltonian
axisymmetric_hamiltonian = smp.lambdify((r,pr,phi), H_ax, 'numpy')
# Lambdify hamiltonian
hamiltonian = smp.lambdify((r,phi,pr,pphi), H, 'numpy')
# Lambdify Vax derivatives
axisymmetric_potential_d1 = smp.lambdify(r, dVax, 'numpy')
axisymmetric_potential_d2 = smp.lambdify(r, ddVax, 'numpy')
hamiltonian_dr = smp.lambdify((r,phi,pr,pphi), dH_dr, 'numpy')
hamiltonian_dphi = smp.lambdify((r,phi,pr,pphi), dH_dphi, 'numpy')
hamiltonian_dpr = smp.lambdify((r,phi,pr,pphi), dH_dpr, 'numpy')
hamiltonian_dpphi = smp.lambdify((r,phi,pr,pphi), dH_dpphi, 'numpy')
laplacian = smp.lambdify((r, phi), delta, 'numpy')
# Lambdify spiral potential derivatives
spiral_potential_dr1 = smp.lambdify((r,phi), dVsp_dr, 'numpy')
spiral_potential_dphi1 = smp.lambdify((r,phi), dVsp_dphi, 'numpy')
# Lambdify epicyclic and omega.
epicyclic_frequency = smp.lambdify(r, epic, 'numpy')
angular_velocity = smp.lambdify(r, omega, 'numpy')

# Print potentials and other important quantities
"""print("V_b =",V_b)
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
# Plot angular velocity
ax[0][0].plot(ar, angular_velocity(ar), label='Ω')
ax[0][0].plot(ar, angular_velocity(ar) - epicyclic_frequency(ar)/2.0, label='Ω - κ/2')
ax[0][0].plot(ar, angular_velocity(ar) + epicyclic_frequency(ar)/2.0, label='Ω + κ/2')
ax[0][0].plot(ar, angular_velocity(ar) - epicyclic_frequency(ar)/4.0, label='Ω - κ/4')
ax[0][0].plot(ar, 15*np.ones(len(ar)))
ax[0][0].set_xlabel("r")
ax[0][0].set_ylabel("Ω")
ax[0][0].legend()
ax[0][0].grid(True)
ax[0][0].set_ylim(0, 80)
ax[0][0].set_xlim(0, 20)
ax[0][0].set_box_aspect(1)
# Plot potentials
ax[0][1].plot(ar, disk_potential(ar), label='Disk')
ax[0][1].plot(ar, bulge_potential(ar), label='Bulge')
ax[0][1].plot(ar, axisymmetric_potential(ar), label='Axisymmetric (Disk + Bulge)')
ax[0][1].set_title('Axisymmetric Potential')
ax[0][1].set_xlim(0,20)
ax[0][1].set_box_aspect(1)
ax[0][1].set_xlabel('r')
ax[0][1].set_ylabel('V')
ax[0][1].grid(True)
ax[0][1].legend()
# Plot spiral density and minima
# Calculate spiral density
x, y = np.meshgrid(np.linspace(-20,20,1000), np.linspace(-20,20,1000))
r = np.sqrt(x**2 + y**2)
phi = np.atan2(y,x)
density = laplacian(r,phi)/(4.0*np.pi*G)
# Calculate minima
# Define mask
grad_den_y, grad_den_x = np.gradient(density)
mask1 = (np.abs(np.cos(phi)*grad_den_x + np.sin(phi)*grad_den_y) < 20000)
mask2 = np.sqrt(x**2 + y**2) < 10
mask3 = np.sqrt(x**2 + y**2) > 2
mask4 = density > 0.0
mask = mask1*mask2*mask3*mask4
minima_x = np.where(mask, x, False)
minima_y = np.where(mask, y, False)
minima_x_arr = minima_x[minima_x != 0.0]
minima_y_arr = minima_y[minima_y != 0.0]
pos = ax[1][0].pcolor(x, y, density, cmap='magma')
ax[1][0].scatter(minima_x_arr, minima_y_arr, c='black', s=5)
ax[1][0].set_box_aspect(1)
ax[1][0].set_xlabel('x')
ax[1][0].set_ylabel('y')
ax[1][0].set_title('Spiral density and minima')
ax[1][0].set_aspect('auto')
fig.colorbar(pos, ax=ax[1][0])
print('Max of spiral:', np.max(density))
# Plot the f-strength
ar, fi = np.meshgrid(ar, fi)
for i in [1, 3, 6]:
    f_sp = np.max(np.sqrt(((1/ar)*i*spiral_potential_dphi1(ar, fi))**2 + (i*spiral_potential_dr1(ar,fi))**2), axis=0)
    f_ax = axisymmetric_potential_d1(ar[0,:])
    f_strength = f_sp/f_ax
    ax[1][1].plot(ar[0,:], f_strength, label=f'$ρ_0 = {i*5}$')
    ax[1][1].set_box_aspect(1)
    ax[1][1].set_title('F-Strength')
    ax[1][1].legend()
    ax[1][1].set_xlim(0, 20)
    ax[1][1].set_ylim(0, 0.3)
    ax[1][1].set_xlabel('r[kpc]')
    ax[1][1].grid(True)
    ax[1][1].set_ylabel('$F_{all}$')
plt.show()
"""
