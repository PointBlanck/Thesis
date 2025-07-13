"""
Module containing the definition of the galactic gravitational potentials.
Lines pertaining to the halo potential have been commented off.
"""

# Import necessary modules
import sympy as smp

# Global constants
Omega = 15.0
G = 4.302 * 10**(-6)
# Bulge constants
M_b = 5.0 * 10**(10)
b = 1.9
# Halo constants
#M_h0 = 10.7 * 10**(10)
#r_h = 12.0
#gamma = 1.02
#r_hmax = 100.0
# Disk constants
R_d = 2.0
M_d = 8.56 * 10**(10)
a_d = 5.3
b_d = 0.25
# Spiral constants
h_z = 0.18
r_0 = 8.0
C = 8/(3*smp.pi)
R_s = 3.0
R_s0 = 6.0
rho = 5.0 * 10**(7)
pitch_angle = -13.0
a = pitch_angle*smp.pi/180.0
rho_0 = 15.0*10**7 # Remember to change this for preprocessing stage if you want to have the same results as the paper.

# Define the potentials symbolically.
r, phi = smp.symbols("r phi", real=True)
x, y = smp.symbols("x y", real=True)
#M_h = (M_h0*(r/r_h)**(gamma + 1))/(1 + (r/r_h)**gamma)
cutoff = (((2/smp.pi)*smp.atan(r - R_s0) + 1) - 0.1)/1.9
kappa = 2/(r*smp.Abs(smp.sin(a)))
beta = (1 + kappa*h_z + 0.3*(kappa*h_z)**2)/(1 + 0.3*kappa*h_z)
g = 2.0*(phi - smp.log(r/r_0)/smp.tan(a))
sumxy1 = C/(kappa*beta)*smp.cos(g)
V_b = -G*M_b/smp.sqrt(r**2 + b**2)
#V_h = -(G*M_h/r) - (G*M_h0/(gamma*r_h))*(gamma/(1 + (r/r_h)**gamma) - smp.log(1 + (r/r_h)**gamma))
V_d = -G*M_d/smp.sqrt(r**2 + (a_d + b_d)**2)
V_sp = -4.0*smp.pi*G*h_z*rho_0*cutoff*smp.exp(-(r - r_0)/R_s)*sumxy1
V_tot = V_d + V_b

# Calculate the first order partial derivatives needed.
dV_totdr = smp.diff(V_tot, r)
dV_spdr = smp.diff(V_sp, r)
dV_spdphi = smp.diff(V_sp, phi)

#Calculate the second order derivatives needed.
d2V_totdr2 = smp.diff(dV_totdr, r)
d2V_spdr2 = smp.diff(dV_spdr, r)
d2V_spdphi2 = smp.diff(dV_spdphi, phi)

# Converting them into cartesian coordinates
subs = {
    r: smp.sqrt(x**2 + y**2),
    phi: smp.atan2(y, x)
}
c_dV_totdr = dV_totdr.subs(subs)
c_dV_spdr = dV_spdr.subs(subs)
c_dV_spdphi = dV_spdphi.subs(subs)
c_d2V_totdr2 = d2V_totdr2.subs(subs)
c_d2V_spdr2 = d2V_spdr2.subs(subs)
c_d2V_spdphi2 = d2V_spdphi2.subs(subs)

# Define angular velocity and epicyclic frequency
oo = smp.sqrt((1/r)*dV_totdr)
epic = smp.sqrt(d2V_totdr2 + (3/r)*dV_totdr)

# Lambdify all potentials so they can be worked with numpy.
disk_potential = smp.lambdify(r, V_d, 'numpy')
bulge_potential = smp.lambdify(r, V_b, 'numpy')
total_potential = smp.lambdify(r, V_tot, 'numpy')
#halo_potential = smp.lambdify(r, V_h, 'numpy')
spiral_potential = smp.lambdify((r, phi), V_sp, 'numpy')
total_potential_derivative = smp.lambdify(r, dV_totdr, 'numpy')
spiral_potential_dr_derivative = smp.lambdify((r, phi), dV_spdr, 'numpy')
spiral_potential_dphi_derivative = smp.lambdify((r,phi), dV_spdphi, 'numpy')
second_total_potential_dr_derivative = smp.lambdify(r, d2V_totdr2, 'numpy')
second_spiral_potential_dr_derivative = smp.lambdify((r, phi), d2V_spdr2, 'numpy')
second_spiral_potential_dphi_derivative = smp.lambdify((r, phi), d2V_spdphi2, 'numpy')

# Lambdify omega and epicyclic
omega = smp.lambdify(r, oo, 'numpy')
epicyclic = smp.lambdify(r, epic, 'numpy')

# Convert them into cartesian coordinates
c_total_potential_derivative = smp.lambdify((x, y), c_dV_totdr, 'numpy')
c_spiral_potential_dr_derivative = smp.lambdify((x, y), c_dV_spdr, 'numpy')
c_spiral_potential_dphi_derivative = smp.lambdify((x,y), c_dV_spdphi, 'numpy')
c_second_total_potential_dr_derivative = smp.lambdify((x, y), c_d2V_totdr2, 'numpy')
c_second_spiral_potential_dr_derivative = smp.lambdify((x, y), c_d2V_spdr2, 'numpy')
c_second_spiral_potential_dphi_derivative = smp.lambdify((x, y), c_d2V_spdphi2, 'numpy')

