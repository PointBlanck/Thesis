""" 
Module containing the gravitational potential definitions.
Potentials are defined in array form.
"""
import numpy as np

#?????????????????????????????????????????????????
# Check these numbers against the mathematica notebook
#?????????????????????????????????????????????????
# Global variables
G = 4.302 * 10**(-6) # (kpc km^2)/(M_sol s^2)
# Miyamoto-Nagai Disk potential constants
M_d = 8.56 * 10**(10) # M_sol
a_d = 5.3 # kpc
b_d = 0.25 # kpc
# Plummer Bulge potential constants
M_b = 5.0 * 10**(10) # M_sol
beta = 1.5 # kpc
# Gamma Halo potential constants
M_h0 = 10.7 * 10**(10) # M_sol
gamma = 1.02
r_hmax = 100 # kpc
r_h = 12.0 # kpc
# Logarithmic Spiral potential constants
h_z = 0.18 # kpc
rho_0 = 5.0 * 10**7 # M_sol/(kpc^(-3))
C = 8.0/(3.0*np.pi)
consts = 4.0*np.pi*G*h_z*rho_0*C
pitch_angle = -13.0 # degrees
alpha = (pitch_angle*np.pi)/180.0 # rads
b = 0.474
c = 0.335
R_s0 = 6.0 # kpc
r_0 = 8.0 # kpc
R_s = 7.0 # kpc

def disk_potential(r):
    """ 
    Miyamoto-Nagai Model for the disk gravitational potential component.
    A 2D disk is defined here (z = 0)
    """
    return (-G*M_d)/np.sqrt(r**2 + (a_d + b_d)**2)

def bulge_potential(r):
    """
    Plummer Model for the bulge gravitational potential component.
    """
    return (-G*M_b)/np.sqrt(r**2 + b**2)

#??????????????????????????????????????????????????????
# Need to check the formula of the halo potential.
# The paper most likely contains a mistake.
#??????????????????????????????????????????????????????
def halo_potential(r):
    """
    Gamma Model for the halo gravitational potential component.
    """
    r_ratio = r/r_h
    M_hr = (M_h0*((r_ratio)**(gamma+1.0)))/(1 + r_ratio**gamma)
    return ((-G*M_hr)/r) + ((G*M_h0)/(gamma*r))*((-gamma/(1+(r_ratio**gamma))) + (np.log(1 + r_ratio)**gamma))

def spiral_potential(r, phi):
    """
    Logarithmic Model for the spiral gravitational perturbation potential.
    """
    K = 2/(r*np.abs(np.sin(alpha)))
    khz = K*h_z
    B = (1.0 + khz + 0.3*(khz)**2)/(1 + 0.3*khz)
    return consts*(b - c*np.atan(R_s0 - r))*np.exp(-(r - r_0)/R_s)*np.cos(2*(phi - (np.log(r/r_0)/np.tan(alpha))))