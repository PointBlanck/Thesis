""" 
Module containing the gravitational potential definitions.
Potentials are defined in array form.
"""
import numpy as np

# Global variables
G = 4.302 * 10**(-6) # (kpc km^2)/(M_sol s^2)
# Miyamoto-Nagai Disk potential constants
M_d = 8.56 * 10**(10) # M_sol
a_d = 5.3 # kpc
b_d = 0.25 # kpc
# Plummer Bulge potential constants
M_b = 5 * 10**(10) # M_sol
b = 1.5 # kpc

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