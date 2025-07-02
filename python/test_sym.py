"""
Testing of the symbolic potentials
"""

import potentials_sym as ptnsm
import matplotlib.pyplot as plt
import numpy as np
import sympy as smp

def potentials(r, phi):
    """
    Prints and plots symbolically defined potentials.
    """
    print(50*"=")
    print("# Printing potentials in symbolic form...")
    print(50*"=")
    print("V_d =", ptnsm.V_d)
    print(50*"=")
    print("V_b =",ptnsm.V_b)
    #print(50*"=")
    #print(ptnsm.V_h)
    print(50*"=")
    print("V_sp =",ptnsm.V_sp)
    print(50*"=")
    print("# End of printing")
    print(50*"=")
    print("# Plotting potentials...")

    #??????????????????????????????????????????????????
    # Add titles on axes and units of measurement.
    #??????????????????????????????????????????????????
    fig, ax = plt.subplots(1, 2, layout="constrained")
    ax[0].plot(r, ptnsm.disk_potential(r), label="Disk")
    ax[0].plot(r, ptnsm.bulge_potential(r), label="Bulge")
    #ax[0].plot(r, ptnsm.halo_potential(r), label="Halo")
    ax[0].grid(True)
    ax[0].legend()
    ax[0].set_title("Axisymmetric Potentials")
    ax[1].plot(phi, ptnsm.spiral_potential(7, phi), label="Spiral Potential")
    ax[1].legend()
    ax[1].grid(True)
    ax[1].set_title("Spiral Potential on r = 7")
    plt.show()
    print(50*"=")
    print("# End of potentials function.")
    print(50*"=")
    return 0

def f_strength(r, phi):
    """
    Plots the f-strength of the potential perturbation.
    """
    R, F = np.meshgrid(r, phi)
    
    fig, ax = plt.subplots()
    for rho in [5, 15, 30]:
        F_spmax = np.max(np.sqrt((1/R)*rho*ptnsm.spiral_potential_dphi_derivative(R, F)**2 + (rho*ptnsm.spiral_potential_dr_derivative(R,F))**2), axis=0)
        F_radial = ptnsm.total_potential_derivative(r)
        ax.plot(r, F_spmax/F_radial, label=f"$ρ_0 = {rho}$")
    ax.legend()
    ax.grid(True)
    ax.set_title("F-Strength")
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("$F_{all}$")
    plt.show()
    return  F_spmax

r = np.linspace(0.01, 100, 10000)
phi = np.linspace(0, 2*np.pi, 1000)
#potentials(r, phi)
r = np.linspace(0.01, 25, 1000)
f_strength(r, phi)
