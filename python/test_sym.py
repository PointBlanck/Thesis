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
    ax[0].plot(r, np.abs(ptnsm.disk_potential(r)), label="Disk")
    ax[0].plot(r, np.abs(ptnsm.bulge_potential(r)), label="Bulge")
    ax[0].plot(r, np.abs(ptnsm.total_potential(r)), label="Total")
    #ax[0].plot(r, ptnsm.halo_potential(r), label="Halo")
    ax[0].grid(True)
    ax[0].legend()
    ax[0].set_title("Axisymmetric Potentials")
    ax[0].set_xlabel("r [kpc]")
    ax[0].set_ylabel("Log(V)")
    ax[0].set_yscale("log")
    ax[1].plot(phi, ptnsm.spiral_potential(7, phi), label="Spiral Potential")
    ax[1].legend()
    ax[1].grid(True)
    ax[1].set_title("Spiral Potential on r = 7")
    plt.show()
    print(50*"=")
    print("# End of potentials function.")
    print(50*"=")
    return 0

def preprocess(r, phi, what="all"):
    """
    Plots the f-strength of the potential perturbation.
    """
    fig, ax = plt.subplots(2, 2, layout="constrained")
    if what == "all":
        # F-Strength Plotting
        R, F = np.meshgrid(r, phi)
        for rho in [5, 15, 30]:
            F_spmax = np.max(np.sqrt((1/R)*rho*ptnsm.spiral_potential_dphi_derivative(R, F)**2 + (rho*ptnsm.spiral_potential_dr_derivative(R,F))**2), axis=0)
            F_radial = ptnsm.total_potential_derivative(r)
            ax[0][0].plot(r, F_spmax/F_radial, label=f"$ρ_0 = {rho}$")
        ax[0][0].legend()
        ax[0][0].grid(True)
        ax[0][0].set_title("F-Strength")
        ax[0][0].set_xlabel("r [kpc]")
        ax[0][0].set_ylabel("$F_{all}$")

        # Rotation curve
        U_rot = np.sqrt(r*ptnsm.total_potential_derivative(r))
        ax[0][1].plot(r, U_rot)
        ax[0][1].set_title("Rotation Curve")
        ax[0][1].set_xlabel("r")
        ax[0][1].set_ylabel("U_rot")
        ax[0][1].set_xlim(0,25)
        ax[0][1].grid(True)

        # Spiral potential plotting
        X = np.linspace(-20, 20, 1000)
        Y = np.linspace(-20, 20, 1000)
        x, y = np.meshgrid(X,Y)
        #laplacian1 = ptnsm.c_second_total_potential_dr_derivative(x,y) + (1/np.sqrt(x**2 + y**2))*ptnsm.c_total_potential_derivative(x,y)
        laplacian = ptnsm.c_second_spiral_potential_dr_derivative(x,y) + (1/np.sqrt(x**2 + y**2))*ptnsm.c_spiral_potential_dr_derivative(x,y) + (1/(x**2 + y**2))*ptnsm.c_second_spiral_potential_dphi_derivative(x,y)
        #laplacian = laplacian1 + laplacian2
        density = laplacian/(4*np.pi*ptnsm.G)
        ax[1][0].pcolormesh(X,Y,density, cmap="inferno")
        ax[1][0].set_title("Isodensity color map")
        ax[1][0].set_xlabel("x")
        ax[1][0].set_ylabel("y")
        plt.show()

    else:
        # Spiral potential plotting
        X = np.linspace(-20, 20, 1000)
        Y = np.linspace(-20, 20, 1000)
        x, y = np.meshgrid(X,Y)
        #laplacian1 = ptnsm.c_second_total_potential_dr_derivative(x,y) + (1/np.sqrt(x**2 + y**2))*ptnsm.c_total_potential_derivative(x,y)
        laplacian = ptnsm.c_second_spiral_potential_dr_derivative(x,y) + (1/np.sqrt(x**2 + y**2))*ptnsm.c_spiral_potential_dr_derivative(x,y) + (1/(x**2 + y**2))*ptnsm.c_second_spiral_potential_dphi_derivative(x,y)
        #laplacian = laplacian1 + laplacian2
        density = laplacian/(4*np.pi*ptnsm.G)
        ax[1][0].pcolormesh(X,Y,density, cmap="inferno")
        ax[1][0].set_title("Isodensity color map")
        ax[1][0].set_xlabel("x")
        ax[1][0].set_ylabel("y")
        plt.show()

    return 0

r = np.linspace(0.01, 25, 1000)
phi = np.linspace(0, 2*np.pi, 1000)
potentials(r, phi)
preprocess(r, phi)
