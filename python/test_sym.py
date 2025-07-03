"""
Testing of the symbolic potentials
"""

import potentials_sym as ptnsm
import matplotlib.pyplot as plt
import numpy as np
import sympy as smp
import scipy as scp

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
    #fig, ax = plt.subplots(2, 2, layout="constrained")
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

        # Angular velocity plotting
        Omega = np.sqrt((1/r)*ptnsm.total_potential_derivative(r))
        epic_freq = np.sqrt(ptnsm.second_total_potential_dr_derivative(r) + (3/r)*ptnsm.total_potential_derivative(r))
        ax[1][1].plot(r, Omega, label="Ω")
        ax[1][1].plot(r, Omega+(epic_freq/2), label="Ω + κ/2")
        ax[1][1].plot(r, Omega-(epic_freq/2), label="Ω - κ/2")
        ax[1][1].plot(r, Omega-(epic_freq/4), label="Ω - κ/4")
        ax[1][1].plot(r, 15*np.ones(r.size))
        ax[1][1].set_xlim(0,20)
        ax[1][1].set_ylim(0,80)
        ax[1][1].set_title("Angular velocity regions")
        ax[1][1].set_xlabel("r")
        ax[1][1].set_ylabel("Ω(r)")
        ax[1][1].grid(True)
        ax[1][1].legend()
        plt.show()

    else:
        # Spiral potential plotting
        fig, ax = plt.subplots()
        R, F = np.meshgrid(r, phi)
        #X = np.linspace(-20, 20, 1000)
        #Y = np.linspace(-20, 20, 1000)
        #x, y = np.meshgrid(X,Y)
        #laplacian = ptnsm.c_second_total_potential_dr_derivative(x,y) + (1/np.sqrt(x**2 + y**2))*ptnsm.c_total_potential_derivative(x,y)
        #c_laplacian = ptnsm.c_second_spiral_potential_dr_derivative(x,y) + (1/np.sqrt(x**2 + y**2))*ptnsm.c_spiral_potential_dr_derivative(x,y) + (1/(x**2 + y**2))*ptnsm.c_second_spiral_potential_dphi_derivative(x,y)
        laplacian = ptnsm.second_spiral_potential_dr_derivative(R, F) + (1/R)*ptnsm.spiral_potential_dr_derivative(R,F) + (1/(R**2))*ptnsm.second_spiral_potential_dphi_derivative(R,F)
        #laplacian = laplacian1 + laplacian2
        #c_density = c_laplacian/(4*np.pi*ptnsm.G)
        density = laplacian/(4*np.pi*ptnsm.G)
        density_grad = np.gradient(density)
        # Calculating the minima in the radial direction.
        # That way we will get the minima that follows the peak of the spiral.
        # I need to create a mask that selects only the points that the radial gradient comes close to zero.
        # Then I use that to select the appropriate fs and then I plot the result. FUCK ME!
        #X = R* np.cos(F)
        #Y = R* np.sin(F)
        #mask_X = mask_R*np.cos(mask_F)
        #mask_Y = mask_R*np.sin(mask_F)
        #ax.pcolormesh(X,Y,density, cmap="inferno")
        #ax.scatter(mask_X, mask_Y)
        #ax.set_title("Isodensity color map")
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        #plt.show()

    return 0

r = np.linspace(0.01, 25, 10000)
phi = np.linspace(0, 2*np.pi, 1000)
#potentials(r, phi)
preprocess(r, phi, what="bruh")
