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
    ax[0].set_xlim(0,20)
    ax[1].plot(phi, ptnsm.spiral_potential(7, phi), label="Spiral Potential")
    ax[1].legend()
    ax[1].grid(True)
    ax[1].set_xlim(0, 2*np.pi)
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
        ax[0][0].set_xlim(0,20)

        # Rotation curve
        U_rot = np.sqrt(r*ptnsm.total_potential_derivative(r))
        ax[0][1].plot(r, U_rot)
        ax[0][1].set_title("Rotation Curve")
        ax[0][1].set_xlabel("r")
        ax[0][1].set_ylabel("$U_{rot}$")
        ax[0][1].set_xlim(0,20)
        ax[0][1].grid(True)

        # Spiral potential plotting
        #????????????????????????????????????????????????????????
        # Can be significantly improved. Right now it calculates the
        # ptnsn.second_spiral_potential_dr_derivative of every point,
        # while it can just do it on the ones that we want after we pick them.
        #????????????????????????????????????????????????????????
        R, F = np.meshgrid(r, phi)
        laplacian = ptnsm.second_spiral_potential_dr_derivative(R, F) + (1/R)*ptnsm.spiral_potential_dr_derivative(R,F) + (1/(R**2))*ptnsm.second_spiral_potential_dphi_derivative(R,F)
        density = laplacian/(4*np.pi*ptnsm.G)
        density_grad_r = np.gradient(density, axis = 1)
        mask1 = np.abs(density_grad_r) < 2.5
        mask2 = ptnsm.second_spiral_potential_dr_derivative(R, F) > 0
        mask3 = R < 10.0
        mask4 = R > 2.0
        mask = mask1*mask2*mask3*mask4
        mask_r = np.array(R[mask])
        mask_f = np.array(F[mask])
        mask_x = mask_r*np.cos(mask_f)
        mask_y = mask_r*np.sin(mask_f)
        x = R*np.cos(F)
        y = R*np.sin(F)
        mesh1 = ax[1][0].pcolormesh(x, y, density, cmap='magma')
        fig.colorbar(mesh1, ax=ax[1][0])
        ax[1][0].scatter(mask_x[~np.isnan(mask_x)], mask_y[~np.isnan(mask_y)], c="black", s=5)
        ax[1][0].set_title("Isodensity color map")
        ax[1][0].set_xlabel("x")
        ax[1][0].set_ylabel("y")
        ax[1][0].set_aspect("equal")

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
        #????????????????????????????????????????????????????????
        # Can be significantly improved. Right now it calculates the
        # ptnsn.second_spiral_potential_dr_derivative of every point,
        # while it can just do it on the ones that we want after we pick them.
        #????????????????????????????????????????????????????????
        R, F = np.meshgrid(r, phi)
        laplacian = ptnsm.second_spiral_potential_dr_derivative(R, F) + (1/R)*ptnsm.spiral_potential_dr_derivative(R,F) + (1/(R**2))*ptnsm.second_spiral_potential_dphi_derivative(R,F)
        density = laplacian/(4*np.pi*ptnsm.G)
        density_grad_r = np.gradient(density, axis = 1)
        mask1 = np.abs(density_grad_r) < 10
        mask2 = ptnsm.second_spiral_potential_dr_derivative(R, F) > 0
        mask3 = R < 10.0
        mask4 = R > 2.0
        mask = mask1*mask2*mask3*mask4
        mask_r = np.array(R[mask])
        mask_f = np.array(F[mask])
        mask_x = mask_r*np.cos(mask_f)
        mask_y = mask_r*np.sin(mask_f)
        x = R*np.cos(F)
        y = R*np.sin(F)
        mesh1 = ax[1][0].pcolormesh(x, y, density, cmap='magma')
        fig.colorbar(mesh1, ax=ax[1][0])
        ax[1][0].scatter(mask_x[~np.isnan(mask_x)], mask_y[~np.isnan(mask_y)], c="black", s=5)
        ax[1][0].set_title("Isodensity color map")
        ax[1][0].set_xlabel("x")
        ax[1][0].set_ylabel("y")
        ax[1][0].set_aspect("equal")
        plt.show()

    return 0

r = np.linspace(0.01, 20, 5000)
phi = np.linspace(0, 2*np.pi, 5000)
potentials(r, phi)
preprocess(r, phi)
