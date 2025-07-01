"""
Module containing testing code for main program.
"""

# Importing necessary modules
import potentials as ptns
import numpy as np
import matplotlib.pyplot as plt

#=====================================================
# Tests
#=====================================================
def plot_axisymmetric_potentials(r):
    """
    Function performing tests on the potential functions.
    """
    print(ptns.disk_potential(6))
    print(ptns.bulge_potential(6))
    print(ptns.halo_potential(6))
    print(ptns.spiral_potential(6,0.6))
    #=======================================================
    # Potential plots
    #=======================================================
    
    # Prepare the potentials for plotting.
    # Unless the problem with the halo potential is fixed
    # we compartmentalized the plotting of our function.
    disk = ptns.disk_potential(r)
    bulge = ptns.bulge_potential(r)
    halo = ptns.halo_potential(r)
    total1 = disk[0:1200] + bulge[0:1200]
    total2 = disk[1200:] + bulge[1200:] + halo[1200:]
    disk_norm1 = disk[0:1200]/total1
    disk_norm2 = disk[1200:]/total2
    bulge_norm1 = bulge[0:1200]/total1
    bulge_norm2 = bulge[1200:]/total2
    halo_norm = halo[1200:]/total2

    # Make the plots
    fig, ax = plt.subplots(1,2)
    ax[0].plot(r, disk, label="Disk")
    ax[0].plot(r, bulge, label="Bulge")
    ax[0].plot(r[1200:], halo[1200:], label="Halo")
    ax[0].grid(True)
    ax[0].legend()
    ax[1].plot(r[0:1200], bulge_norm1, label="Bulge 1")
    ax[1].plot(r[1200:], bulge_norm2, label="Bulge 2")
    ax[1].plot(r[0:1200], disk_norm1, label="Disk 1")
    ax[1].plot(r[1200:], disk_norm2, label="Disk 2")
    ax[1].plot(r[1200:], halo_norm, label="Halo 1")
    ax[1].set_xscale("log")
    ax[1].grid("True")
    ax[1].legend()
    plt.show()

def f_strength():
    """
    This function plots the F-Strength of the potentials.
    """
    pass

r = np.linspace(1, 100, 10000)
plot_axisymmetric_potentials(r)

