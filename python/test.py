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
def test_potentials():
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

    r = np.linspace(1, 100, 10000)
    disk = ptns.disk_potential(r)
    bulge = ptns.bulge_potential(r)
    halo = ptns.halo_potential(r)
    total = disk + bulge + halo
    disk_norm = disk/total
    bulge_norm = bulge/total
    halo_norm = halo/total

    # Make the plots
    fig, ax = plt.subplots(1,2)
    ax[0].plot(r, disk, label="Disk")
    ax[0].plot(r, bulge, label="Bulge")
    ax[0].plot(r[1200:], halo[1200:], label="Halo")
    ax[0].grid(True)
    ax[0].legend()
    ax[1].plot(r, bulge_norm, label="Bulge")
    ax[1].plot(r, disk_norm, label="Disk")
    ax[1].plot(r, halo_norm, label="Halo")
    ax[1].set_xscale("log")
    ax[1].grid("True")
    ax[1].legend()
    plt.show()

test_potentials()

