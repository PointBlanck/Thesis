"""
Module containing testing code for main program.
"""

# Importing necessary modules
import potentials as ptns

#=====================================================
# Potential Tests
#=====================================================
def test_potentials():
    """
    Function performing tests on the potential functions.
    """
    print(ptns.disk_potential(0))
    print(ptns.bulge_potential(0))

test_potentials()

