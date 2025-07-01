"""
Testing of the symbolic potentials
"""

import potentials_sym as ptnsm
import matplotlib.pyplot as plt
import numpy as np
import sympy as smp

phip = np.linspace(0, 2*np.pi, 1000)
r, phi = smp.symbols("r phi", real=True)
f = smp.lambdify((r,phi), ptnsm.V_sp, 'numpy')
fig, ax = plt.subplots()
ax.plot(phip, f(7.5,phip))
ax.grid(True)
plt.show()