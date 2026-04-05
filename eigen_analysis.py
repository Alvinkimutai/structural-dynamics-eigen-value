import numpy as np
from scipy import linalg

# 1. Define simplified mass and stiffness matrices (e.g., for a 3-storey building)
# Mass of each floor (kg)
m1, m2, m3 = 10000, 10000, 10000 
# Stiffness of each storey (N/m)
k1, k2, k3 = 5e6, 4e6, 3e6

# Mass matrix (diagonal)
M = np.diag([m1, m2, m3])

# Stiffness matrix for a shear frame
K = np.array([
    [k1 + k2, -k2, 0],
    [-k2, k2 + k3, -k3],
    [0, -k3, k3]
])

# 2. Solve the generalized eigenvalue problem
# Using eigh because K and M are symmetric
eigenvalues, eigenvectors = linalg.eigh(K, M)

# 3. Extract Natural Frequencies
omega = np.sqrt(eigenvalues) # circular frequencies in rad/s
frequencies_hz = omega / (2 * np.pi) # frequencies in Hz

print("Natural Frequencies (Hz):", frequencies_hz)
print("Mode Shapes (Eigenvectors):\n", eigenvectors)