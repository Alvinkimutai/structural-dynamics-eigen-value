import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# =============================================================================
# BUILDING SPECIFICATIONS
# 15m x 15m, 10-storey, Kenya standard (BS 8110 / KS EAS 18)
# =============================================================================

# --- Geometry ---
n_storeys   = 10
storey_h    = 3.0           # m (standard Kenya storey height)
n_cols      = 16            # 4x4 grid on 5m spacing (0,5,10,15m each direction)

# --- Material: C25 Concrete ---
fck         = 25e6          # Pa
E           = 9500 * (fck / 1e6 + 8) ** (1/3) * 1e6  # Eurocode EC2 formula (Pa) ≈ 30.5 GPa

# --- Column section: 200mm x 700mm ---
# Lateral direction assumed along 200mm dimension (weak axis bending)
b_col       = 0.200         # m (lateral direction)
d_col       = 0.700         # m (perpendicular)
I_col       = (b_col * d_col**3) / 12  # strong axis: bending in plane of 700mm

# --- Storey lateral stiffness: k = 12EI/h^3 per column (fixed-fixed) ---
k_col       = 12 * E * I_col / storey_h**3
k_storey    = n_cols * k_col            # N/m

# --- Floor mass ---
# Slab: 150mm thick, 15x15m plan, concrete density 2500 kg/m³
slab_mass   = 0.150 * 15 * 15 * 2500   # kg

# Superimposed dead load: 1.5 kN/m² (finishes + services)
SDL_mass    = (1.5e3 / 9.81) * 15 * 15  # kg

# Live load (seismic combination, 30% of LL per EC8): 2.5 kN/m²
LL_mass     = 0.30 * (2.5e3 / 9.81) * 15 * 15  # kg

# Beam self-weight: 450x200mm beams, 4 bays x 2 directions x 5m span
beam_vol    = 0.450 * 0.200 * 5.0      # m³ per beam
n_beams     = 4 * 2 * 2                # 4 bays, 2 directions, 2 lines each
beam_mass   = n_beams * beam_vol * 2500 # kg

floor_mass  = slab_mass + SDL_mass + LL_mass + beam_mass  # kg per floor

print("=" * 55)
print("BUILDING PARAMETERS")
print("=" * 55)
print(f"  Concrete E modulus    : {E/1e9:.2f} GPa")
print(f"  Column I (strong axis): {I_col*1e6:.0f} cm⁴  ({I_col:.6f} m⁴)")
print(f"  Storey stiffness k    : {k_storey/1e6:.3f} MN/m")
print(f"  Floor mass            : {floor_mass/1e3:.2f} tonnes")
print()

# =============================================================================
# 1. MASS AND STIFFNESS MATRICES (10 DOF shear frame)
# =============================================================================

M = np.diag([floor_mass] * n_storeys)

K = np.zeros((n_storeys, n_storeys))
for i in range(n_storeys):
    K[i, i] += k_storey                        # current storey above
    if i > 0:
        K[i, i]     += k_storey                # storey below
        K[i, i-1]   -= k_storey
        K[i-1, i]   -= k_storey

# Correct: top storey only has one storey stiffness contribution
# Rebuild properly using tridiagonal assembly
K = np.zeros((n_storeys, n_storeys))
for i in range(n_storeys):
    if i < n_storeys - 1:
        K[i, i]     += k_storey
        K[i+1, i+1] += k_storey
        K[i, i+1]   -= k_storey
        K[i+1, i]   -= k_storey
# Add ground storey stiffness to first DOF
K[0, 0] += k_storey

# =============================================================================
# 2. SOLVE GENERALISED EIGENVALUE PROBLEM: K*phi = omega^2 * M * phi
# =============================================================================

eigenvalues, eigenvectors = linalg.eigh(K, M)

# =============================================================================
# 3. NATURAL FREQUENCIES AND MODE SHAPES
# =============================================================================

omega_n = np.sqrt(eigenvalues)              # rad/s
freq_hz = omega_n / (2 * np.pi)            # Hz
periods = 1.0 / freq_hz                    # seconds

# Normalise mode shapes: unit displacement at roof
mode_shapes = eigenvectors / eigenvectors[-1, :]

print("=" * 55)
print("EIGENVALUE RESULTS")
print("=" * 55)
print(f"{'Mode':<6} {'ω (rad/s)':<14} {'f (Hz)':<12} {'T (s)':<10}")
print("-" * 55)
for i in range(n_storeys):
    print(f"  {i+1:<4} {omega_n[i]:<14.4f} {freq_hz[i]:<12.4f} {periods[i]:<10.4f}")

# =============================================================================
# 4. PLOT: First 4 Mode Shapes
# =============================================================================

floors = np.arange(0, n_storeys + 1)       # 0 = ground (fixed base)

fig, axes = plt.subplots(1, 4, figsize=(14, 7), sharey=True)
fig.suptitle("Mode Shapes — 10-Storey Shear Frame\n(15m×15m, C25, 200×700mm columns)",
             fontsize=13)

for idx in range(4):
    shape = np.concatenate([[0], mode_shapes[:, idx]])  # pin ground to 0
    axes[idx].plot(shape, floors, 'b-o', markersize=5)
    axes[idx].axvline(0, color='k', linewidth=0.8, linestyle='--')
    axes[idx].set_title(f"Mode {idx+1}\nT = {periods[idx]:.3f} s\nf = {freq_hz[idx]:.3f} Hz",
                        fontsize=10)
    axes[idx].set_xlabel("Relative Displacement")
    axes[idx].grid(True, alpha=0.3)

axes[0].set_ylabel("Floor Level")
axes[0].set_yticks(floors)
axes[0].set_yticklabels([f"F{i}" if i > 0 else "GF" for i in floors])

plt.tight_layout()
plt.savefig("mode_shapes2.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nMode shape plot saved to mode_shapes.png")
