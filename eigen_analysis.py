import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# =============================================================================
# BUILDING SPECIFICATIONS
# 25m x 25m, 10-storey, Kenya standard (BS 8110 / KS EAS 18)
# =============================================================================

# --- Geometry ---
n_storeys   = 10
storey_h    = 3.0           # m (standard Kenya storey height)
n_cols      = 36            # 6x6 grid on 5m spacing (0,5,10,15,20,25m each direction)

# --- Material: C25 Concrete ---
fck         = 25e6          # Pa
E           = 9500 * (fck / 1e6 + 8) ** (1/3) * 1e6  # Eurocode EC2 formula (Pa) ≈ 30.5 GPa

# --- Column section: 200mm x 700mm ---
# Lateral direction assumed along 200mm dimension (weak axis bending)
b_col       = 0.200         # m (lateral direction)
d_col       = 0.700         # m (perpendicular)
I_col       = (b_col * d_col**3) / 12  # strong axis: bending in plane of 700mm

# --- Storey lateral stiffness ---
# Ground storey: fixed-pinned (pinned base) → k = 3EI/h³
# Upper storeys: fixed-fixed (both ends restrained by beams) → k = 12EI/h³
k_ground    = n_cols * (3  * E * I_col / storey_h**3)   # N/m  (storey 1)
k_upper     = n_cols * (12 * E * I_col / storey_h**3)   # N/m  (storeys 2–10)

# --- Floor mass ---
# Slab: 150mm thick, 25x25m plan, concrete density 2500 kg/m³
slab_mass   = 0.150 * 25 * 25 * 2500   # kg

# Superimposed dead load: 1.5 kN/m² (finishes + services)
SDL_mass    = (1.5e3 / 9.81) * 25 * 25  # kg

# Live load (seismic combination, 30% of LL per EC8): 2.0 kN/m²
LL_mass     = 0.30 * (2.0e3 / 9.81) * 25 * 25  # kg

# Beam self-weight: 500x200mm beams, 5 bays x 2 directions x 2 lines x 5m span
beam_vol    = 0.500 * 0.200 * 5.0      # m³ per beam
n_beams     = 5 * 2 * 2                # 5 bays, 2 directions, 2 lines each
beam_mass   = n_beams * beam_vol * 2500 # kg

# Total dead load mass per floor (slab + SDL + beams)
total_DL_mass = slab_mass + SDL_mass + beam_mass  # kg

# Total floor mass (slab + SDL + LL + beams)
floor_mass  = total_DL_mass + LL_mass  # kg per floor

print("=" * 55)
print("BUILDING PARAMETERS")
print("=" * 55)
print(f"  Slab mass             : {slab_mass:.2f} kg")
print(f"  Beam mass             : {beam_mass:.2f} kg")
print(f"  Superimposed DL       : {SDL_mass:.2f} kg")
print(f"  Live load (30% of LL) : {LL_mass:.2f} kg")
print(f"  Total dead load mass  : {total_DL_mass:.2f} kg")
print(f"  Total floor mass      : {floor_mass:.2f} kg")
print(f"  Concrete E modulus    : {E/1e9:.2f} GPa")
print(f"  Column I (strong axis): {I_col*1e6:.0f} cm⁴  ({I_col:.6f} m⁴)")
print(f"  Ground storey k       : {k_ground/1e6:.3f} MN/m  (3EI/h³, pinned base)")
print(f"  Upper storey k        : {k_upper/1e6:.3f} MN/m  (12EI/h³, fixed-fixed)")
print(f"  Floor mass            : {floor_mass/1e3:.2f} tonnes")
print()

# =============================================================================
# 1. MASS AND STIFFNESS MATRICES (10 DOF shear frame)
# =============================================================================

M = np.diag([floor_mass] * n_storeys)

# Assign correct stiffness per storey:
# k[0] = ground storey (pinned base), k[1..9] = upper storeys (fixed-fixed)
k_storeys = [k_ground] + [k_upper] * (n_storeys - 1)

K = np.zeros((n_storeys, n_storeys))
for i in range(n_storeys):
    k = k_storeys[i]
    if i < n_storeys - 1:
        K[i,   i  ] += k
        K[i+1, i+1] += k
        K[i,   i+1] -= k
        K[i+1, i  ] -= k
# Add ground storey stiffness to first DOF (contribution from below floor 1)
K[0, 0] += k_storeys[0]

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
fig.suptitle("Mode Shapes — 10-Storey Shear Frame\n(25m×25m, C25, 200×700mm columns, Pinned Base)",
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
plt.savefig("mode_shapes3.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nMode shape plot saved to mode_shapes.png")
