import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# =============================================================================
# BUILDING GEOMETRY
# =============================================================================

n_storeys   = 10
storey_h    = 3.0           # m
bay_spacing = 5.0           # m
n_bays      = 3             # 3 bays x 5m = 15m each direction
col_w       = 0.20          # m (column width for patch drawing)
beam_h      = 0.45          # m (beam depth for patch drawing)

bay_coords  = np.arange(0, (n_bays + 1) * bay_spacing, bay_spacing)   # [0,5,10,15]
floor_coords = np.arange(0, (n_storeys + 1) * storey_h, storey_h)     # [0,3,...,30]

# =============================================================================
# FIGURE 1 — 2D ELEVATION (front frame, one grid line)
# =============================================================================

fig1, ax = plt.subplots(figsize=(10, 12))
ax.set_aspect('equal')
ax.set_facecolor('#f7f7f7')
fig1.patch.set_facecolor('#ffffff')

# --- Columns ---
for x in bay_coords:
    for j in range(n_storeys):
        y0 = floor_coords[j]
        y1 = floor_coords[j + 1]
        rect = mpatches.FancyBboxPatch(
            (x - col_w / 2, y0), col_w, y1 - y0,
            boxstyle="square,pad=0",
            linewidth=0.8, edgecolor='#2c3e50', facecolor='#bdc3c7'
        )
        ax.add_patch(rect)

# --- Beams ---
for y in floor_coords[1:]:                 # skip ground slab
    for i in range(n_bays):
        x0 = bay_coords[i] + col_w / 2
        x1 = bay_coords[i + 1] - col_w / 2
        rect = mpatches.FancyBboxPatch(
            (x0, y - beam_h), x1 - x0, beam_h,
            boxstyle="square,pad=0",
            linewidth=0.8, edgecolor='#2c3e50', facecolor='#85929e'
        )
        ax.add_patch(rect)

# --- Ground hatch ---
ax.fill_between([-1, bay_coords[-1] + 1], [-0.3, -0.3], [0, 0],
                color='#aab7b8', zorder=0)
ax.hlines(0, -0.5, bay_coords[-1] + 0.5, colors='#2c3e50', linewidths=1.5)
for xh in np.linspace(-0.5, bay_coords[-1] + 0.5, 20):
    ax.plot([xh, xh - 0.3], [0, -0.3], color='#7f8c8d', linewidth=0.7)

# --- Floor level labels ---
for j, y in enumerate(floor_coords):
    label = "GF" if j == 0 else f"F{j}"
    ax.text(-0.8, y, label, va='center', ha='right', fontsize=8, color='#2c3e50')
    ax.axhline(y, color='#d5d8dc', linewidth=0.4, linestyle='--', zorder=0)

# --- Column grid labels ---
for i, x in enumerate(bay_coords):
    ax.text(x, -0.9, f"C{i+1}", ha='center', fontsize=9,
            color='#2c3e50', fontweight='bold')

# --- Dimensions: bay widths ---
for i in range(n_bays):
    x_mid = (bay_coords[i] + bay_coords[i + 1]) / 2
    ax.annotate('', xy=(bay_coords[i + 1], -1.6), xytext=(bay_coords[i], -1.6),
                arrowprops=dict(arrowstyle='<->', color='#2c3e50', lw=1.0))
    ax.text(x_mid, -1.9, f"{bay_spacing:.0f} m", ha='center', fontsize=8, color='#2c3e50')

# --- Dimension: storey height ---
ax.annotate('', xy=(bay_coords[-1] + 1.2, storey_h), xytext=(bay_coords[-1] + 1.2, 0),
            arrowprops=dict(arrowstyle='<->', color='#c0392b', lw=1.0))
ax.text(bay_coords[-1] + 1.5, storey_h / 2, f"{storey_h:.1f} m",
        va='center', fontsize=8, color='#c0392b')

ax.set_xlim(-1.5, bay_coords[-1] + 2.5)
ax.set_ylim(-2.5, floor_coords[-1] + 1.0)
ax.axis('off')
ax.set_title("2D Elevation — 10-Storey RC Frame\n(15m × 15m plan, C25, 200×700mm columns, 450×200mm beams)",
             fontsize=12, pad=12)

plt.tight_layout()
plt.savefig("frame_elevation.png", dpi=150, bbox_inches='tight')
print("Saved: frame_elevation.png")

# =============================================================================
# FIGURE 2 — 3D WIREFRAME
# =============================================================================

fig2 = plt.figure(figsize=(12, 10))
ax3 = fig2.add_subplot(111, projection='3d')

col_lines   = []
beam_lines  = []

# --- 3D columns ---
for x in bay_coords:
    for z in bay_coords:
        for j in range(n_storeys):
            y0 = floor_coords[j]
            y1 = floor_coords[j + 1]
            col_lines.append([(x, y0, z), (x, y1, z)])

# --- 3D beams (X-direction and Z-direction) ---
for y in floor_coords[1:]:
    for z in bay_coords:
        for i in range(n_bays):
            beam_lines.append([(bay_coords[i], y, z), (bay_coords[i+1], y, z)])
    for x in bay_coords:
        for i in range(n_bays):
            beam_lines.append([(x, y, bay_coords[i]), (x, y, bay_coords[i+1])])

col_coll  = Line3DCollection(col_lines,  colors='#2c3e50', linewidths=1.4, zorder=3)
beam_coll = Line3DCollection(beam_lines, colors='#c0392b', linewidths=0.9, zorder=2)

ax3.add_collection3d(col_coll)
ax3.add_collection3d(beam_coll)

# --- Ground plane ---
xx, zz = np.meshgrid(bay_coords, bay_coords)
yy = np.zeros_like(xx)
ax3.plot_surface(xx, yy, zz, alpha=0.15, color='#aab7b8', zorder=0)

# --- Axis formatting ---
ax3.set_xlabel("X (m)", labelpad=8)
ax3.set_ylabel("Height (m)", labelpad=8)
ax3.set_zlabel("Z (m)", labelpad=8)
ax3.set_yticks(floor_coords[::2])
ax3.set_title("3D Frame — 10-Storey RC Building\n(15m×15m plan, 3m storeys)",
              fontsize=12, pad=14)

col_patch  = mpatches.Patch(color='#2c3e50', label='Columns')
beam_patch = mpatches.Patch(color='#c0392b', label='Beams')
ax3.legend(handles=[col_patch, beam_patch], loc='upper left', fontsize=9)

ax3.view_init(elev=20, azim=-55)
ax3.set_box_aspect([1, 2, 1])

plt.tight_layout()
plt.savefig("frame_3d.png", dpi=150, bbox_inches='tight')
print("Saved: frame_3d.png")

plt.show()
