import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# =============================================================================
# BUILDING GEOMETRY — 25m x 25m, 10-storey, 5m bay spacing (6x6 column grid)
# =============================================================================

n_storeys   = 10
storey_h    = 3.0           # m
bay_spacing = 5.0           # m
n_bays      = 5             # 5 bays x 5m = 25m each direction
col_w       = 0.20          # m  (200mm column width, to scale)
beam_h      = 0.45          # m  (450mm beam depth, to scale)

bay_coords   = np.arange(0, (n_bays + 1) * bay_spacing, bay_spacing)  # [0,5,10,15,20,25]
floor_coords = np.arange(0, (n_storeys + 1) * storey_h, storey_h)     # [0,3,6,...,30]

# =============================================================================
# FIGURE 1 — 2D ELEVATION (front frame, one grid line)
# =============================================================================

fig1, ax = plt.subplots(figsize=(16, 12))
ax.set_aspect('equal')
ax.set_facecolor('#f7f7f7')
fig1.patch.set_facecolor('#ffffff')

# --- Columns (upper storeys: fixed-fixed, darker shade) ---
for x in bay_coords:
    for j in range(n_storeys):
        y0 = floor_coords[j]
        y1 = floor_coords[j + 1]
        # Ground storey slightly lighter to visually distinguish pinned base
        fc = '#c8d0d4' if j == 0 else '#bdc3c7'
        rect = mpatches.FancyBboxPatch(
            (x - col_w / 2, y0), col_w, y1 - y0,
            boxstyle="square,pad=0",
            linewidth=0.8, edgecolor='#2c3e50', facecolor=fc
        )
        ax.add_patch(rect)

# --- Beams ---
for y in floor_coords[1:]:
    for i in range(n_bays):
        x0 = bay_coords[i] + col_w / 2
        x1 = bay_coords[i + 1] - col_w / 2
        rect = mpatches.FancyBboxPatch(
            (x0, y - beam_h), x1 - x0, beam_h,
            boxstyle="square,pad=0",
            linewidth=0.8, edgecolor='#2c3e50', facecolor='#85929e'
        )
        ax.add_patch(rect)

# --- Ground line ---
ax.hlines(0, -0.8, bay_coords[-1] + 0.8, colors='#2c3e50', linewidths=1.8)

# --- Ground hatch (soil) ---
ax.fill_between([-0.8, bay_coords[-1] + 0.8], [-0.5, -0.5], [0, 0],
                color='#d5d8dc', zorder=0)
for xh in np.linspace(-0.5, bay_coords[-1] + 0.5, 30):
    ax.plot([xh, xh - 0.25], [0, -0.35], color='#95a5a6', linewidth=0.6)

# --- Pinned base symbols (triangle) at each column base ---
for x in bay_coords:
    pin_size = 0.18
    triangle = plt.Polygon(
        [[x, 0], [x - pin_size, -pin_size * 1.2], [x + pin_size, -pin_size * 1.2]],
        closed=True, facecolor='#ffffff', edgecolor='#2c3e50', linewidth=1.2, zorder=5
    )
    ax.add_patch(triangle)
    # Small circle at pin point
    circle = Circle((x, 0), radius=0.05, facecolor='#2c3e50', zorder=6)
    ax.add_patch(circle)
    # Roller line below triangle
    ax.hlines(-pin_size * 1.2, x - pin_size, x + pin_size,
              colors='#2c3e50', linewidths=1.2, zorder=5)

# --- Floor level labels ---
for j, y in enumerate(floor_coords):
    label = "GF" if j == 0 else f"F{j}"
    ax.text(-1.1, y, label, va='center', ha='right', fontsize=8, color='#2c3e50')
    ax.axhline(y, color='#d5d8dc', linewidth=0.4, linestyle='--', zorder=0)

# --- Column grid labels ---
for i, x in enumerate(bay_coords):
    ax.text(x, -1.1, f"C{i+1}", ha='center', fontsize=8,
            color='#2c3e50', fontweight='bold')

# --- Bay width dimension arrows ---
for i in range(n_bays):
    x_mid = (bay_coords[i] + bay_coords[i + 1]) / 2
    ax.annotate('', xy=(bay_coords[i + 1], -1.8), xytext=(bay_coords[i], -1.8),
                arrowprops=dict(arrowstyle='<->', color='#2c3e50', lw=1.0))
    ax.text(x_mid, -2.1, f"{bay_spacing:.0f} m", ha='center', fontsize=8, color='#2c3e50')

# --- Total width dimension ---
ax.annotate('', xy=(bay_coords[-1], -2.7), xytext=(bay_coords[0], -2.7),
            arrowprops=dict(arrowstyle='<->', color='#7f8c8d', lw=1.0))
ax.text(bay_coords[-1] / 2, -3.0, f"Total = {bay_coords[-1]:.0f} m",
        ha='center', fontsize=9, color='#7f8c8d', fontstyle='italic')

# --- Storey height dimension ---
ax.annotate('', xy=(bay_coords[-1] + 1.2, storey_h), xytext=(bay_coords[-1] + 1.2, 0),
            arrowprops=dict(arrowstyle='<->', color='#c0392b', lw=1.0))
ax.text(bay_coords[-1] + 1.5, storey_h / 2, f"{storey_h:.1f} m",
        va='center', fontsize=8, color='#c0392b')

# --- Total height dimension ---
ax.annotate('', xy=(bay_coords[-1] + 2.2, floor_coords[-1]),
            xytext=(bay_coords[-1] + 2.2, floor_coords[0]),
            arrowprops=dict(arrowstyle='<->', color='#922b21', lw=1.0))
ax.text(bay_coords[-1] + 2.5, floor_coords[-1] / 2,
        f"{floor_coords[-1]:.0f} m", va='center', fontsize=8, color='#922b21')

# --- Legend annotation ---
ax.text(bay_coords[-1] / 2, floor_coords[-1] + 0.6,
        "▲ Pinned base  |  Fixed-fixed upper storeys  |  C25 concrete",
        ha='center', fontsize=8, color='#555555', style='italic')

ax.set_xlim(-1.8, bay_coords[-1] + 3.2)
ax.set_ylim(-3.5, floor_coords[-1] + 1.2)
ax.axis('off')
ax.set_title(
    "2D Elevation — 10-Storey RC Frame\n"
    "25m × 25m plan  |  C25 concrete  |  200×700mm columns  |  450×200mm beams  |  Pinned base",
    fontsize=12, pad=14
)

plt.tight_layout()
plt.savefig("frame_elevation_25m.png", dpi=150, bbox_inches='tight')
print("Saved: frame_elevation_25m.png")

# =============================================================================
# FIGURE 2 — 3D WIREFRAME
# =============================================================================

fig2 = plt.figure(figsize=(14, 11))
ax3  = fig2.add_subplot(111, projection='3d')

col_lines        = []
beam_lines       = []
ground_col_lines = []   # ground storey columns drawn separately (pinned base)

# --- 3D columns ---
for x in bay_coords:
    for z in bay_coords:
        for j in range(n_storeys):
            y0 = floor_coords[j]
            y1 = floor_coords[j + 1]
            if j == 0:
                ground_col_lines.append([(x, y0, z), (x, y1, z)])
            else:
                col_lines.append([(x, y0, z), (x, y1, z)])

# --- 3D beams (X and Z directions) ---
for y in floor_coords[1:]:
    for z in bay_coords:
        for i in range(n_bays):
            beam_lines.append([(bay_coords[i], y, z), (bay_coords[i+1], y, z)])
    for x in bay_coords:
        for i in range(n_bays):
            beam_lines.append([(x, y, bay_coords[i]), (x, y, bay_coords[i+1])])

# Upper storey columns — dark blue
col_coll = Line3DCollection(col_lines, colors='#2c3e50', linewidths=1.4, zorder=3,
                             label='Columns (fixed-fixed)')
# Ground storey columns — distinct colour to show pinned base
gcol_coll = Line3DCollection(ground_col_lines, colors='#1a6b9a', linewidths=1.8,
                              linestyles='solid', zorder=4, label='Ground storey (pinned base)')
beam_coll = Line3DCollection(beam_lines, colors='#c0392b', linewidths=0.8, zorder=2,
                              label='Beams')

ax3.add_collection3d(col_coll)
ax3.add_collection3d(gcol_coll)
ax3.add_collection3d(beam_coll)

# --- Pin symbols at column bases (small triangles projected in 3D) ---
for x in bay_coords:
    for z in bay_coords:
        ax3.scatter(x, 0, z, marker='^', s=30, color='#1a6b9a', zorder=5)

# --- Ground plane ---
xx, zz = np.meshgrid(bay_coords, bay_coords)
yy     = np.zeros_like(xx)
ax3.plot_surface(xx, yy, zz, alpha=0.12, color='#aab7b8', zorder=0)

# --- Axis formatting ---
ax3.set_xlabel("X (m)", labelpad=10)
ax3.set_ylabel("Height (m)", labelpad=10)
ax3.set_zlabel("Z (m)", labelpad=10)
ax3.set_xticks(bay_coords)
ax3.set_zticks(bay_coords)
ax3.set_yticks(floor_coords[::2])
ax3.set_title(
    "3D Frame — 10-Storey RC Building\n"
    "25m × 25m plan  |  6×6 column grid  |  3m storeys  |  Pinned base",
    fontsize=12, pad=16
)

col_patch  = mpatches.Patch(color='#2c3e50', label='Columns (fixed-fixed, upper)')
gcol_patch = mpatches.Patch(color='#1a6b9a', label='Columns (pinned base, ground)')
beam_patch = mpatches.Patch(color='#c0392b', label='Beams')
ax3.legend(handles=[col_patch, gcol_patch, beam_patch], loc='upper left', fontsize=9)

ax3.view_init(elev=22, azim=-50)
ax3.set_box_aspect([1, 2, 1])

plt.tight_layout()
plt.savefig("frame_3d_25m.png", dpi=150, bbox_inches='tight')
print("Saved: frame_3d_25m.png")

plt.show()
