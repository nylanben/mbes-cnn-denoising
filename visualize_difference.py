import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from pathlib import Path

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
VIZ_DIR  = Path("path_to_visualizations")
DATASET_ID = "A_xyz_status_pos_down"
SAVE_DIR = Path("path_to_output")

# Load colored point cloud
print("Loading point cloud...")
pcd = np.loadtxt(VIZ_DIR / f"pointcloud_colored_{DATASET_ID}.txt")

x         = pcd[:, 0]
y         = pcd[:, 1]
z         = pcd[:, 2]
gt_label  = pcd[:, 3].astype(int)
cnn_label = pcd[:, 4].astype(int)

print(f"Loaded {len(pcd):,} points")

# ------------------------------------------------------------
# Build full rasters first to find largest differences
# ------------------------------------------------------------
CELL_SIZE = 1.0
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
x_bins = int(np.ceil((x_max - x_min) / CELL_SIZE))
y_bins = int(np.ceil((y_max - y_min) / CELL_SIZE))

def rasterize_full(x, y, z):
    raster, _, _, _ = binned_statistic_2d(
        x, y, z, statistic="min",
        bins=[x_bins, y_bins],
        range=[[x_min, x_max], [y_min, y_max]]
    )
    return raster

print("Building full rasters...")
raster_raw = rasterize_full(x, y, z)
raster_gt  = rasterize_full(x[gt_label == 1],  y[gt_label == 1],  z[gt_label == 1])
raster_cnn = rasterize_full(x[cnn_label == 1], y[cnn_label == 1], z[cnn_label == 1])

diff_map = raster_cnn - raster_gt

# ------------------------------------------------------------
# Find location of largest differences
# ------------------------------------------------------------
# Flatten and sort by absolute difference
valid_mask = ~np.isnan(diff_map)
diff_flat  = np.abs(diff_map.copy())
diff_flat[~valid_mask] = 0

# Get top N largest difference locations
N = 5
top_indices = np.argsort(diff_flat.ravel())[-N:][::-1]
top_coords  = np.unravel_index(top_indices, diff_map.shape)

print(f"\nTop {N} largest difference locations:")
for i in range(N):
    xi, yi = top_coords[0][i], top_coords[1][i]
    world_x = x_min + xi * CELL_SIZE
    world_y = y_min + yi * CELL_SIZE
    diff_val = diff_map[xi, yi]
    print(f"  Location {i+1}: X={world_x:.1f}, Y={world_y:.1f}, diff={diff_val:.2f}m")

# ------------------------------------------------------------
# Visualize zoomed area around largest difference
# ------------------------------------------------------------
ZOOM_SIZE = 64  # meters

def visualize_zoomed_area(center_x, center_y, zoom_size=ZOOM_SIZE,
                           title="Zoomed area", save_path=None):
    X_MIN = center_x - zoom_size / 2
    X_MAX = center_x + zoom_size / 2
    Y_MIN = center_y - zoom_size / 2
    Y_MAX = center_y + zoom_size / 2

    mask = (x >= X_MIN) & (x <= X_MAX) & (y >= Y_MIN) & (y <= Y_MAX)
    xm = x[mask]; ym = y[mask]; zm = z[mask]
    gt  = gt_label[mask]
    cnn = cnn_label[mask]

    if len(xm) == 0:
        print(f"No points in area around ({center_x:.1f}, {center_y:.1f})")
        return

    x_bins_z = int(np.ceil(zoom_size / CELL_SIZE))
    y_bins_z = int(np.ceil(zoom_size / CELL_SIZE))

    def rasterize_zoom(x, y, z):
        if len(x) == 0:
            return np.full((x_bins_z, y_bins_z), np.nan)
        raster, _, _, _ = binned_statistic_2d(
            x, y, z, statistic="min",
            bins=[x_bins_z, y_bins_z],
            range=[[X_MIN, X_MAX], [Y_MIN, Y_MAX]]
        )
        return raster

    rz_raw = rasterize_zoom(xm, ym, zm)
    rz_gt  = rasterize_zoom(xm[gt == 1],  ym[gt == 1],  zm[gt == 1])
    rz_cnn = rasterize_zoom(xm[cnn == 1], ym[cnn == 1], zm[cnn == 1])

    diff_z   = rz_cnn - rz_gt
    diff_abs = np.nanmax(np.abs(diff_z)) if not np.all(np.isnan(diff_z)) else 1.0
    z_min    = np.nanmin([rz_raw, rz_gt, rz_cnn])
    z_max    = np.nanmax([rz_raw, rz_gt, rz_cnn])

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(title, fontsize=20)

    for ax, raster, t in zip(
        axes[:3],
        [rz_raw, rz_gt, rz_cnn],
        ["Raw", "Ground Truth", "CNN Predicted"]
    ):
        im = ax.imshow(raster.T, origin="lower", cmap="viridis_r",
                       vmin=z_min, vmax=z_max,
                       extent=[X_MIN - center_x, X_MAX - center_x,
                               Y_MIN - center_y, Y_MAX - center_y])
        ax.set_title(t, fontsize=18)
        ax.set_xlabel("X offset (m)", fontsize=14)
        ax.set_ylabel("Y offset (m)", fontsize=14)
        cb = plt.colorbar(im, ax=ax, label="Depth (m)", pad=0.02)
        cb.ax.tick_params(labelsize=14)
        cb.set_label("Depth (m)", fontsize=16)
        cb.ax.invert_yaxis()

    im3 = axes[3].imshow(diff_z.T, origin="lower", cmap="RdBu",
                          vmin=-diff_abs, vmax=diff_abs,
                          extent=[X_MIN - center_x, X_MAX - center_x,
                                  Y_MIN - center_y, Y_MAX - center_y])
    axes[3].set_title("Difference (CNN - GT)", fontsize=18)
    axes[3].set_xlabel("X offset (m)", fontsize=14)
    axes[3].set_ylabel("Y offset (m)", fontsize=14)
    cb3 = plt.colorbar(im3, ax=axes[3], label="Diff (m)", pad=0.02)
    cb3.ax.tick_params(labelsize=14)
    cb3.set_label("Diff (m)", fontsize=16)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

# ------------------------------------------------------------
# Visualize top 3 largest difference areas
# ------------------------------------------------------------
for i in range(min(3, N)):
    xi, yi = top_coords[0][i], top_coords[1][i]
    cx = x_min + xi * CELL_SIZE
    cy = y_min + yi * CELL_SIZE
    diff_val = diff_map[xi, yi]
    print(f"\nVisualizing area {i+1}: diff={diff_val:.2f}m")

    visualize_zoomed_area(
        cx, cy,
        title=f"A_xyz (k3) — Largest difference area {i+1} — diff={diff_val:.2f}m",
        save_path=SAVE_DIR / f"zoom_diff_area_k3_{i+1}.png"
    )