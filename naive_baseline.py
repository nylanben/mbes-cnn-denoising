import numpy as np
from pathlib import Path
from scipy.stats import binned_statistic_2d

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------
DATA_DIR  = Path("path_to_visualizations")
OUTPUT_DIR = Path("path_to_visualizations")

# Load colored point cloud — already has all soundings with labels
# Use A_xyz since it has more meaningful noise level
pcd = np.loadtxt(DATA_DIR / "pointcloud_colored_A_xyz_status_pos_down.txt")

x          = pcd[:, 0]
y          = pcd[:, 1]
z          = pcd[:, 2]
gt_label   = pcd[:, 3].astype(int)

print(f"Loaded {len(pcd):,} points")

# ------------------------------------------------------------
# Rasterize
# ------------------------------------------------------------
CELL_SIZE = 1.0

x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

x_bins = int(np.ceil((x_max - x_min) / CELL_SIZE))
y_bins = int(np.ceil((y_max - y_min) / CELL_SIZE))

def rasterize(x, y, z):
    raster, _, _, _ = binned_statistic_2d(
        x, y, z, statistic="min",
        bins=[x_bins, y_bins],
        range=[[x_min, x_max], [y_min, y_max]]
    )
    return raster

# Naive — accept all soundings
print("Rasterizing naive (all soundings)...")
raster_naive = rasterize(x, y, z)

# Ground truth — only accepted soundings
print("Rasterizing ground truth...")
gt_mask = gt_label == 1
raster_gt = rasterize(x[gt_mask], y[gt_mask], z[gt_mask])

# ------------------------------------------------------------
# Compare
# ------------------------------------------------------------
valid_both = ~np.isnan(raster_naive) & ~np.isnan(raster_gt)
diff = raster_naive[valid_both] - raster_gt[valid_both]

rmse = np.sqrt(np.mean(diff**2))
mae  = np.mean(np.abs(diff))
bias = np.mean(diff)

print(f"\nNaive baseline (accept all) vs Ground Truth:")
print(f"Valid cells : {valid_both.sum():,}")
print(f"RMSE        : {rmse:.4f} m")
print(f"MAE         : {mae:.4f} m")
print(f"Bias        : {bias:.4f} m")
print(f"Max diff    : {diff.max():.4f} m")
print(f"Min diff    : {diff.min():.4f} m")

# IHO TVU compliance
def iho_tvu(depth, a, b):
    return np.sqrt(a**2 + (b * np.abs(depth))**2)

orders = {
    "Exclusive": (0.15, 0.0075),
    "Special":   (0.25, 0.0075),
    "Order 1a":  (0.50, 0.013),
    "Order 1b":  (0.50, 0.013),
    "Order 2":   (1.00, 0.023),
}

gt_valid_depths = raster_gt[valid_both]
print(f"\nIHO TVU compliance:")
for order, (a, b) in orders.items():
    tvu_per_cell = iho_tvu(gt_valid_depths, a, b)
    within = np.sum(np.abs(diff) <= tvu_per_cell)
    pct = 100 * within / len(diff)
    print(f"{order:12s}: {pct:.1f}%")