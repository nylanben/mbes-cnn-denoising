import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import binned_statistic_2d

from config import VOXEL_SIZE, GRID_SHAPE

# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------
DATA_DIR     = Path("path_to_realdata")
CNN_DATA_DIR = Path("path_to_CNN_ready_data")
PRED_DIR     = Path("path_to_predictions")
OUTPUT_DIR   = Path("path_to_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID      = "A_xyz_status_pos_down"
FILENAME_SUFFIX = "vox1_cub32x32x32_ch2"

INPUT_FILE = DATA_DIR / f"A_xyz_status_pos_down/A_xyz_status_pos_down.txt"

assert INPUT_FILE.exists(),  f"File not found: {INPUT_FILE}"

# ------------------------------------------------------------
# 2. Load raw data
# ------------------------------------------------------------
print("Loading raw ASCII file...")
df = pd.read_csv(
    INPUT_FILE,
    sep=r"\s+",
    header=None,
    usecols=[0, 1, 2, 3],
    engine="python"
)
df.columns = ["X", "Y", "Z", "Status"]
df["Status"] = df["Status"].astype(str).str.lower()
accepted_mask = (df["Status"].values == "accepted").astype(np.int32)
points = df[["X", "Y", "Z"]].values.astype(np.float32)

print(f"Loaded {len(points):,} points")

# ------------------------------------------------------------
# 3. Load preprocessed data
# ------------------------------------------------------------
print("Loading preprocessed data...")
cuboid_keys  = np.load(CNN_DATA_DIR / f"cuboid_keys_{DATASET_ID}_{FILENAME_SUFFIX}.npy")
pcd_center   = np.load(CNN_DATA_DIR / f"pcd_center_{DATASET_ID}.npy")
preds_all    = np.load(PRED_DIR / f"preds_{DATASET_ID}.npy")

print(f"Loaded {len(cuboid_keys)} cuboid keys")
print(f"Loaded predictions for {len(preds_all)} cuboids")

# ------------------------------------------------------------
# 4. Center points (same as preprocessing)
# ------------------------------------------------------------
points_centered = points - pcd_center

# ------------------------------------------------------------
# 5. Assign each point to its voxel and cuboid
# ------------------------------------------------------------
print("Assigning points to voxels...")

# Compute voxel indices for each point
voxel_idx = np.floor(points_centered / VOXEL_SIZE).astype(np.int32)

# Compute cuboid indices
voxels_per_cuboid = np.array(GRID_SHAPE, dtype=np.int32)
cuboid_idx_per_point = voxel_idx // voxels_per_cuboid
local_idx_per_point  = voxel_idx % voxels_per_cuboid

# Build lookup: cuboid key -> cuboid array index
cuboid_key_to_idx = {tuple(k): i for i, k in enumerate(cuboid_keys)}


# ------------------------------------------------------------
# 6. Assign CNN prediction to each point (vectorized)
# ------------------------------------------------------------
print("Assigning CNN predictions to points...")

# Build lookup array — cuboid key -> cuboid array index
# Convert cuboid_keys to a dictionary for fast lookup
cuboid_key_to_idx = {tuple(k): i for i, k in enumerate(cuboid_keys)}

# Compute cuboid and local indices for all points at once
cuboid_idx_per_point = voxel_idx // voxels_per_cuboid  # (N, 3)
local_idx_per_point  = voxel_idx % voxels_per_cuboid   # (N, 3)

# Initialize labels as -1 (not in any cuboid)
cnn_labels = np.full(len(points), -1, dtype=np.int8)

# Get unique cuboid keys present in the data
unique_cuboid_keys = np.unique(cuboid_idx_per_point, axis=0)
print(f"Unique cuboids in point cloud: {len(unique_cuboid_keys)}")

for ck in unique_cuboid_keys:
    key = tuple(ck)
    if key not in cuboid_key_to_idx:
        continue

    arr_idx = cuboid_key_to_idx[key]
    pred_volume = preds_all[arr_idx]  # (32, 32, 32)

    # Find all points in this cuboid
    point_mask = np.all(cuboid_idx_per_point == ck, axis=1)
    point_indices = np.where(point_mask)[0]

    # Get local indices for these points
    lx = local_idx_per_point[point_indices, 0]
    ly = local_idx_per_point[point_indices, 1]
    lz = local_idx_per_point[point_indices, 2]

    # Bounds check
    valid = ((lx >= 0) & (lx < GRID_SHAPE[0]) &
             (ly >= 0) & (ly < GRID_SHAPE[1]) &
             (lz >= 0) & (lz < GRID_SHAPE[2]))

    valid_indices = point_indices[valid]
    lx_v = lx[valid]
    ly_v = ly[valid]
    lz_v = lz[valid]

    # Assign CNN prediction
    cnn_labels[valid_indices] = pred_volume[lx_v, ly_v, lz_v].astype(np.int8)

print(f"Points assigned     : {(cnn_labels >= 0).sum():,}")
print(f"Points not in cuboid: {(cnn_labels < 0).sum():,}")

# ------------------------------------------------------------
# 7. Save colored point cloud
# ------------------------------------------------------------
print("Saving colored point cloud...")

# Save as txt with columns: X Y Z gt_label cnn_label
output_data = np.column_stack([
    points,
    accepted_mask,
    np.where(cnn_labels >= 0, cnn_labels, 0)
])

np.savetxt(
    OUTPUT_DIR / f"pointcloud_colored_{DATASET_ID}.txt",
    output_data,
    fmt="%.4f %.4f %.4f %d %d",
    header="X Y Z gt_label cnn_label"
)

print(f"Saved colored point cloud")

# ------------------------------------------------------------
# 8. High precision rasters using actual sounding depths
# ------------------------------------------------------------
print("Building high-precision rasters...")

CELL_SIZE = 1.0

# Use only points that are in a cuboid
valid_mask = cnn_labels >= 0

x_coords = points[valid_mask, 0]
y_coords = points[valid_mask, 1]
z_coords = points[valid_mask, 2]
gt_labels_valid  = accepted_mask[valid_mask]
cnn_labels_valid = cnn_labels[valid_mask]

x_min, x_max = x_coords.min(), x_coords.max()
y_min, y_max = y_coords.min(), y_coords.max()

x_bins = int(np.ceil((x_max - x_min) / CELL_SIZE))
y_bins = int(np.ceil((y_max - y_min) / CELL_SIZE))

def rasterize(x, y, z):
    raster, _, _, _ = binned_statistic_2d(
        x, y, z,
        statistic="min",
        bins=[x_bins, y_bins],
        range=[[x_min, x_max], [y_min, y_max]]
    )
    return raster

print("Rasterizing raw points...")
raster_raw = rasterize(x_coords, y_coords, z_coords)

print("Rasterizing ground truth accepted points...")
gt_mask_bool = gt_labels_valid == 1
if gt_mask_bool.sum() > 0:
    raster_gt = rasterize(x_coords[gt_mask_bool],
                          y_coords[gt_mask_bool],
                          z_coords[gt_mask_bool])
else:
    raster_gt = np.full_like(raster_raw, np.nan)

print("Rasterizing CNN predicted accepted points...")
cnn_mask_bool = cnn_labels_valid == 1
if cnn_mask_bool.sum() > 0:
    raster_cnn = rasterize(x_coords[cnn_mask_bool],
                           y_coords[cnn_mask_bool],
                           z_coords[cnn_mask_bool])
else:
    raster_cnn = np.full_like(raster_raw, np.nan)

print(f"Raster shape: {raster_raw.shape}")
print(f"Raw coverage : {np.sum(~np.isnan(raster_raw)):,} cells")
print(f"GT coverage  : {np.sum(~np.isnan(raster_gt)):,} cells")
print(f"CNN coverage : {np.sum(~np.isnan(raster_cnn)):,} cells")

# ------------------------------------------------------------
# 9. Statistical comparison
# ------------------------------------------------------------
valid_both = ~np.isnan(raster_cnn) & ~np.isnan(raster_gt)
diff = raster_cnn[valid_both] - raster_gt[valid_both]

rmse = np.sqrt(np.mean(diff**2))
mae  = np.mean(np.abs(diff))
bias = np.mean(diff)
std  = np.std(diff)

print(f"\nHigh-precision raster comparison:")
print(f"Valid cells : {valid_both.sum():,}")
print(f"RMSE        : {rmse:.4f} m")
print(f"MAE         : {mae:.4f} m")
print(f"Bias        : {bias:.4f} m")
print(f"Std         : {std:.4f} m")
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
print(f"\nIHO TVU compliance (per-cell depth, 95% confidence):")
for order, (a, b) in orders.items():
    tvu_per_cell = iho_tvu(gt_valid_depths, a, b)
    within = np.sum(np.abs(diff) <= tvu_per_cell)
    pct = 100 * within / len(diff)
    print(f"{order:12s}: {pct:.1f}% of cells within tolerance")

# ------------------------------------------------------------
# 10. Visualization
# ------------------------------------------------------------
print("Saving visualizations...")

diff_map = raster_cnn - raster_gt
diff_abs = np.nanmax(np.abs(diff_map))

z_min = np.nanmin([raster_raw, raster_gt, raster_cnn])
z_max = np.nanmax([raster_raw, raster_gt, raster_cnn])

# Three-way comparison
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle(f"High-precision bathymetric comparison — {DATASET_ID}", fontsize=14)

for ax, raster, title in zip(
    axes,
    [raster_raw, raster_gt, raster_cnn],
    ["Raw (all soundings)", "Ground Truth (accepted)", "CNN Predicted"]
):
    im = ax.imshow(raster.T, origin="lower", cmap="viridis_r",
                   vmin=z_min, vmax=z_max)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(im, ax=ax, label="Depth (m)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"raster_hires_3way_{DATASET_ID}.png", dpi=150)
plt.close()

# Difference map
fig2, ax2 = plt.subplots(figsize=(10, 8))
im2 = ax2.imshow(diff_map.T, origin="lower", cmap="RdBu",
                  vmin=-diff_abs, vmax=diff_abs)
ax2.set_title(f"Difference (CNN - GT) — {DATASET_ID}")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
plt.colorbar(im2, ax=ax2, label="Difference (m)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"raster_hires_diff_{DATASET_ID}.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# 11. Save rasters
# ------------------------------------------------------------
np.save(OUTPUT_DIR / f"raster_hires_raw_{DATASET_ID}.npy",  raster_raw)
np.save(OUTPUT_DIR / f"raster_hires_gt_{DATASET_ID}.npy",   raster_gt)
np.save(OUTPUT_DIR / f"raster_hires_cnn_{DATASET_ID}.npy",  raster_cnn)
np.save(OUTPUT_DIR / f"raster_hires_diff_{DATASET_ID}.npy", diff_map)

print(f"Saved all outputs to {OUTPUT_DIR}")