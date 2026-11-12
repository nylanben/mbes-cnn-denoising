import numpy as np
from pathlib import Path

VIZ_DIR = Path("path_to_visualizations")
DATASET_ID = "A_xyz_status_pos_down"

print("Loading colored point cloud...")
pcd = np.loadtxt(VIZ_DIR / f"pointcloud_colored_{DATASET_ID}.txt")

x         = pcd[:, 0]
y         = pcd[:, 1]
z         = pcd[:, 2]
gt_label  = pcd[:, 3].astype(int)
cnn_label = pcd[:, 4].astype(int)

print(f"Loaded {len(pcd):,} points")

# Find noise soundings that are shallower than the shallowest accepted sounding
# in the same area (above seafloor noise)

# Get all noise soundings
noise_mask     = gt_label == 0
accepted_mask  = gt_label == 1

# For each noise sounding, check if it is shallower than
# the minimum accepted depth in the same 1m XY cell
CELL_SIZE = 1.0
x_min = x.min()
y_min = y.min()

# Bin all points to 1m cells
x_bin = ((x - x_min) / CELL_SIZE).astype(int)
y_bin = ((y - y_min) / CELL_SIZE).astype(int)

# Build lookup of minimum accepted depth per cell
print("Building min accepted depth per cell...")
from scipy.stats import binned_statistic_2d
x_bins = int(np.ceil((x.max() - x_min) / CELL_SIZE))
y_bins = int(np.ceil((y.max() - y_min) / CELL_SIZE))

min_accepted_depth, _, _, binnumber = binned_statistic_2d(
    x[accepted_mask], y[accepted_mask], z[accepted_mask],
    statistic='min',
    bins=[x_bins, y_bins],
    range=[[x_min, x.max()], [y_min, y.max()]]
)

# For each noise sounding, find its cell and compare to min accepted depth
print("Classifying noise soundings...")
noise_x = x[noise_mask]
noise_y = y[noise_mask]
noise_z = z[noise_mask]
noise_cnn = cnn_label[noise_mask]

# Get cell indices for noise soundings
noise_xi = np.clip(((noise_x - x_min) / CELL_SIZE).astype(int), 0, x_bins - 1)
noise_yi = np.clip(((noise_y - y_min) / CELL_SIZE).astype(int), 0, y_bins - 1)

# Get min accepted depth for each noise sounding's cell
min_acc_per_noise = min_accepted_depth[noise_xi, noise_yi]

# Above seafloor noise = shallower than min accepted in same cell
above_seafloor = noise_z < min_acc_per_noise

total_noise          = len(noise_z)
above_seafloor_count = above_seafloor.sum()
below_seafloor_count = (~above_seafloor).sum()

# CNN removed above seafloor noise
cnn_removed_above = (above_seafloor & (noise_cnn == 0)).sum()
cnn_removed_below = (~above_seafloor & (noise_cnn == 0)).sum()

print(f"\nTotal noise soundings        : {total_noise:,}")
print(f"Above seafloor noise         : {above_seafloor_count:,} ({100*above_seafloor_count/total_noise:.1f}%)")
print(f"Below/at seafloor noise      : {below_seafloor_count:,} ({100*below_seafloor_count/total_noise:.1f}%)")
print(f"\nCNN removed above seafloor   : {cnn_removed_above:,} ({100*cnn_removed_above/above_seafloor_count:.1f}% of above seafloor noise)")
print(f"CNN removed below seafloor   : {cnn_removed_below:,} ({100*cnn_removed_below/below_seafloor_count:.1f}% of below seafloor noise)")
print(f"\nTotal CNN removed            : {(cnn_removed_above + cnn_removed_below):,} ({100*(cnn_removed_above+cnn_removed_below)/total_noise:.1f}% of all noise)")
