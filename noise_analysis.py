import numpy as np
from pathlib import Path

# Load colored point cloud
pcd = np.loadtxt("path_to_visualizations/pointcloud_colored_A_xyz_status_pos_down.txt")

gt_label  = pcd[:, 3].astype(int)  # expert label
cnn_label = pcd[:, 4].astype(int)  # CNN label

# All noise soundings (GT rejected)
total_noise = (gt_label == 0).sum()

# Noise soundings correctly removed by CNN (GT=0, CNN=0)
removed_noise = ((gt_label == 0) & (cnn_label == 0)).sum()

# Noise soundings missed by CNN (GT=0, CNN=1)
missed_noise = ((gt_label == 0) & (cnn_label == 1)).sum()

print(f"Total noise soundings    : {total_noise:,}")
print(f"Correctly removed        : {removed_noise:,} ({100*removed_noise/total_noise:.1f}%)")
print(f"Missed (still included)  : {missed_noise:,} ({100*missed_noise/total_noise:.1f}%)")

# Also check accepted soundings
total_accepted = (gt_label == 1).sum()
correctly_kept = ((gt_label == 1) & (cnn_label == 1)).sum()
incorrectly_removed = ((gt_label == 1) & (cnn_label == 0)).sum()

print(f"\nTotal accepted soundings : {total_accepted:,}")
print(f"Correctly kept           : {correctly_kept:,} ({100*correctly_kept/total_accepted:.1f}%)")
print(f"Incorrectly removed      : {incorrectly_removed:,} ({100*incorrectly_removed/total_accepted:.1f}%)")