import numpy as np
import pandas as pd
from pathlib import Path

from config import VOXEL_SIZE, GRID_SHAPE

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

# Paths
CNN_DATA_DIR = Path("path_to_CNN_ready_data")
PRED_DIR     = Path("path_to_predictions")
DATA_DIR     = Path("path_to_realdata")
OUTPUT_DIR   = Path("path_to_predictions")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID      = "A_xyz_status_pos_down"
FILENAME_SUFFIX = "vox1_cub32x32x32_ch2"

INPUT_FILE = DATA_DIR / f"{DATASET_ID}/{DATASET_ID}.txt"

assert INPUT_FILE.exists(), f"Raw data file not found: {INPUT_FILE}"

# ------------------------------------------------------------
# 2. Load preprocessed data
# ------------------------------------------------------------
print("Loading cuboid keys and predictions...")
cuboid_keys = np.load(CNN_DATA_DIR / f"cuboid_keys_{DATASET_ID}_{FILENAME_SUFFIX}.npy")
pcd_center  = np.load(CNN_DATA_DIR / f"pcd_center_{DATASET_ID}.npy")
preds_all   = np.load(PRED_DIR / f"preds_{DATASET_ID}.npy")

print(f"Loaded {len(cuboid_keys)} cuboids")

# Build lookup: cuboid key -> array index
cuboid_key_to_idx = {tuple(k): i for i, k in enumerate(cuboid_keys)}
voxels_per_cuboid = np.array(GRID_SHAPE, dtype=np.int32)

# ------------------------------------------------------------
# 3. Load raw data and assign CNN labels in chunks
# ------------------------------------------------------------
print("Loading raw ASCII file and assigning CNN labels...")

output_path = OUTPUT_DIR / f"predicted_{DATASET_ID}.xyz"

# Write header
with open(output_path, "w") as f:
    f.write("X Y Z Status\n")

total_written   = 0
total_accepted  = 0
total_rejected  = 0
total_unmatched = 0

for chunk_num, chunk in enumerate(pd.read_csv(
        INPUT_FILE, sep=r"\s+", header=None,
        usecols=[0, 1, 2], chunksize=500_000, engine="python")):

    chunk.columns = ["X", "Y", "Z"]
    points = chunk.values.astype(np.float64)

    # Center points
    points_c = points - pcd_center

    # Compute voxel and cuboid indices
    voxel_idx        = np.floor(points_c / VOXEL_SIZE).astype(np.int32)
    cuboid_idx_point = voxel_idx // voxels_per_cuboid
    local_idx_point  = voxel_idx % voxels_per_cuboid

    # Initialize labels as -1 (unmatched)
    cnn_labels = np.full(len(points), -1, dtype=np.int8)

    # Get unique cuboids in this chunk
    unique_cuboids = np.unique(cuboid_idx_point, axis=0)

    for ck in unique_cuboids:
        key = tuple(ck)
        if key not in cuboid_key_to_idx:
            continue

        arr_idx     = cuboid_key_to_idx[key]
        pred_volume = preds_all[arr_idx]

        # Find points in this cuboid
        in_cuboid = np.all(cuboid_idx_point == ck, axis=1)
        pt_idx    = np.where(in_cuboid)[0]

        lx = local_idx_point[pt_idx, 0]
        ly = local_idx_point[pt_idx, 1]
        lz = local_idx_point[pt_idx, 2]

        # Bounds check
        valid = ((lx >= 0) & (lx < GRID_SHAPE[0]) &
                 (ly >= 0) & (ly < GRID_SHAPE[1]) &
                 (lz >= 0) & (lz < GRID_SHAPE[2]))

        valid_pt_idx = pt_idx[valid]
        cnn_labels[valid_pt_idx] = pred_volume[
            lx[valid], ly[valid], lz[valid]
        ].astype(np.int8)

    # Build status labels vectorized
    status_arr = np.where(cnn_labels == 1, "Accepted",
                 np.where(cnn_labels == 0, "Rejected", "Unmatched"))

    # Fast batch write using pandas
    df_out = pd.DataFrame({
        "X":      points[:, 0].round(4),
        "Y":      points[:, 1].round(4),
        "Z":      points[:, 2].round(4),
        "Status": status_arr
    })

    df_out.to_csv(
        output_path,
        sep=" ",
        index=False,
        header=False,
        mode="a"
    )

    total_accepted  += (cnn_labels == 1).sum()
    total_rejected  += (cnn_labels == 0).sum()
    total_unmatched += (cnn_labels == -1).sum()
    total_written   += len(points)

    if chunk_num % 10 == 0:
        print(f"  Chunk {chunk_num} — {total_written:,} points written")

print(f"\nDone!")
print(f"Total points  : {total_written:,}")
print(f"Accepted      : {total_accepted:,}")
print(f"Rejected      : {total_rejected:,}")
print(f"Unmatched     : {total_unmatched:,}")
print(f"Saved to      : {output_path}")