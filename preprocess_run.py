import numpy as np
import pandas as pd
from pathlib import Path
from preprocess_utils import voxelize_and_divide, cuboids_to_segmentation_tensors
from config import VOXEL_SIZE, GRID_SHAPE, DENSITY_SCALE, SEED, CHANNELS, MIN_SOUNDINGS, PER_DATASET_SCALE
import matplotlib.pyplot as plt
 
VISUALIZE = False  #  You may set to True if you have a screen and open3d installed
 
if VISUALIZE:
    import open3d as o3d
 
PROJECT_ROOT = Path(__file__).resolve().parent
 
# Reproducibility
np.random.seed(SEED)
 
# --------
# Load CSV
# --------
DATA = "path_to_data"
dataset_id = Path(DATA).stem  # For informative naming when saving
 
assert Path(DATA).exists(), f"File not found: {DATA}"  # Ensure that the file exists
 
df = pd.read_csv(
    DATA,
    sep=r"\s+",
    header=None,
    usecols=[0, 1, 2, 3],
    engine="python"
)
 
# Assign column names
df.columns = ["X", "Y", "Z", "Status"]
df["Status"] = df["Status"].astype(str).str.lower()
accepted_mask = df["Status"].values == "accepted"  # Accepted points
points = df[["X", "Y", "Z"]].values.astype(np.float32)  # For PCD construction
 
print("Actual Z statistics (world coordinates before centering):")
print(f"Min depth: {df['Z'].min():.2f} m")
print(f"Max depth: {df['Z'].max():.2f} m")
print(f"Mean depth: {df['Z'].mean():.2f} m")
print(f"Std depth: {df['Z'].std():.2f} m")
 
# Sounding statistics
total_soundings    = len(df)
accepted_soundings = int(accepted_mask.sum())
rejected_soundings = total_soundings - accepted_soundings
 
print(f"\nSounding statistics:")
print(f"Total soundings    : {total_soundings:,}")
print(f"Accepted soundings : {accepted_soundings:,} ({100*accepted_soundings/total_soundings:.1f}%)")
print(f"Rejected soundings : {rejected_soundings:,} ({100*rejected_soundings/total_soundings:.1f}%)")
 
# Save directory for preprocessed data
SAVE_DIR = PROJECT_ROOT / ".venv/DATA/CNN_ready_data"  # local
SAVE_DIR.mkdir(parents=True, exist_ok=True)
 
# Save histogram of depth distribution
 
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['Z'].values, bins=100, alpha=0.7, label='All soundings', color='gray')
ax.hist(df['Z'].values[~accepted_mask], bins=100, alpha=0.7, label='Rejected', color='red')
ax.set_xlabel('Depth (m)')
ax.set_ylabel('Number of soundings')
ax.set_title(f'Depth distribution — {dataset_id}')
ax.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR / f"depth_distribution_{dataset_id}.png", dpi=150)
plt.close()
print(f"Saved depth distribution plot")
 
# -------------------------
# Create point cloud
# -------------------------
if VISUALIZE:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros((len(df), 3))
    colors[accepted_mask] = [0, 1, 0]   # green
    colors[~accepted_mask] = [1, 0, 0]  # red
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd_center = pcd.get_center()  # Center the point cloud
    pcd.translate(-pcd_center)
    o3d.visualization.draw_geometries([pcd])  # Visualize raw point cloud
else:
    pcd_center = points.mean(axis=0)
    points -= pcd_center
 
# Save pcd_center (works with and without o3d)
np.save(SAVE_DIR / f"pcd_center_{dataset_id}.npy", pcd_center)
 
cuboids = voxelize_and_divide(
    points,
    accepted_mask,
    voxel_size=VOXEL_SIZE,
    grid_shape=GRID_SHAPE
)
 
# ---------------
# Inspect cuboids
# ---------------
print("Number of cuboids:", len(cuboids))
print("First 20 cuboid keys:", list(cuboids.keys())[:20])
 
example_key = next(
    key for key, data in cuboids.items()
    if len(data["voxels"]) > 0
)
example_voxels = cuboids[example_key]["voxels"]
 
print("Example cuboid:", example_key)
print("Voxel array shape:", example_voxels.shape)
print("First 5 voxels:\n", example_voxels[:5])
 
# --------------------------------------------------
# Visualize voxel centers (accepted vs non-accepted)
# --------------------------------------------------
if VISUALIZE:
    def visualize_cuboid_voxels(cuboid_voxels, cuboid_key,
                                pcd_center,
                                voxel_size=VOXEL_SIZE,
                                grid_shape=GRID_SHAPE):
 
        local_idx = cuboid_voxels[:, :3]
        accepted  = cuboid_voxels[:, 4]
 
        cuboid_idx = np.array(cuboid_key)
        voxels_per_cuboid = np.array(grid_shape)
 
        # reconstruct original world coordinates
        centers = (
            (cuboid_idx * voxels_per_cuboid + local_idx + 0.5)
            * voxel_size
        ) + pcd_center
 
        colors = np.zeros((len(accepted), 3))
        colors[accepted == 1] = [0, 1, 0]  # green = accepted voxel
        colors[accepted == 0] = [1, 0, 0]  # red = not accepted
 
        vox_pcd = o3d.geometry.PointCloud()
        vox_pcd.points = o3d.utility.Vector3dVector(centers)
        vox_pcd.colors = o3d.utility.Vector3dVector(colors)
 
        o3d.visualization.draw_geometries([vox_pcd])
 
    visualize_cuboid_voxels(example_voxels, example_key, pcd_center=pcd_center)
 
 
def save_cnn_dataset(
    X,
    y,
    cuboid_keys,
    dataset_id,
    voxel_size,
    grid_shape,
    channels,
    out_dir=None
):
    if out_dir is None:
        out_dir = PROJECT_ROOT / ".venv/DATA/CNN_ready_data"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    name = (
        f"{dataset_id}"
        f"_vox{voxel_size:g}"
        f"_cub{grid_shape[0]}x{grid_shape[1]}x{grid_shape[2]}"
        f"_ch{channels}"
    )
 
    np.save(out_dir / f"X_{name}.npy", X)
    np.save(out_dir / f"Y_mask_{name}.npy", y)
    np.save(out_dir / f"cuboid_keys_{name}.npy", cuboid_keys)
 
    print(f"Saved dataset: {name}")
    print(f"Saved cuboid keys: {cuboid_keys.shape}")
 
 
# --------------------------------
# Build CNN-ready dataset and save
# --------------------------------
 
# Check density distribution
all_densities = np.concatenate([
    data["voxels"][:, 3]
    for data in cuboids.values()
    if len(data["voxels"]) > 0
])
print("Density percentiles:", np.percentile(all_densities, [50, 90, 95, 99, 99.9]))
 
if PER_DATASET_SCALE:
    DENSITY_SCALE = float(np.percentile(all_densities, 95))
    print(f"Per-dataset DENSITY_SCALE (p95): {DENSITY_SCALE:.2f}")
else:
    print(f"Global DENSITY_SCALE: {DENSITY_SCALE:.2f}")
 
# Save density scale alongside dataset
np.save(SAVE_DIR / f"density_scale_{dataset_id}.npy", np.array(DENSITY_SCALE))
 
# Convert cuboids to dense tensors
X, Y_mask, cuboid_keys = cuboids_to_segmentation_tensors(
    cuboids,
    GRID_SHAPE,
    DENSITY_SCALE,
    MIN_SOUNDINGS)
 
# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------
print("\nCNN dataset summary:")
print("X shape:", X.shape)          # (N, C, X, Y, Z)
print("X dtype:", X.dtype)
print("Y_mask shape:", Y_mask.shape)
print("Cuboid keys shape:", cuboid_keys.shape)
print("Mask unique values:", np.unique(Y_mask))
print("X channel 0 (density) max:", X[:, 0].max())    # should be <= 1.0
print("X channel 1 (occupancy) unique:", np.unique(X[:, 1]))  # should be [0. 1.]
 
# ------------------------------------------------------------
# Save dataset
# ------------------------------------------------------------
save_cnn_dataset(
    X,
    Y_mask,
    cuboid_keys,
    dataset_id=dataset_id,
    voxel_size=VOXEL_SIZE,
    grid_shape=GRID_SHAPE,
    channels=CHANNELS,
    out_dir=SAVE_DIR
)