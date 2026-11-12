import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
from mpl_toolkits.mplot3d import Axes3D
from config import VOXEL_SIZE, GRID_SHAPE, MIN_SOUNDINGS
from uNet3D_CNN import UNet3D as UNet3D # Model K3

# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

VISUALIZE_ONLY = False  # False runs the full inference; True only visualizes saved predictions

MODEL_NAME = "Axyz_test_f64"  # Change this for each model

CNN_DATA_DIR = Path("path_to_CNN_ready_data")
MODEL_PATH = Path(f"path_to_models/best_model_{MODEL_NAME}.pt")
OUTPUT_DIR = Path(f"path_to_predictions/{MODEL_NAME}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID      = "A_xyz_status_pos_down"
FILENAME_SUFFIX = "vox1_cub32x32x32_ch2"

# ------------------------------------------------------------
# 2. Load preprocessed data
# ------------------------------------------------------------
print("Loading X...")
X            = np.load(CNN_DATA_DIR / f"X_{DATASET_ID}_{FILENAME_SUFFIX}.npy")
print("Loading Y_mask...")
Y_mask       = np.load(CNN_DATA_DIR / f"Y_mask_{DATASET_ID}_{FILENAME_SUFFIX}.npy")
print("Loading cuboid_keys...")
cuboid_keys  = np.load(CNN_DATA_DIR / f"cuboid_keys_{DATASET_ID}_{FILENAME_SUFFIX}.npy")
print("Loading pcd_center...")
pcd_center   = np.load(CNN_DATA_DIR / f"pcd_center_{DATASET_ID}.npy")
print("Loading density_scale...")
density_scale = float(np.load(CNN_DATA_DIR / f"density_scale_{DATASET_ID}.npy"))

print(f"Loaded {len(X)} cuboids for {DATASET_ID}")
print(f"Density scale: {density_scale:.2f}")

# ------------------------------------------------------------
# 3. Load model (needed even for visualization only)
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = UNet3D().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

print(f"Model loaded from {MODEL_PATH}")
print(f"Running on: {device}")

# ------------------------------------------------------------
# 4. Full inference (skipped when VISUALIZE_ONLY=True)
# ------------------------------------------------------------
if not VISUALIZE_ONLY:

    # Check files exist
    for prefix in ["X", "Y_mask", "cuboid_keys"]:
        f = CNN_DATA_DIR / f"{prefix}_{DATASET_ID}_{FILENAME_SUFFIX}.npy"
        assert f.exists(), f"File not found: {f}"
    assert (CNN_DATA_DIR / f"density_scale_{DATASET_ID}.npy").exists()
    assert (CNN_DATA_DIR / f"pcd_center_{DATASET_ID}.npy").exists()
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"

    all_points_accepted = []
    all_points_gt       = []
    all_points_raw      = []
    y_true = []
    y_pred = []
    cuboid_preds = {}

    voxels_per_cuboid = np.array(GRID_SHAPE)

    print(f"Running inference on {len(X)} cuboids...")

    with torch.no_grad():
        for i in range(len(X)):
            if i % 100 == 0:
                print(f"Processing cuboid {i}/{len(X)}")

            input_tensor = torch.from_numpy(X[i]).unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs  = torch.sigmoid(logits)
            preds  = (probs > 0.5).squeeze().cpu().numpy()

            cuboid_preds[i] = preds

            occupancy  = X[i, 1]
            gt_mask    = Y_mask[i, 0]
            cuboid_idx = cuboid_keys[i]

            occ_x, occ_y, occ_z = np.where(occupancy > 0)

            if len(occ_x) > 0:
                y_true.extend(gt_mask[occ_x, occ_y, occ_z].astype(int).tolist())
                y_pred.extend(preds[occ_x, occ_y, occ_z].astype(int).tolist())

                local_positions = np.stack([occ_x, occ_y, occ_z], axis=1)
                world_positions = (
                    (cuboid_idx * voxels_per_cuboid + local_positions + 0.5)
                    * VOXEL_SIZE
                ) + pcd_center

                all_points_raw.append(world_positions)

                pred_mask = preds[occ_x, occ_y, occ_z].astype(bool)
                if pred_mask.any():
                    all_points_accepted.append(world_positions[pred_mask])

                gt_accepted_mask = gt_mask[occ_x, occ_y, occ_z].astype(bool)
                if gt_accepted_mask.any():
                    all_points_gt.append(world_positions[gt_accepted_mask])

    if len(all_points_accepted) == 0:
        raise ValueError("No accepted points predicted — check model and input data.")

    all_points_accepted = np.concatenate(all_points_accepted, axis=0).astype(np.float32)
    all_points_gt       = np.concatenate(all_points_gt,       axis=0).astype(np.float32)
    all_points_raw      = np.concatenate(all_points_raw,      axis=0).astype(np.float32)

    np.save(OUTPUT_DIR / f"predicted_accepted_{DATASET_ID}.npy", all_points_accepted)
    np.save(OUTPUT_DIR / f"gt_accepted_{DATASET_ID}.npy",        all_points_gt)
    np.save(OUTPUT_DIR / f"raw_points_{DATASET_ID}.npy",         all_points_raw)

    print(f"\nSaved {len(all_points_accepted)} predicted accepted points")
    print(f"Saved {len(all_points_gt)} ground truth accepted points")
    print(f"Saved {len(all_points_raw)} raw points")

    with open(OUTPUT_DIR / f"cuboid_preds_{DATASET_ID}.pkl", "wb") as f:
        pickle.dump(cuboid_preds, f)

    preds_array = np.array([cuboid_preds[i] for i in range(len(X))])
    np.save(OUTPUT_DIR / f"preds_{DATASET_ID}.npy", preds_array)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    print(f"\nTest set evaluation:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Kappa: {kappa:.4f}")
    print(f"Confusion matrix:\n{cm}")

# ------------------------------------------------------------
# 5. Per-cuboid visualization
# ------------------------------------------------------------
def visualize_cuboid_comparison(X, Y_mask, preds_all, cuboid_idx,
                                 cuboid_keys, pcd_center,
                                 grid_shape=GRID_SHAPE, save_path=None):
    """
    Visualize a single cuboid showing:
    - Ground truth accepted voxels (shoalest world depth per XY cell)
    - CNN predicted accepted voxels (shoalest world depth per XY cell)
    - Difference map (CNN - GT) in meters
    """
    occupancy = X[cuboid_idx, 1]
    gt_mask   = Y_mask[cuboid_idx, 0]
    pred_mask = preds_all[cuboid_idx].astype(bool)

    grid_x, grid_y, grid_z = grid_shape

    raster_gt  = np.full((grid_x, grid_y), np.nan)
    raster_cnn = np.full((grid_x, grid_y), np.nan)

    # Compute world Z offset for this cuboid
    cuboid_key = cuboid_keys[cuboid_idx]
    z_offset = cuboid_key[2] * grid_shape[2] * VOXEL_SIZE + pcd_center[2]

    occ_x, occ_y, occ_z = np.where(occupancy > 0)

    for idx in range(len(occ_x)):
        x, y, z = occ_x[idx], occ_y[idx], occ_z[idx]

        # Convert local z index to world depth in meters
        world_z = z * VOXEL_SIZE + z_offset

        # Ground truth — shoalest accepted voxel in world depth
        if gt_mask[x, y, z] == 1:
            if np.isnan(raster_gt[x, y]) or world_z < raster_gt[x, y]:
                raster_gt[x, y] = world_z

        # CNN — shoalest predicted accepted voxel in world depth
        if pred_mask[x, y, z]:
            if np.isnan(raster_cnn[x, y]) or world_z < raster_cnn[x, y]:
                raster_cnn[x, y] = world_z

    diff_map = raster_cnn - raster_gt
    diff_abs = np.nanmax(np.abs(diff_map)) if not np.all(np.isnan(diff_map)) else 1.0

    z_min = np.nanmin([raster_gt, raster_cnn])
    z_max = np.nanmax([raster_gt, raster_cnn])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Cuboid index {cuboid_idx}", fontsize=14)

    im0 = axes[0].imshow(raster_gt.T, origin="lower",
                         cmap="viridis_r", vmin=z_min, vmax=z_max)
    axes[0].set_title("Ground Truth (shoalest)")
    axes[0].set_xlabel("X voxels")
    axes[0].set_ylabel("Y voxels", labelpad=10)
    plt.colorbar(im0, ax=axes[0], label="Depth (m)", pad=0.02)

    im1 = axes[1].imshow(raster_cnn.T, origin="lower",
                         cmap="viridis_r", vmin=z_min, vmax=z_max)
    axes[1].set_title("CNN Prediction (shoalest)")
    axes[1].set_xlabel("X voxels")
    axes[1].set_ylabel("Y voxels", labelpad=10)
    plt.colorbar(im1, ax=axes[1], label="Depth (m)", pad=0.02)

    im2 = axes[2].imshow(diff_map.T, origin="lower",
                         cmap="RdBu", vmin=-diff_abs, vmax=diff_abs)
    axes[2].set_title("Difference (CNN - GT)")
    axes[2].set_xlabel("X voxels")
    axes[2].set_ylabel("Y voxels", labelpad=10)
    plt.colorbar(im2, ax=axes[2], label="Depth difference (m)", pad=0.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def visualize_cuboid_pointcloud(X, Y_mask, preds_all, cuboid_idx,
                                 cuboid_keys, pcd_center,
                                 grid_shape=GRID_SHAPE, save_path=None):
    """
    Visualize a single cuboid as two 3D scatter plots side by side:
    - Left: ground truth (blue=accepted, red=rejected)
    - Right: CNN prediction (blue=accepted, red=rejected)
    Each point represents a voxel center.
    """
    occupancy  = X[cuboid_idx, 1]
    gt_mask    = Y_mask[cuboid_idx, 0]
    pred_mask  = preds_all[cuboid_idx].astype(bool)

    # Get cuboid world offset
    cuboid_key = cuboid_keys[cuboid_idx]
    x_offset = cuboid_key[0] * grid_shape[0] * VOXEL_SIZE + pcd_center[0]
    y_offset = cuboid_key[1] * grid_shape[1] * VOXEL_SIZE + pcd_center[1]
    z_offset = cuboid_key[2] * grid_shape[2] * VOXEL_SIZE + pcd_center[2]

    # Find occupied voxels
    occ_x, occ_y, occ_z = np.where(occupancy > 0)

    if len(occ_x) == 0:
        print(f"Cuboid {cuboid_idx} is empty.")
        return

    # Convert to world coordinates
    world_x = occ_x * VOXEL_SIZE + x_offset + 0.5
    world_y = occ_y * VOXEL_SIZE + y_offset + 0.5
    world_z = occ_z * VOXEL_SIZE + z_offset + 0.5

    # Colors based on ground truth
    gt_colors = np.where(gt_mask[occ_x, occ_y, occ_z] == 1, 'blue', 'red')

    # Colors based on CNN prediction
    cnn_colors = np.where(pred_mask[occ_x, occ_y, occ_z], 'blue', 'red')

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f"Cuboid index {cuboid_idx} — voxel centers\n"
                 f"Blue = accepted, Red = rejected", fontsize=13)

    # Ground truth
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(world_x, world_y, world_z,
                c=gt_colors, s=2, alpha=0.6)
    ax1.set_title("Ground Truth")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.invert_zaxis()  # shallow at top, deep at bottom

    # CNN prediction
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(world_x, world_y, world_z,
                c=cnn_colors, s=2, alpha=0.6)
    ax2.set_title("CNN Prediction")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.invert_zaxis()  # shallow at top, deep at bottom

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

# ------------------------------------------------------------
# 6. Load predictions and visualize
# ------------------------------------------------------------
print("Loading predictions...")
preds_all = np.load(OUTPUT_DIR / f"preds_{DATASET_ID}.npy")

# Find cuboids with mix of accepted and rejected
for i in range(len(X)):
    gt  = Y_mask[i, 0]
    occ = X[i, 1] > 0
    accepted = (gt == 1) & occ
    rejected = (gt == 0) & occ
    if accepted.sum() > 50 and rejected.sum() > 50:
        print(f"Cuboid {i}: {accepted.sum()} accepted, {rejected.sum()} rejected")
        break

# Change cuboid_idx to any number between 0 and 6832
CUBOID_IDX = 344
visualize_cuboid_comparison(X, Y_mask, preds_all, cuboid_idx=CUBOID_IDX,
                             cuboid_keys=cuboid_keys, pcd_center=pcd_center)
visualize_cuboid_pointcloud(X, Y_mask, preds_all, cuboid_idx=CUBOID_IDX,
                             cuboid_keys=cuboid_keys, pcd_center=pcd_center)