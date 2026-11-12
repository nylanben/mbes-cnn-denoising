import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from config import VOXEL_SIZE, GRID_SHAPE

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------
CNN_DATA_DIR = Path("C:/Users/Benjamin/Desktop/Python_project/.venv/DATA/CNN_ready_data")
PRED_DIR     = Path("C:/Users/Benjamin/Desktop/Python_project/.venv/DATA/preds/Axyz_test_k3")
VIZ_DIR      = Path("C:/Users/Benjamin/Desktop/Python_project/.venv/DATA/visualizations")
SAVE_DIR     = Path("C:/Users/Benjamin/Desktop/Python_project")

DATASET_ID      = "A_xyz_status_pos_down"
FILENAME_SUFFIX = "vox1_cub32x32x32_ch2"

# ------------------------------------------------------------
# 2. Load preprocessed data
# ------------------------------------------------------------
print("Loading preprocessed data...")
X           = np.load(CNN_DATA_DIR / f"X_{DATASET_ID}_{FILENAME_SUFFIX}.npy")
Y_mask      = np.load(CNN_DATA_DIR / f"Y_mask_{DATASET_ID}_{FILENAME_SUFFIX}.npy")
cuboid_keys = np.load(CNN_DATA_DIR / f"cuboid_keys_{DATASET_ID}_{FILENAME_SUFFIX}.npy")
pcd_center  = np.load(CNN_DATA_DIR / f"pcd_center_{DATASET_ID}.npy")
print(f"pcd_center: {pcd_center}")
preds_all   = np.load(PRED_DIR / f"preds_{DATASET_ID}.npy")
print("Preprocessed data loaded.")

# ------------------------------------------------------------
# 3. Cuboid search functions
# ------------------------------------------------------------
def find_perfect_cuboids(min_tn=50, min_tp=200, max_fp=20):
    """CNN successfully removed most noise."""
    results = []
    for i in range(len(X)):
        occ  = X[i, 1] > 0
        gt   = Y_mask[i, 0].astype(bool)
        pred = preds_all[i].astype(bool)
        tn = int((occ & ~gt & ~pred).sum())
        fp = int((occ & ~gt & pred).sum())
        fn = int((occ & gt & ~pred).sum())
        tp = int((occ & gt & pred).sum())
        if tn >= min_tn and fp <= max_fp and fn == 0 and tp >= min_tp:
            results.append((i, tn, fp, tp))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def find_ideal_cuboids(min_tn=10, min_tp=500):
    """CNN perfectly cleaned — zero FP, zero FN, high TP."""
    results = []
    for i in range(len(X)):
        occ  = X[i, 1] > 0
        gt   = Y_mask[i, 0].astype(bool)
        pred = preds_all[i].astype(bool)
        tn = int((occ & ~gt & ~pred).sum())
        fp = int((occ & ~gt & pred).sum())
        fn = int((occ & gt & ~pred).sum())
        tp = int((occ & gt & pred).sum())
        total_occ = int(occ.sum())
        if tn >= min_tn and fp == 0 and fn == 0 and tp >= min_tp:
            results.append((i, tn, tp, total_occ))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def find_no_removal_cuboids(min_fp=100, min_tp=100):
    """CNN failed to remove noise."""
    results = []
    for i in range(len(X)):
        occ  = X[i, 1] > 0
        gt   = Y_mask[i, 0].astype(bool)
        pred = preds_all[i].astype(bool)
        tn = int((occ & ~gt & ~pred).sum())
        fp = int((occ & ~gt & pred).sum())
        tp = int((occ & gt & pred).sum())
        if fp >= min_fp and tn == 0 and tp >= min_tp:
            results.append((i, fp, tp))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def find_overfill_cuboids(min_fp=200, min_tp=100):
    """CNN accepted more than GT — potential over-cleaning correction."""
    results = []
    for i in range(len(X)):
        occ  = X[i, 1] > 0
        gt   = Y_mask[i, 0].astype(bool)
        pred = preds_all[i].astype(bool)
        fp = int((occ & ~gt & pred).sum())
        tp = int((occ & gt & pred).sum())
        if fp >= min_fp and tp >= min_tp:
            results.append((i, fp, tp))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ------------------------------------------------------------
# 4. Raster visualization
# ------------------------------------------------------------
def rasterize_cuboid_data(occupancy_mask, label_mask, grid_shape=GRID_SHAPE,
                           cuboid_key=None):
    """Rasterize a cuboid to 2D using world depth values."""
    grid_x, grid_y, grid_z = grid_shape
    raster = np.full((grid_x, grid_y), np.nan)

    if cuboid_key is not None:
        z_offset = cuboid_key[2] * grid_shape[2] * VOXEL_SIZE + pcd_center[2]
    else:
        z_offset = 0.0

    occ_x, occ_y, occ_z = np.where(occupancy_mask)
    for idx in range(len(occ_x)):
        x, y, z = occ_x[idx], occ_y[idx], occ_z[idx]
        if label_mask[x, y, z]:
            world_z = z * VOXEL_SIZE + z_offset
            if np.isnan(raster[x, y]) or world_z < raster[x, y]:
                raster[x, y] = world_z
    return raster

def visualize_cuboid_comparison(cuboid_idx, save_path=None):
    """Three-way raster comparison: Raw | Ground Truth | CNN Predicted | Difference"""
    occupancy  = X[cuboid_idx, 1] > 0
    gt_mask    = Y_mask[cuboid_idx, 0].astype(bool)
    pred_mask  = preds_all[cuboid_idx].astype(bool)
    cuboid_key = cuboid_keys[cuboid_idx]

    raster_raw = rasterize_cuboid_data(occupancy, occupancy,   cuboid_key=cuboid_key)
    raster_gt  = rasterize_cuboid_data(occupancy, gt_mask,     cuboid_key=cuboid_key)
    raster_cnn = rasterize_cuboid_data(occupancy, pred_mask,   cuboid_key=cuboid_key)

    diff_map = raster_cnn - raster_gt
    diff_abs = np.nanmax(np.abs(diff_map)) if not np.all(np.isnan(diff_map)) else 1.0
    z_min    = np.nanmin([raster_raw, raster_gt, raster_cnn])
    z_max    = np.nanmax([raster_raw, raster_gt, raster_cnn])

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f"Cuboid {cuboid_idx} - Raw vs GT vs CNN", fontsize=20)

    for ax, raster, title in zip(
        axes[:3],
        [raster_raw, raster_gt, raster_cnn],
        ["Raw (all soundings)", "Ground Truth", "CNN Predicted"]
    ):
        im = ax.imshow(raster.T, origin="lower", cmap="viridis_r",
                       vmin=z_min, vmax=z_max)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("X voxels", fontsize=14)
        ax.set_ylabel("Y voxels", fontsize=14)
        cb = plt.colorbar(im, ax=ax, label="Depth (m)", pad=0.02)
        cb.ax.tick_params(labelsize=14)
        cb.set_label("Depth (m)", fontsize=16)
        cb.ax.invert_yaxis()

    im3 = axes[3].imshow(diff_map.T, origin="lower", cmap="RdBu",
                          vmin=-diff_abs, vmax=diff_abs)
    axes[3].set_title("Difference (CNN - GT)", fontsize=18)
    axes[3].set_xlabel("X voxels", fontsize=14)
    axes[3].set_ylabel("Y voxels", fontsize=14)
    cb3 = plt.colorbar(im3, ax=axes[3], label="Depth diff (m)", pad=0.02)
    cb3.ax.tick_params(labelsize=14)
    cb3.set_label("Diff (m)", fontsize=16)
    cb3.ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()

# ------------------------------------------------------------
# 5. 3D sounding visualization (loads point cloud only when called)
# ------------------------------------------------------------
def visualize_cuboid_soundings_3d(cuboid_idx, elev=20, azim=-60, save_path=None):
    """
    Visualize individual soundings within a cuboid in 3D.
    Blue = accepted, Red = rejected
    Loads the colored point cloud on demand.
    """

    # Load colored point cloud only when needed
    print("Loading colored point cloud...")
    PCD_FILE   = VIZ_DIR / f"pointcloud_colored_{DATASET_ID}.txt"
    pcd_data   = np.loadtxt(PCD_FILE)
    all_points = pcd_data[:, :3]
    all_gt     = pcd_data[:, 3].astype(int)
    all_cnn    = pcd_data[:, 4].astype(int)
    print(f"Loaded {len(all_points):,} points")

    cuboid_key = cuboid_keys[cuboid_idx]

    # Compute world coordinate bounds of this cuboid
    x_lo = cuboid_key[0] * 32 * VOXEL_SIZE + pcd_center[0]
    x_hi = x_lo + 32 * VOXEL_SIZE
    y_lo = cuboid_key[1] * 32 * VOXEL_SIZE + pcd_center[1]
    y_hi = y_lo + 32 * VOXEL_SIZE
    z_lo = cuboid_key[2] * 32 * VOXEL_SIZE + pcd_center[2]
    z_hi = z_lo + 32 * VOXEL_SIZE

    # Filter soundings to this cuboid
    mask = (
        (all_points[:, 0] >= x_lo) & (all_points[:, 0] < x_hi) &
        (all_points[:, 1] >= y_lo) & (all_points[:, 1] < y_hi) &
        (all_points[:, 2] >= z_lo) & (all_points[:, 2] < z_hi)
    )

    pts = all_points[mask]
    gt  = all_gt[mask]
    cnn = all_cnn[mask]

    if len(pts) == 0:
        print(f"No soundings found in cuboid {cuboid_idx}")
        return

    print(f"Cuboid {cuboid_idx}: {len(pts):,} soundings")

    # Relative X and Y within cuboid (0-32m)
    pts_rel_x = pts[:, 0] - x_lo
    pts_rel_y = pts[:, 1] - y_lo
    pts_z     = pts[:, 2]
    z_data_min = pts_z.min()
    z_data_max = pts_z.max()
    z_padding  = 2  # 2 meter padding above and below

    z_plot_lo = z_data_min - z_padding
    z_plot_hi = z_data_max + z_padding

    gt_colors  = np.where(gt == 1,  'blue', 'red')
    cnn_colors = np.where(cnn == 1, 'blue', 'red')

    fig = plt.figure(figsize=(12, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pts_rel_x, pts_rel_y, pts_z,
                c=gt_colors, s=0.5, alpha=0.5)
    ax1.set_title(f"Cuboid {cuboid_idx} — Ground Truth", fontsize=14, pad=1)
    ax1.set_xlabel("X (m)", fontsize=14, labelpad=4)
    ax1.set_ylabel("Y (m)", fontsize=14, labelpad=4)
    ax1.set_zlabel("Depth (m)", fontsize=14, labelpad=4)
    ax1.set_xlim(0, 32)
    ax1.set_ylim(0, 32)
    ax1.set_zlim(z_plot_lo, z_plot_hi)
    ax1.invert_zaxis()
    ax1.tick_params(labelsize=12)
    ax1.view_init(elev=elev, azim=azim)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(pts_rel_x, pts_rel_y, pts_z,
                c=cnn_colors, s=0.5, alpha=0.5)
    ax2.set_title(f"Cuboid {cuboid_idx} — CNN prediction", fontsize=14, pad=1)
    ax2.set_xlabel("X (m)", fontsize=14, labelpad=4)
    ax2.set_ylabel("Y (m)", fontsize=14, labelpad=4)
    ax2.set_zlabel("Depth (m)", fontsize=14, labelpad=4)
    ax2.set_xlim(0, 32)
    ax2.set_ylim(0, 32)
    ax2.set_zlim(z_plot_lo, z_plot_hi)
    ax2.invert_zaxis()
    ax2.tick_params(labelsize=12)
    ax2.view_init(elev=elev, azim=azim)

    # plt.subplots_adjust(wspace=0.1, left=0.05, right=0.95, top=0.85, bottom=0.1)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.35)
        print(f"Saved: {save_path}")
    else:
        plt.show()

# ------------------------------------------------------------
# 6. Search and visualize
# ------------------------------------------------------------

# Set to True to automatically visualize best examples from each category
# Set to False to manually choose cuboid index
AUTO_VISUALIZE = False

# Set to True to also generate 3D sounding visualization (slow — loads 84M points)
VISUALIZE_3D = True

print("\nSearching for interesting cuboids...")
perfect  = find_perfect_cuboids()
ideal    = find_ideal_cuboids()
no_clean = find_no_removal_cuboids()
overfill = find_overfill_cuboids()

print(f"\nTop 10 good cleaning cuboids:")
for idx, tn, fp, tp in perfect[:10]:
    print(f"  Cuboid {idx}: {tn} TN, {fp} FP, 0 FN, {tp} TP")

print(f"\nTop 10 ideal cleaning cuboids:")
for idx, tn, tp, total in ideal[:10]:
    print(f"  Cuboid {idx}: {tn} TN, 0 FP, 0 FN, {tp} TP, {total} total occupied")

print(f"\nTop 10 no removal cuboids:")
for idx, fp, tp in no_clean[:10]:
    print(f"  Cuboid {idx}: {fp} FP, 0 TN, {tp} TP")

print(f"\nTop 10 overfill cuboids:")
for idx, fp, tp in overfill[:10]:
    print(f"  Cuboid {idx}: {fp} FP, {tp} TP")

if AUTO_VISUALIZE:
    if perfect:
        print(f"\nVisualizing best good cleaning cuboid: {perfect[0][0]}")
        visualize_cuboid_comparison(perfect[0][0],
                                    save_path=SAVE_DIR / f"case1_good_{perfect[0][0]}.png")
        if VISUALIZE_3D:
            visualize_cuboid_soundings_3d(perfect[0][0],
                                          save_path=SAVE_DIR / f"case1_good_{perfect[0][0]}_3d.png")
    if ideal:
        print(f"\nVisualizing best ideal cleaning cuboid: {ideal[0][0]}")
        visualize_cuboid_comparison(ideal[0][0],
                                    save_path=SAVE_DIR / f"case1_ideal_{ideal[0][0]}.png")
        if VISUALIZE_3D:
            visualize_cuboid_soundings_3d(ideal[0][0],
                                          save_path=SAVE_DIR / f"case1_ideal_{ideal[0][0]}_3d.png")
    if no_clean:
        print(f"\nVisualizing best no removal cuboid: {no_clean[0][0]}")
        visualize_cuboid_comparison(no_clean[0][0],
                                    save_path=SAVE_DIR / f"case2_no_removal_{no_clean[0][0]}.png")
        if VISUALIZE_3D:
            visualize_cuboid_soundings_3d(no_clean[0][0],
                                          save_path=SAVE_DIR / f"case2_no_removal_{no_clean[0][0]}_3d.png")
    if overfill:
        print(f"\nVisualizing best overfill cuboid: {overfill[0][0]}")
        visualize_cuboid_comparison(overfill[0][0],
                                    save_path=SAVE_DIR / f"case3_overfill_{overfill[0][0]}.png")
        if VISUALIZE_3D:
            visualize_cuboid_soundings_3d(overfill[0][0],
                                          save_path=SAVE_DIR / f"case3_overfill_{overfill[0][0]}_3d.png")

else:
    # Manually choose cuboid to visualize
    CUBOID_IDX = 467  # change this to any cuboid index
    print(f"\nVisualizing cuboid {CUBOID_IDX}")
    visualize_cuboid_comparison(CUBOID_IDX,
                                save_path=SAVE_DIR / f"cuboid_{CUBOID_IDX}.png")
    if VISUALIZE_3D:
        visualize_cuboid_soundings_3d(CUBOID_IDX, elev=5, azim=-40,
                                      save_path=SAVE_DIR / f"cuboid_{CUBOID_IDX}_3d.png")

# Temporary — find cuboids with floating noise above seafloor
def find_floating_noise_cuboids(min_fp=20):
    results = []
    for i in range(len(X)):
        occ  = X[i, 1] > 0
        gt   = Y_mask[i, 0].astype(bool)
        pred = preds_all[i].astype(bool)
        
        fp_z = np.where(occ & ~gt & pred)[2]
        tp_z = np.where(occ & gt & pred)[2]
        
        if len(fp_z) >= min_fp and len(tp_z) > 0:
            min_fp_z = fp_z.min()
            min_tp_z = tp_z.min()
            depth_diff = int(min_tp_z - min_fp_z)
            if depth_diff > 3:  # FP at least 3 voxels above seafloor
                results.append((i, len(fp_z), depth_diff))
    
    results.sort(key=lambda x: x[2], reverse=True)
    return results

floating = find_floating_noise_cuboids()
print("\nFloating noise cuboids:")
for idx, fp, diff in floating[:10]:
    print(f"  Cuboid {idx}: {fp} FP, {diff} voxels above seafloor")