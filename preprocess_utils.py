import numpy as np


def voxelize_and_divide(points, accepted_mask, voxel_size=1.0, grid_shape=(32, 32, 32), verbose=True):
    """
    Voxelize a point cloud and divide into cuboids.
    Each voxel stores:
        [vx, vy, vz, density, accepted_binary]

    Returns:
        cuboids: dict
            key   -> (cx, cy, cz) cuboid index
            value -> dict with:
                     "voxels": ndarray (N,5)
    """
    points = np.asarray(points, dtype=np.float32)
    accepted_mask = accepted_mask.astype(np.int32)

    # Make sure mask and points do not misalign
    assert len(points) == len(accepted_mask), \
        f"Mismatch: {len(points)} points vs {len(accepted_mask)} mask entries"

    # --------------------------------------------------------
    # Integer voxelization
    # --------------------------------------------------------
    voxel_idx = np.floor(points / voxel_size).astype(np.int32)

    # --------------------------------------------------------
    # Hash voxel indices -> group points into voxels
    # --------------------------------------------------------
    voxel_keys = np.rec.fromarrays(
        voxel_idx.T,
        names=("x", "y", "z")
    )
    _, first_occurrence, inverse = np.unique(voxel_keys, return_index=True, return_inverse=True)

    total_counts    = np.bincount(inverse)
    accepted_counts = np.bincount(inverse, weights=accepted_mask)

    # binary accepted per voxel (ANY accepted point)
    accepted_binary = (accepted_counts > 0).astype(np.uint8)

    unique_vox = voxel_idx[first_occurrence]

    if verbose:
        # Sanity check: actual z distribution in world voxel space
        print("Z distribution (world voxel indices):")
        print(np.percentile(unique_vox[:, 2], [1, 5, 50, 95, 99]))

    # --------------------------------------------------------
    # Cuboid indexing
    # --------------------------------------------------------
    voxels_per_cuboid = np.array(grid_shape, dtype=np.int32)

    cuboid_idx = unique_vox // voxels_per_cuboid
    local_idx  = unique_vox % voxels_per_cuboid

    # --------------------------------------------------------
    # Group voxels into cuboids
    # --------------------------------------------------------
    cuboids = {}

    for i in range(len(unique_vox)):
        key = tuple(int(x) for x in cuboid_idx[i])

        if key not in cuboids:
            cuboids[key] = {"voxels": []}

        cuboids[key]["voxels"].append([
            local_idx[i, 0],
            local_idx[i, 1],
            local_idx[i, 2],
            total_counts[i],
            accepted_binary[i]
        ])

    # Convert voxel lists to numpy arrays
    for key, data in cuboids.items():
        data["voxels"] = np.asarray(data["voxels"], dtype=np.int32)

    return cuboids


def cuboids_to_segmentation_tensors(cuboids, grid_shape=(32,32,32), density_scale=100.0, min_soundings=10):
    """
    Convert cuboids dict to dense tensors for CNN training/inference.

    Returns:
        X        : ndarray (N, 2, gx, gy, gz) — input tensors
        Y        : ndarray (N, 1, gx, gy, gz) — mask tensors
        keys     : ndarray (N, 3) — cuboid keys for world coordinate reconstruction
    """
    X_list   = []
    Y_list   = []
    key_list = []
    removed  = 0

    Xdim, Ydim, Zdim = grid_shape

    for i, (key, data) in enumerate(cuboids.items()):
        if i % 100 == 0:
            print(f"Processing cuboid {i}/{len(cuboids)}")

        vox = data["voxels"]
        if len(vox) == 0:
            continue

        # Filter cuboids with too few soundings
        total_soundings = vox[:, 3].sum()
        if total_soundings < min_soundings:
            removed += 1
            continue

        input_tensor = np.zeros((2, Xdim, Ydim, Zdim), dtype=np.float32)
        mask_tensor  = np.zeros((1, Xdim, Ydim, Zdim), dtype=np.float32)

        # bounds check is a safety net; indices always valid by construction
        for x, y, z, density, accepted in vox:
            if 0 <= x < Xdim and 0 <= y < Ydim and 0 <= z < Zdim:
                input_tensor[0, x, y, z] = min(density / density_scale, 1.0)
                input_tensor[1, x, y, z] = 1.0
                mask_tensor[0, x, y, z]  = accepted

        X_list.append(input_tensor)
        Y_list.append(mask_tensor)
        key_list.append(list(key))  # store (cx, cy, cz)

    print(f"\nCuboid filtering summary:")
    print(f"Total cuboids    : {len(cuboids)}")
    print(f"Removed (< {min_soundings} soundings): {removed}")
    print(f"Remaining        : {len(cuboids) - removed}")

    if not X_list:
        raise ValueError("No valid cuboids found - check voxel_size and input data.")

    return np.stack(X_list), np.stack(Y_list), np.array(key_list, dtype=np.int32)