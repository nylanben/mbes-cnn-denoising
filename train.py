from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import precision_score, recall_score, f1_score

from datasets import CuboidDataset
from uNet3D_CNN import UNet3D as UNet3D # K3 model
from config import GRID_SHAPE, SEED

# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------
DATA_DIR = Path("path_to_CNN_ready_data")

# Training datasets
DATASETS = [
    "broa4_xyzstat_pos_down",
    "aiKK_a02_incl_rejected_pos_down",
    "bhk_a2_ext_incl_rejected_pos_down",
    "m01sh_xyzstat_pos_down",
    "HK_K02_NS_ext_incl_rejected_pos_down_new",
]

FILENAME_SUFFIX = "vox1_cub32x32x32_ch2"

# Check all files exist
for ds in DATASETS:
    for prefix in ["X", "Y_mask"]:
        f = DATA_DIR / f"{prefix}_{ds}_{FILENAME_SUFFIX}.npy"
        assert f.exists(), f"File not found: {f}"
    assert (DATA_DIR / f"density_scale_{ds}.npy").exists(), \
        f"Density scale not found for {ds}"

# ------------------------------------------------------------
# 2. Load datasets with per-dataset density scaling
# ------------------------------------------------------------
train_datasets = []
val_datasets   = []

Y_all = []
X_all = []

for ds in DATASETS:
    X = np.load(DATA_DIR / f"X_{ds}_{FILENAME_SUFFIX}.npy")
    Y = np.load(DATA_DIR / f"Y_mask_{ds}_{FILENAME_SUFFIX}.npy")
    density_scale = float(np.load(DATA_DIR / f"density_scale_{ds}.npy"))

    print(f"{ds}: {len(X)} cuboids, density_scale={density_scale:.2f}")

    X_all.append(X)
    Y_all.append(Y)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, Y = X[idx], Y[idx]

    # 80/20 split
    split = int(0.8 * len(X))

    train_datasets.append(CuboidDataset(X[:split], Y[:split], augment=True))
    val_datasets.append(CuboidDataset(X[split:], Y[split:], augment=False))

# Combine all datasets
train_ds = ConcatDataset(train_datasets)
val_ds   = ConcatDataset(val_datasets)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=4)

# ------------------------------------------------------------
# 3. Class balance across all training data
# ------------------------------------------------------------
Y_combined      = np.concatenate(Y_all)
X_combined      = np.concatenate(X_all)
total_voxels    = Y_combined.size
positive_voxels = Y_combined.sum()

print(f"\nPositive voxel ratio: {positive_voxels / total_voxels:.4f}")

# ------------------------------------------------------------
# 4. Model, loss & optimizer
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = UNet3D().to(device)

print(f"Training on: {device}")
print(f"Train cuboids: {len(train_ds)}")
print(f"Val cuboids:   {len(val_ds)}")

# DEBUG
DEBUG = True
if DEBUG:
    xb, yb = next(iter(train_loader))
    print("Input batch:", xb.shape)
    print("Target batch:", yb.shape)
    out = model(xb.to(device))
    print("Model output:", out.shape)

# Class imbalance
pos_weight_value = (total_voxels - positive_voxels) / positive_voxels
pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32).to(device)
print(f"pos_weight: {pos_weight_value:.4f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.3,
    patience=4
)

# ------------------------------------------------------------
# 5. Training loop
# ------------------------------------------------------------
PROJECT_ROOT    = Path(__file__).resolve().parent
best_val_loss   = float("inf")
epochs_no_improve = 0
early_stop_patience = 16
NUM_EPOCHS = 100

for epoch in range(NUM_EPOCHS):

    # Training
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.float().to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.float().to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), PROJECT_ROOT / "best_model_Axyz_test_K3.pt")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stop_patience:
        print("Early stopping triggered.")
        break

    print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f}")

# ------------------------------------------------------------
# 6. Final evaluation on validation set
# ------------------------------------------------------------
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        y_true.extend(yb.cpu().numpy().ravel())
        y_pred.extend(preds.cpu().numpy().ravel())

precision = precision_score(y_true, y_pred, zero_division=0)
recall    = recall_score(y_true, y_pred, zero_division=0)
f1        = f1_score(y_true, y_pred, zero_division=0)

print(f"\nValidation set evaluation:")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1        : {f1:.4f}")
occupied_mask = X_combined[:, 1].ravel() > 0
print(f"Accepted (occupied): {np.sum(Y_combined.ravel()[occupied_mask] == 1)}")
print(f"Rejected (occupied): {np.sum(Y_combined.ravel()[occupied_mask] == 0)}")
