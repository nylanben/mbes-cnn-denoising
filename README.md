# CNN-based denoising of MBES bathymetric data

Code for the Master's thesis **"Automated Removal of False Echoes in Multibeam Echosounder Data Using Convolutional Neural Networks - Applications in Finnish Marine and Inland Waters"** (Benjamin Nylander, University of Helsinki, 2026).

A 3D U-Net classifies 1 x 1 x 1 m voxels of multibeam echosounder (MBES) soundings as accepted or rejected, following the approach of Stephens et al. (2020). The output is a labelled point cloud (accepted/rejected status for every sounding) and a one-meter resolution raster surface. The model is evaluated against the classification made by a hydrographer (the ground truth).

Thesis: [link to the thesis] # Will be added when published.

## How it works

1. **Preprocessing.** The soundings are centered on the dataset mean, divided into 1 x 1 x 1 m voxels and grouped into 32 x 32 x 32 voxel cuboids (32 m per side) on a fixed grid. Each cuboid has two input channels: density (the number of soundings in the voxel, scaled by the 95th percentile of the dataset and capped at 1.0) and occupancy (1 for voxels that contain soundings). A voxel is labelled accepted if it contains at least one accepted sounding. Cuboids with fewer than 10 soundings are discarded.
2. **Training.** A 3D U-Net is trained on five datasets (random 80/20 split of the cuboids of each dataset into training and validation sets). Four variants were tested (see below).
3. **Inference.** The trained model classifies every voxel of the held-out test dataset (A_xyz). Test precision, recall, F1 and kappa are printed.
4. **Export and evaluation.** The voxel predictions are assigned back to the individual soundings, and the rasters, point clouds and statistics are produced.

## Repository contents

### Preprocessing

| File | Purpose |
|---|---|
| `config.py` | Settings used by the other scripts: voxel size (1 m), cuboid size (32 x 32 x 32), number of channels (2), density scaling, minimum soundings per cuboid (10), random seed (42). |
| `preprocess_utils.py` | Functions used by `preprocess_run.py`: voxelization and division into cuboids, and conversion of the cuboids to dense tensors. |
| `preprocess_run.py` | Preprocesses one dataset. Edit `DATA` (the path to the input file) and run it once per dataset. |

### Model and training

| File | Purpose |
|---|---|
| `uNet3D_CNN.py` | The 3D U-Net variants (see the table below). |
| `datasets.py` | PyTorch dataset class. The training data are augmented with random 90-degree rotations around the vertical axis and random flips along the x and y axes. |
| `train.py` | Trains the model on five datasets and saves the model with the lowest validation loss. Early stopping is used. |

### Prediction, export and evaluation

| File | Purpose |
|---|---|
| `inference.py` | Runs the trained model on the test dataset and saves the voxel predictions (`preds_*.npy`). Prints the test metrics and the confusion matrix. |
| `export_point_cloud.py` | Assigns the predictions to the individual soundings and writes an ASCII file `predicted_<dataset>.xyz` with the columns X, Y, Z and Status (Accepted, Rejected or Unmatched). Unmatched soundings lie in cuboids that were discarded in the preprocessing. |
| `visualize_pointcloud.py` | Creates the colored point cloud `pointcloud_colored_<dataset>.txt` (X, Y, Z, ground-truth label, CNN label), builds 1 m rasters (shoalest sounding per cell) from the raw, ground-truth and CNN-accepted soundings, and prints the raster RMSE, MAE, bias and IHO TVU compliance. |
| `naive_baseline.py` | Raster statistics for the naive baseline, in which all soundings are accepted, compared with the ground truth. |
| `noise_analysis.py` | Sounding-level counts: how many of the soundings rejected in the ground truth were also rejected by the CNN, and how many accepted soundings were kept. |
| `noise_depth_analysis.py` | Splits the rejected soundings into those above the seafloor (shallower than the shallowest accepted sounding in the same 1 m cell) and those at or below it, and shows how many of each were removed. |

### Figures

| File | Purpose |
|---|---|
| `visualize_difference.py` | Zoomed figures of the areas where the CNN raster differs most from the ground-truth raster. |
| `visualize_cuboid.py` | Finds example cuboids (good cleaning, ideal cleaning, no removal, overfill) and plots a single cuboid as a raster comparison and in 3D. |
| `generate_combined_histograms.py` | Depth distribution (all and rejected soundings) of each dataset in one figure. Reads the raw ASCII files. |

## Model variants

All variants are in `uNet3D_CNN.py`. The model is chosen by the import near the top of `train.py` and `inference.py`; remember to change the name of the saved model file (`MODEL_NAME` and the file name in `train.py`) to match.

| Variant | Class | Kernel size | Base filters |
|---|---|---|---|
| K3 (default) | `UNet3D` | 3 | 32 |
| K5 | `UNet3D_k5` | 5 | 32 |
| F16 | `UNet3D_f16` | 3 | 16 |
| F64 | `UNet3D_f64` | 3 | 64 |

For example, `from uNet3D_CNN import UNet3D_f64 as UNet3D` selects F64.

## Data

The MBES data were provided by the Finnish Transport and Communications Agency (Traficom) and are **not included** in this repository.

The input files are whitespace-separated ASCII files without a header, with the columns `X Y Z Status`, where Status is `accepted` or `rejected` (the hydrographer's classification). Six datasets were used: five for training and validation, and one (`A_xyz`) as the held-out test dataset.

## Usage

All file paths in the scripts are placeholders of the form `path_to_...`. Replace them with your own paths before running. Run the scripts in this order:

```
# 1. Preprocess each dataset (edit DATA in the script first)
python preprocess_run.py

# 2. Train (one run per model variant)
python train.py

# 3. Run the model on the test dataset
python inference.py

# 4. Export the labelled point cloud and compute the raster statistics
python export_point_cloud.py
python visualize_pointcloud.py

# 5. Additional analyses and figures (all optional)
python naive_baseline.py
python noise_analysis.py
python noise_depth_analysis.py
python visualize_difference.py
python visualize_cuboid.py
python generate_combined_histograms.py
```

`visualize_pointcloud.py` must be run before the scripts that read `pointcloud_colored_<dataset>.txt` (`naive_baseline.py`, `noise_analysis.py`, `noise_depth_analysis.py`, `visualize_difference.py` and the 3D view in `visualize_cuboid.py`).

Settings that can be changed:
- `VISUALIZE` in `preprocess_run.py`: set to `True` to open interactive 3D windows (needs a screen and `open3d`).
- `VISUALIZE_ONLY` in `inference.py`: set to `True` to skip the inference and only show saved predictions.
- `AUTO_VISUALIZE`, `VISUALIZE_3D` and `CUBOID_IDX` in `visualize_cuboid.py`.

## Requirements

Python 3.10. Install the packages with

```
pip install -r requirements.txt
```

The versions in `requirements.txt` are from the local development environment. A GPU (CUDA-enabled PyTorch) is recommended for training. `open3d` is only needed for the optional interactive visualization.

## Reference

Stephens, D., Smith, A., Redfern, T., Talbot, A., Lessnoff, A., & Dempsey, K. (2020). Using three dimensional convolutional neural networks for denoising echosounder point cloud data. Applied Computing and Geosciences, 5, 100016. https://doi.org/10.1016/J.ACAGS.2019.100016

## License

MIT License

Copyright (c) 2026 nylanben

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
[MIT License — see the LICENSE file.]


This README was written with the assistance of generative AI (Claude Sonnet 5.5).
