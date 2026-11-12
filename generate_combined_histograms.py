import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

RAW_DIR  = Path("path_to_realdata")
SAVE_DIR = Path("path_to_output")

datasets = {
    "aiKK_a02_incl_rejected_pos_down": "aiKK_a02",
    "bhk_a2_ext_incl_rejected_pos_down": "BBHK_A2",
    "broa4_xyzstat_pos_down": "Bro_a4",
    "HK_K02_NS_ext_incl_rejected_pos_down_new": "HK_K02",
    "m01sh_xyzstat_pos_down": "m01sh",
    "A_xyz_status_pos_down": "A_xyz",
}

depth_ranges = {
    "aiKK_a02": (-1, 36),
    "BBHK_A2":   (4, 60),
    "Bro_a4":   (0, 22),
    "HK_K02":   (-1, 32),
    "m01sh":    (-1, 21),
    "A_xyz":    (0, 63),
}

def dynamic_formatter(ax):
    max_val = ax.get_ylim()[1]
    if max_val >= 1e8:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1e8:.1f}'))
        ax.set_ylabel('Number of soundings (×10⁸)', fontsize=12)
    elif max_val >= 1e7:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1e7:.1f}'))
        ax.set_ylabel('Number of soundings (×10⁷)', fontsize=12)
    elif max_val >= 1e6:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1e6:.1f}'))
        ax.set_ylabel('Number of soundings (×10⁶)', fontsize=12)
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
        ax.set_ylabel('Number of soundings', fontsize=12)

BIN_WIDTH = 0.5

fig, axes = plt.subplots(3, 2, figsize=(14, 20))
axes = axes.ravel()

for i, (ds_id, ds_name) in enumerate(datasets.items()):
    print(f"Processing {ds_name}...")
    raw_file = RAW_DIR / ds_id / f"{ds_id}.txt"

    if not raw_file.exists():
        print(f"  File not found: {raw_file}")
        continue

    d_min, d_max = depth_ranges[ds_name]
    bin_edges = np.arange(d_min, d_max + BIN_WIDTH, BIN_WIDTH)

    counts_all      = np.zeros(len(bin_edges) - 1)
    counts_rejected = np.zeros(len(bin_edges) - 1)

    for chunk_num, chunk in enumerate(pd.read_csv(
            raw_file, sep=r"\s+", header=None,
            usecols=[2, 3], chunksize=500000, engine="python")):

        chunk.columns = ["Z", "Status"]
        chunk["Status"] = chunk["Status"].astype(str).str.lower()
        mask = chunk["Status"] == "accepted"

        counts_all      += np.histogram(chunk["Z"].values, bins=bin_edges)[0]
        counts_rejected += np.histogram(chunk["Z"].values[~mask], bins=bin_edges)[0]

        if chunk_num % 20 == 0:
            print(f"  Chunk {chunk_num} done")

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = bin_edges[1] - bin_edges[0]

    ax = axes[i]
    ax.bar(bin_centers, counts_all,      width=width, alpha=0.5, label='All soundings', color='gray')
    ax.bar(bin_centers, counts_rejected, width=width, alpha=0.7, label='Rejected',      color='red')
    ax.set_xlabel('Depth (m)', fontsize=12)
    ax.set_title(f'Depth distribution — {ds_name}', fontsize=13, pad=10)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=10)
    dynamic_formatter(ax)

    print(f"  {ds_name} done!")

plt.tight_layout(pad=3.0)
plt.savefig(SAVE_DIR / "depth_distributions_all.png", dpi=150)
plt.show()
print("Saved!")