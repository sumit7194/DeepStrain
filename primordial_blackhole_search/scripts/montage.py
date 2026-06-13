"""Visual sanity check of the built dataset: look at what the model will see.

Top two rows: positives across the SNR range (label: chirp mass, in-window SNR).
Bottom two rows: real-noise negatives (glitches and all).

Run:  uv run python scripts/montage.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C


def main() -> None:
    x = np.load(C.SHARD_DIR / "x_val.npy", mmap_mode="r")
    meta = pd.read_parquet(C.SHARD_DIR / "meta_val.parquet")

    pos = meta[meta.label == 1].sort_values("in_window_snr")
    # spread positives across the SNR range, weakest to loudest
    pick_pos = pos.iloc[np.linspace(0, len(pos) - 1, 8).astype(int)]
    pick_neg = meta[meta.label == 0].sample(8, random_state=0)

    fig, axes = plt.subplots(4, 4, figsize=(18, 10))
    for ax, (_, r) in zip(axes[:2].flat, pick_pos.iterrows()):
        ax.imshow(np.asarray(x[int(r.row)], dtype=np.float32), aspect="auto",
                  origin="lower", cmap="viridis", vmin=0, vmax=3)
        ax.set_title(f"POS Mc={r.chirp_mass:.2f} snr={r.in_window_snr:.1f}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax, (_, r) in zip(axes[2:].flat, pick_neg.iterrows()):
        ax.imshow(np.asarray(x[int(r.row)], dtype=np.float32), aspect="auto",
                  origin="lower", cmap="viridis", vmin=0, vmax=3)
        ax.set_title("NEG (real noise)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("model inputs: 256 s log-f spectrograms, 50-1024 Hz")
    fig.tight_layout()
    C.RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(C.RESULTS_DIR / "dataset_montage.png", dpi=110)
    print(f"wrote {C.RESULTS_DIR / 'dataset_montage.png'}")


if __name__ == "__main__":
    main()
