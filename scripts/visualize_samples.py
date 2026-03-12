#!/usr/bin/env python3
"""Visualize generated synthetic samples as a grid of plots with metadata text."""

import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _decode_meta_json_from_npz(data):
    """Decode embedded metadata from a loaded npz payload."""
    if "meta_json" not in data.files:
        return None

    raw = data["meta_json"]
    if isinstance(raw, np.ndarray):
        if raw.ndim == 0:
            raw = raw.item()
        else:
            raw = "".join(str(x) for x in raw.tolist())

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    elif not isinstance(raw, str):
        raw = str(raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def load_sample(npz_path):
    """Load arrays and metadata from an NPZ file path.

    Metadata is read from the embedded 'meta_json' key.
    """
    data = np.load(npz_path)
    meta = _decode_meta_json_from_npz(data)
    return data, meta


def plot_samples(run_dir, npz_paths, out_path):
    """Create a figure with subplots for each sample showing pathloss, mask, and info."""
    n = len(npz_paths)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n), squeeze=False)
    fig.suptitle(f"Synthetic Samples from: {os.path.basename(run_dir)}", fontsize=14, y=1.0)

    for i, npz_path in enumerate(npz_paths):
        data, meta = load_sample(npz_path)
        if data is None:
            for j in range(3):
                axes[i, j].text(0.5, 0.5, "Failed to load", ha="center", va="center")
                axes[i, j].set_axis_off()
            continue

        pathloss = np.array(data["pathloss"], dtype=np.float32)
        mask = np.array(data["mask"], dtype=np.uint8)
        transmittance = np.array(data["transmittance"], dtype=np.float32)

        ant_x = meta["antenna"]["x_px"]
        ant_y = meta["antenna"]["y_px"]
        freq = meta["frequency_mhz"]
        hw = meta["shape_hw"]
        canvas = meta.get("canvas", {})
        pix = meta.get("pixel_size_m", 0.25)

        # --- Pathloss heatmap ---
        ax = axes[i, 0]
        pl_display = pathloss.copy()
        pl_display[pl_display >= 32000] = np.nan
        im = ax.imshow(pl_display, cmap="inferno_r", interpolation="nearest")
        ax.plot(ant_x, ant_y, "c+", markersize=10, markeredgewidth=2)
        ax.set_title(f"Pathloss  [{meta.get('sample_name', '')}]", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dB")

        # --- Mask + transmittance overlay ---
        ax = axes[i, 1]
        rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
        rgb[..., 0] = (transmittance > 0).astype(np.float32)  # walls in red
        rgb[..., 1] = mask.astype(np.float32) * 0.5            # interior in green
        rgb[..., 2] = (transmittance > 0).astype(np.float32) * 0.3
        ax.imshow(rgb, interpolation="nearest")
        ax.plot(ant_x, ant_y, "c+", markersize=10, markeredgewidth=2)
        ax.set_title(f"Mask + Walls  [{meta.get('sample_name', '')}]", fontsize=10)

        # --- Info text panel ---
        ax = axes[i, 2]
        ax.set_axis_off()
        info_lines = [
            f"Sample: {meta.get('sample_name', '')}",
            f"Shape: {hw[0]} x {hw[1]} px",
            f"Pixel size: {pix} m",
            f"Canvas: {canvas.get('width_m', 0):.1f} x {canvas.get('height_m', 0):.1f} m",
            f"Frequency: {freq} MHz",
            f"Antenna: ({ant_x}, {ant_y}) px",
            f"Pathloss range: {np.nanmin(pl_display):.1f} - {np.nanmax(pl_display):.1f} dB",
            f"Wall pixels: {int((transmittance > 0).sum())}",
            f"Interior pixels: {int(mask.sum())}",
        ]
        ax.text(
            0.05, 0.95, "\n".join(info_lines),
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )
        ax.set_title("Sample Info", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize synthetic radar samples")
    parser.add_argument("run_dir", help="Path to run directory containing sample subdirs")
    parser.add_argument("--n", type=int, default=10, help="Number of random samples to show")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    parser.add_argument("--out", type=str, default=None, help="Output image path (default: <run_dir>/samples_overview.png)")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        print(f"Error: {run_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Collect all NPZ paths from bucket subdirectories
    all_npz = []
    for bucket in sorted(os.listdir(run_dir)):
        bucket_path = os.path.join(run_dir, bucket)
        if not os.path.isdir(bucket_path):
            continue
        for f in sorted(os.listdir(bucket_path)):
            if f.startswith("s") and f.endswith(".npz"):
                all_npz.append(os.path.join(bucket_path, f))

    if not all_npz:
        print("No sample NPZ files found.", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    chosen = random.sample(all_npz, min(args.n, len(all_npz)))
    chosen.sort()

    print(f"Found {len(all_npz)} samples, visualizing {len(chosen)}")

    out_path = args.out or os.path.join(run_dir, "samples_overview.png")
    plot_samples(run_dir, chosen, out_path)


if __name__ == "__main__":
    main()
