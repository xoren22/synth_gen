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


def load_sample(sample_dir):
    """Load arrays and metadata from a sample directory."""
    npz_files = [f for f in os.listdir(sample_dir) if f.endswith(".npz")]
    json_files = [f for f in os.listdir(sample_dir) if f.endswith(".json")]
    if not npz_files or not json_files:
        return None, None
    data = np.load(os.path.join(sample_dir, npz_files[0]))
    with open(os.path.join(sample_dir, json_files[0])) as f:
        meta = json.load(f)
    return data, meta


def plot_samples(run_dir, sample_dirs, out_path):
    """Create a figure with subplots for each sample showing pathloss, mask, and info."""
    n = len(sample_dirs)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n), squeeze=False)
    fig.suptitle(f"Synthetic Samples from: {os.path.basename(run_dir)}", fontsize=14, y=1.0)

    for i, sdir in enumerate(sample_dirs):
        data, meta = load_sample(os.path.join(run_dir, sdir))
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
        ax.set_title(f"Pathloss  [{sdir}]", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dB")

        # --- Mask + transmittance overlay ---
        ax = axes[i, 1]
        rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
        rgb[..., 0] = (transmittance > 0).astype(np.float32)  # walls in red
        rgb[..., 1] = mask.astype(np.float32) * 0.5            # interior in green
        rgb[..., 2] = (transmittance > 0).astype(np.float32) * 0.3
        ax.imshow(rgb, interpolation="nearest")
        ax.plot(ant_x, ant_y, "c+", markersize=10, markeredgewidth=2)
        ax.set_title(f"Mask + Walls  [{sdir}]", fontsize=10)

        # --- Info text panel ---
        ax = axes[i, 2]
        ax.set_axis_off()
        info_lines = [
            f"Sample: {meta.get('sample_name', sdir)}",
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

    all_samples = sorted([
        d for d in os.listdir(run_dir)
        if os.path.isdir(os.path.join(run_dir, d))
        and any(f.endswith(".npz") for f in os.listdir(os.path.join(run_dir, d)))
    ])

    if not all_samples:
        print("No sample directories found.", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    chosen = random.sample(all_samples, min(args.n, len(all_samples)))
    chosen.sort()

    print(f"Found {len(all_samples)} samples, visualizing {len(chosen)}: {chosen}")

    out_path = args.out or os.path.join(run_dir, "samples_overview.png")
    plot_samples(run_dir, chosen, out_path)


if __name__ == "__main__":
    main()
