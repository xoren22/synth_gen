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
import matplotlib.gridspec as gridspec
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


def _pick_random_samples_fast(run_dir, n, rng):
    """Pick n random sample paths by navigating the directory structure directly.

    Handles both worker-nested (run_dir/workers/worker_*/out/bucket/s*.npz)
    and flat (run_dir/bucket/s*.npz) layouts.
    """
    # Discover all bucket directories containing samples
    bucket_dirs = []
    workers_dir = os.path.join(run_dir, "workers")
    if os.path.isdir(workers_dir):
        for w in os.listdir(workers_dir):
            out_dir = os.path.join(workers_dir, w, "out")
            if not os.path.isdir(out_dir):
                continue
            for b in os.listdir(out_dir):
                bp = os.path.join(out_dir, b)
                if os.path.isdir(bp):
                    bucket_dirs.append(bp)
    else:
        for entry in os.listdir(run_dir):
            bp = os.path.join(run_dir, entry)
            if os.path.isdir(bp):
                bucket_dirs.append(bp)

    if not bucket_dirs:
        return []

    chosen = []
    attempts = 0
    seen_paths = set()
    while len(chosen) < n and attempts < n * 20:
        attempts += 1
        bucket = rng.choice(bucket_dirs)
        try:
            files = [f for f in os.listdir(bucket)
                     if f.startswith("s") and f.endswith(".npz")]
        except OSError:
            continue
        if not files:
            continue
        f = rng.choice(files)
        path = os.path.join(bucket, f)
        if path not in seen_paths:
            seen_paths.add(path)
            chosen.append(path)

    chosen.sort()
    return chosen


def plot_samples(run_dir, npz_paths, out_path):
    """Create a figure with subplots for each sample showing all features."""
    n = len(npz_paths)
    # 6 columns: pathloss, transmittance, reflectance, mask, radiation pattern, info
    fig = plt.figure(figsize=(28, 5.5 * n))
    fig.suptitle(f"Synthetic Samples from: {os.path.basename(run_dir)}", fontsize=14, y=1.0)
    outer = gridspec.GridSpec(n, 6, figure=fig, width_ratios=[1, 1, 1, 1, 0.8, 0.7],
                              wspace=0.35, hspace=0.4)

    for i, npz_path in enumerate(npz_paths):
        data, meta = load_sample(npz_path)
        if data is None:
            for j in range(6):
                ax = fig.add_subplot(outer[i, j])
                ax.text(0.5, 0.5, "Failed to load", ha="center", va="center")
                ax.set_axis_off()
            continue

        pathloss = np.array(data["pathloss"], dtype=np.float32)
        mask = np.array(data["mask"], dtype=np.uint8)
        transmittance = np.array(data["transmittance"], dtype=np.float32)
        reflectance = np.array(data["reflectance"], dtype=np.float32)
        rad_pattern_db = np.array(data["radiation_pattern_db"], dtype=np.float32)

        ant = meta["antenna"]
        ant_x, ant_y = ant["x_px"], ant["y_px"]
        freq = meta["frequency_mhz"]
        hw = meta["shape_hw"]
        canvas = meta.get("canvas", {})
        pix = meta.get("pixel_size_m", 0.25)
        sample_name = meta.get("sample_name", "")

        # --- Pathloss heatmap ---
        ax = fig.add_subplot(outer[i, 0])
        pl_display = pathloss.copy()
        pl_display[pl_display >= 32000] = np.nan
        im = ax.imshow(pl_display, cmap="inferno_r", interpolation="nearest")
        ax.plot(ant_x, ant_y, "c+", markersize=10, markeredgewidth=2)
        ax.set_title(f"Pathloss (dB)", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # --- Transmittance map ---
        ax = fig.add_subplot(outer[i, 1])
        im = ax.imshow(transmittance, cmap="hot", interpolation="nearest")
        ax.plot(ant_x, ant_y, "c+", markersize=10, markeredgewidth=2)
        ax.set_title("Transmittance", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # --- Reflectance map ---
        ax = fig.add_subplot(outer[i, 2])
        im = ax.imshow(reflectance, cmap="viridis", interpolation="nearest")
        ax.plot(ant_x, ant_y, "c+", markersize=10, markeredgewidth=2)
        ax.set_title("Reflectance", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # --- Mask overlay ---
        ax = fig.add_subplot(outer[i, 3])
        rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
        rgb[..., 0] = (transmittance > 0).astype(np.float32)
        rgb[..., 1] = mask.astype(np.float32) * 0.5
        rgb[..., 2] = (transmittance > 0).astype(np.float32) * 0.3
        ax.imshow(rgb, interpolation="nearest")
        ax.plot(ant_x, ant_y, "c+", markersize=10, markeredgewidth=2)
        ax.set_title("Mask + Walls", fontsize=10)

        # --- Radiation pattern (polar) ---
        ax = fig.add_subplot(outer[i, 4], projection="polar")
        angles_deg = np.arange(360)
        angles_rad = np.deg2rad(angles_deg)
        # Plot gain (close the loop by appending first element)
        gain = np.append(rad_pattern_db, rad_pattern_db[0])
        theta = np.append(angles_rad, angles_rad[0])
        ax.plot(theta, gain, "b-", linewidth=1.2)
        ax.fill(theta, gain, alpha=0.15, color="b")
        azimuth_rad = np.deg2rad(ant.get("azimuth_deg", 0))
        ax.axvline(azimuth_rad, color="r", linewidth=1.5, linestyle="--", label="boresight")
        ax.set_title("Radiation Pattern (dB)", fontsize=10, pad=15)
        ax.set_rticks([])
        ax.legend(loc="lower right", fontsize=7)

        # --- Info text panel ---
        ax = fig.add_subplot(outer[i, 5])
        ax.set_axis_off()
        iso = ant.get("is_isotropic", False)
        pat_model = ant.get("pattern_model", "?")
        pat_style = ant.get("pattern_style", "?")
        info_lines = [
            f"Sample: {sample_name}",
            f"Shape: {hw[0]}x{hw[1]} px",
            f"Pixel: {pix} m",
            f"Canvas: {canvas.get('width_m', 0):.1f}x"
            f"{canvas.get('height_m', 0):.1f} m",
            "",
            f"Freq: {freq} MHz",
            f"Antenna: ({ant_x}, {ant_y}) px",
            f"Azimuth: {ant.get('azimuth_deg', 0):.1f} deg",
            f"Isotropic: {iso}",
            f"Model: {pat_model}",
            f"Style: {pat_style}",
            "",
            f"PL range: {np.nanmin(pl_display):.1f}"
            f" to {np.nanmax(pl_display):.1f} dB",
            f"Walls: {int((transmittance > 0).sum())} px",
            f"Interior: {int(mask.sum())} px",
        ]
        ax.text(
            0.05, 0.95, "\n".join(info_lines),
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )

    fig.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.04)
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

    rng = random.Random(args.seed)
    chosen = _pick_random_samples_fast(run_dir, args.n, rng)

    if not chosen:
        print("No sample NPZ files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Visualizing {len(chosen)} samples")

    out_path = args.out or os.path.join(run_dir, "samples_overview.png")
    plot_samples(run_dir, chosen, out_path)


if __name__ == "__main__":
    main()
