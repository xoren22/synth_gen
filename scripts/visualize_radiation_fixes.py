#!/usr/bin/env python3
"""
Generate non-isotropic samples and produce a diagnostic figure.

Each row is one sample with three panels (+ dedicated colorbars):
  Left:   Pattern+FSPL map  =  -radiation_pattern(angle(x,y)) + FSPL(dist(x,y), freq)
  Middle: Output map         =  Approx ray-tracing result (pathloss)
  Right:  Difference         =  Left - Middle  (diverging colormap, centered at 0)

Antenna location is marked with a hollow circle.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("NUMBA_THREADING_LAYER", "tbb")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from antenna_pattern import RadiationPatternConfig
from approx import Approx, _build_pixel_initial_loss_map
from generate import build_sample_from_generated
from room_generator import generate_floor_scene


MAX_LOSS = 32000.0


def fspl(dist_m: np.ndarray, freq_mhz: float, min_dist_m: float = 0.125) -> np.ndarray:
    d = np.maximum(dist_m.astype(np.float64), float(min_dist_m))
    return 20.0 * np.log10(d) + 20.0 * np.log10(freq_mhz) - 27.55


def pattern_plus_fspl(sample):
    """Left panel: radiation_pattern loss (non-negative after fix) + FSPL per pixel."""
    pat_loss = _build_pixel_initial_loss_map(sample)  # non-negative after sign fix
    H, W = int(sample.H), int(sample.W)
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64),
                          np.arange(W, dtype=np.float64), indexing="ij")
    dist_m = np.hypot(xx - float(sample.x_ant), yy - float(sample.y_ant)) * float(sample.pixel_size)
    fspl_map = fspl(dist_m, float(sample.freq_MHz))
    return pat_loss + fspl_map


def output_map(sample, engine):
    result = engine.approximate(sample, max_trans=10, max_refl=5)
    return result.numpy().astype(np.float64)


def generate_samples(n_samples=6, seed_base=1000):
    pattern_cfg = RadiationPatternConfig(
        num_angles=360,
        isotropic_probability=0.0,
        max_loss_db=20.0,
        pattern_model="latent_fourier",
        latent_dim_min=5,
        latent_dim_max=12,
    )

    samples = []
    infos = []
    for i in range(n_samples):
        seed = seed_base + i
        np.random.seed(seed)
        mask, normals, scene, reflectance, transmittance, dist_map = generate_floor_scene(
            seed=seed, freq_min=800, freq_max=3000,
        )
        sample = build_sample_from_generated(
            mask, normals, scene, reflectance, transmittance, dist_map,
            pattern_cfg=pattern_cfg,
            pattern_seed=seed * 7 + 13,
        )
        infos.append(scene.get("antenna_pattern", {}))
        samples.append(sample)
    return samples, infos


def main():
    parser = argparse.ArgumentParser(description="Pattern+FSPL vs Approx output with raw difference")
    parser.add_argument("--n", type=int, default=6, help="Number of samples")
    parser.add_argument("--seed", type=int, default=1000, help="Base seed")
    parser.add_argument("--cmap-main", default="viridis")
    parser.add_argument("--cmap-diff", default="coolwarm")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = args.out or os.path.join(os.path.dirname(__file__), "..", "radiation_fix_verification.png")
    out_path = os.path.abspath(out_path)

    print(f"Generating {args.n} non-isotropic samples ...")
    samples, infos = generate_samples(n_samples=args.n, seed_base=args.seed)

    print("Running Approx (numba warmup on first sample) ...")
    engine = Approx(method="combined")

    rows = []
    for sample, info in zip(samples, infos):
        left = pattern_plus_fspl(sample)
        mid = output_map(sample, engine)

        # Mask out unreachable pixels for display
        left_d = left.copy()
        left_d[left_d >= MAX_LOSS] = np.nan
        mid_d = mid.copy()
        mid_d[mid_d >= MAX_LOSS] = np.nan

        diff = left_d - mid_d
        valid = np.isfinite(diff)
        rmse = np.sqrt(np.nanmean(np.square(diff))) if np.any(valid) else float("nan")
        print(f"  style={info.get('style','?'):14s}  sym={info.get('symmetry','?'):4s}  "
              f"az={sample.azimuth:6.1f}°  freq={int(sample.freq_MHz)}MHz  RMSE={rmse:.2f}")

        rows.append((sample, info, left_d, mid_d, diff))

    # ----- plot -----
    n = len(rows)
    fig_h = max(2.9 * n, 8)
    fig, axes = plt.subplots(
        n, 6,
        figsize=(19, fig_h),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 0.035, 1.0, 0.035, 1.0, 0.035]},
    )
    if n == 1:
        axes = np.array([axes])

    marker_kw = dict(marker="o", markersize=4, markerfacecolor="none",
                     markeredgecolor="red", markeredgewidth=1.0)

    for i, (sample, info, left_d, mid_d, diff) in enumerate(rows):
        ax_l, cax_l, ax_m, cax_m, ax_d, cax_d = axes[i]
        ant_x, ant_y = float(sample.x_ant), float(sample.y_ant)
        style = info.get("style", "?")
        sym = info.get("symmetry", "none")
        az = sample.azimuth
        freq = int(sample.freq_MHz)

        # Left: Pattern + FSPL
        im_l = ax_l.imshow(left_d, cmap=args.cmap_main,
                           vmin=float(np.nanmin(left_d)), vmax=float(np.nanmax(left_d)),
                           origin="upper")
        ax_l.plot(ant_x, ant_y, **marker_kw)
        ax_l.set_title(f"Pattern+FSPL  [{style}, {sym}, az={az:.0f}°]", fontsize=9)
        ax_l.axis("off")
        cb_l = fig.colorbar(im_l, cax=cax_l)
        cb_l.set_label("Pattern + FSPL (dB)")

        # Middle: Output
        im_m = ax_m.imshow(mid_d, cmap=args.cmap_main,
                           vmin=float(np.nanmin(mid_d)), vmax=float(np.nanmax(mid_d)),
                           origin="upper")
        ax_m.plot(ant_x, ant_y, **marker_kw)
        ax_m.set_title(f"Approx Output  [{freq}MHz]", fontsize=9)
        ax_m.axis("off")
        cb_m = fig.colorbar(im_m, cax=cax_m)
        cb_m.set_label("Output (dB)")

        # Right: Difference
        vmax_abs = float(np.nanmax(np.abs(diff)))
        if vmax_abs == 0.0:
            vmax_abs = 1.0
        im_d = ax_d.imshow(diff, cmap=args.cmap_diff,
                           vmin=-vmax_abs, vmax=vmax_abs, origin="upper")
        ax_d.plot(ant_x, ant_y, **marker_kw)
        ax_d.set_title("Difference (left − right)", fontsize=9)
        ax_d.axis("off")
        cb_d = fig.colorbar(im_d, cax=cax_d)
        cb_d.set_label("Difference (dB)")

    fig.suptitle("Pattern+FSPL vs Approx Output — Radiation Pattern Fix Verification", fontsize=13)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
