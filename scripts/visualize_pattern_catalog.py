#!/usr/bin/env python3
"""Build a visual catalog of antenna radiation pattern families and controls.

This script generates multiple figures:
  1) latent_fourier styles overview
  2) latent_fourier petal-order sweep
  3) gaussian_lobes control sweeps

The plots use polar coordinates and show "relative strength":
  strength = max_loss_db - directional_loss_db
So stronger directions extend farther from center.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from antenna_pattern import RadiationPatternConfig, generate_radiation_pattern


BASE_LATENT_CFG = RadiationPatternConfig(
    num_angles=360,
    isotropic_probability=0.0,
    max_loss_db=20.0,
    pattern_model="latent_fourier",
    latent_dim_min=10,
    latent_dim_max=10,
    fourier_order_min=14,
    fourier_order_max=14,
    style_mode="random",
    petal_order_min=3,
    petal_order_max=8,
    symmetry_mode="none",
)

BASE_GAUSS_CFG = RadiationPatternConfig(
    num_angles=360,
    isotropic_probability=0.0,
    max_loss_db=20.0,
    pattern_model="gaussian_lobes",
    lobe_count_min=2,
    lobe_count_max=6,
    lobe_width_deg_min=18.0,
    lobe_width_deg_max=80.0,
    smooth_sigma_deg_min=2.0,
    smooth_sigma_deg_max=8.0,
    symmetry_mode="none",
)


def _latent_style_cfg(style: str, **overrides) -> RadiationPatternConfig:
    style = style.strip().lower()
    cfg = replace(BASE_LATENT_CFG, style_mode=style)
    return replace(cfg, **overrides)


def _sample_pattern(cfg: RadiationPatternConfig, seed: int):
    rng = np.random.default_rng(seed)
    return generate_radiation_pattern(rng, cfg)


def _plot_pattern(ax, sample, max_loss_db: float):
    gain_db = np.asarray(sample.losses_db, dtype=np.float64)
    loss_db = -gain_db
    strength = np.clip(max_loss_db - loss_db, 0.0, max_loss_db)

    theta = np.deg2rad(np.arange(strength.size, dtype=np.float64))
    theta = np.concatenate([theta, theta[:1]])
    strength = np.concatenate([strength, strength[:1]])

    ax.plot(theta, strength, lw=1.35, color="#0d3b66")
    az = np.deg2rad(float(sample.azimuth_deg))
    ax.plot([az, az], [0.0, max_loss_db], lw=0.9, ls="--", color="#ee964b")

    ax.set_ylim(0.0, max_loss_db)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["0", "90", "180", "270"], fontsize=7)
    ax.set_yticks([0.0, max_loss_db * 0.5, max_loss_db])
    ax.set_yticklabels(["weak", "mid", "strong"], fontsize=6)
    ax.grid(alpha=0.35)

    ax.text(
        0.02,
        0.02,
        f"az={sample.azimuth_deg:.0f}  sym={sample.symmetry}",
        transform=ax.transAxes,
        fontsize=6,
        color="#444444",
    )


def _latent_style_overview(out_dir: str, seed: int, dpi: int):
    styles = ["front_back", "bidirectional", "petal", "ripple"]
    variants = [
        ("baseline", dict()),
        ("higher K", dict(fourier_order_min=28, fourier_order_max=28)),
        ("higher d", dict(latent_dim_min=24, latent_dim_max=24)),
        ("forced sym x", dict(symmetry_mode="x")),
    ]

    fig, axes = plt.subplots(
        nrows=len(styles),
        ncols=len(variants),
        figsize=(15.0, 13.2),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    max_loss = float(BASE_LATENT_CFG.max_loss_db)

    for r, style in enumerate(styles):
        for c, (label, overrides) in enumerate(variants):
            cfg = _latent_style_cfg(style, **overrides)
            sample = _sample_pattern(cfg, seed + 1000 * r + 37 * c)
            ax = axes[r, c]
            _plot_pattern(ax, sample, max_loss_db=max_loss)
            if r == 0:
                ax.set_title(label, fontsize=10, pad=14)
            if c == 0:
                ax.text(
                    -0.33,
                    0.5,
                    style,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                    color="#111111",
                )

    fig.suptitle(
        "latent_fourier style map (rows=style, cols=config variant)",
        fontsize=13,
    )
    out_path = os.path.join(out_dir, "pattern_catalog_latent_styles.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _latent_petal_sweep(out_dir: str, seed: int, dpi: int):
    variants = [
        ("petal 3..4", dict(petal_order_min=3, petal_order_max=4)),
        ("petal 5..6", dict(petal_order_min=5, petal_order_max=6)),
        ("petal 8..10", dict(petal_order_min=8, petal_order_max=10)),
        ("petal 12..14", dict(petal_order_min=12, petal_order_max=14)),
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(variants),
        figsize=(14.6, 7.6),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    max_loss = float(BASE_LATENT_CFG.max_loss_db)

    row_overrides = [
        dict(
            fourier_order_min=14,
            fourier_order_max=14,
            latent_dim_min=10,
            latent_dim_max=10,
        ),
        dict(
            fourier_order_min=26,
            fourier_order_max=26,
            latent_dim_min=20,
            latent_dim_max=20,
        ),
    ]
    row_labels = ["moderate complexity", "high complexity"]

    for r, row_cfg_overrides in enumerate(row_overrides):
        for c, (label, petal_cfg) in enumerate(variants):
            cfg = _latent_style_cfg("petal", **row_cfg_overrides, **petal_cfg)
            sample = _sample_pattern(cfg, seed + 5000 + 1000 * r + 31 * c)
            ax = axes[r, c]
            _plot_pattern(ax, sample, max_loss_db=max_loss)
            if r == 0:
                ax.set_title(label, fontsize=10, pad=14)
            if c == 0:
                ax.text(
                    -0.35,
                    0.5,
                    row_labels[r],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=9,
                    color="#111111",
                )

    fig.suptitle("latent_fourier petal sweep (petal_order and global complexity)", fontsize=13)
    out_path = os.path.join(out_dir, "pattern_catalog_latent_petal_sweep.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _gaussian_control_sweep(out_dir: str, seed: int, dpi: int):
    variants = [
        ("few + wide + smooth", dict(lobe_count_min=2, lobe_count_max=2, lobe_width_deg_min=55.0, lobe_width_deg_max=80.0, smooth_sigma_deg_min=8.0, smooth_sigma_deg_max=10.0)),
        ("few + narrow + sharp", dict(lobe_count_min=2, lobe_count_max=2, lobe_width_deg_min=8.0, lobe_width_deg_max=16.0, smooth_sigma_deg_min=0.0, smooth_sigma_deg_max=0.0)),
        ("many + narrow + sharp", dict(lobe_count_min=8, lobe_count_max=8, lobe_width_deg_min=6.0, lobe_width_deg_max=12.0, smooth_sigma_deg_min=0.0, smooth_sigma_deg_max=1.0)),
        ("many + wide + smooth", dict(lobe_count_min=8, lobe_count_max=8, lobe_width_deg_min=30.0, lobe_width_deg_max=70.0, smooth_sigma_deg_min=6.0, smooth_sigma_deg_max=12.0)),
    ]

    sym_rows = [("none", "symmetry none"), ("xy", "symmetry xy")]

    fig, axes = plt.subplots(
        nrows=len(sym_rows),
        ncols=len(variants),
        figsize=(15.0, 8.1),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    max_loss = float(BASE_GAUSS_CFG.max_loss_db)

    for r, (sym, row_label) in enumerate(sym_rows):
        for c, (label, variant_cfg) in enumerate(variants):
            cfg = replace(BASE_GAUSS_CFG, symmetry_mode=sym, **variant_cfg)
            sample = _sample_pattern(cfg, seed + 9000 + 1000 * r + 23 * c)
            ax = axes[r, c]
            _plot_pattern(ax, sample, max_loss_db=max_loss)
            if r == 0:
                ax.set_title(label, fontsize=10, pad=14)
            if c == 0:
                ax.text(
                    -0.35,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=9,
                    color="#111111",
                )

    fig.suptitle("gaussian_lobes controls (count, width, smoothing, symmetry)", fontsize=13)
    out_path = os.path.join(out_dir, "pattern_catalog_gaussian_lobes.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Visual catalog for antenna radiation pattern configs")
    parser.add_argument(
        "--out-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pattern_catalog")),
        help="Output directory for generated figures",
    )
    parser.add_argument("--seed", type=int, default=20260309, help="Base seed")
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    outputs = [
        _latent_style_overview(out_dir, seed=args.seed, dpi=int(args.dpi)),
        _latent_petal_sweep(out_dir, seed=args.seed, dpi=int(args.dpi)),
        _gaussian_control_sweep(out_dir, seed=args.seed, dpi=int(args.dpi)),
    ]

    print("Generated pattern catalog:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
