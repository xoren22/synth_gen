#!/usr/bin/env python3
"""Generate random representative samples for each latent_fourier style.

Each sample uses the default RadiationPatternConfig with only style_mode
pinned — all other parameters (K, d, symmetry, eff_max, petal_order, …)
are drawn by the same RNG process used in production.

Produces one PNG per style: a 4×5 grid of 20 polar subplots, each annotated
with its realized parameters.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from antenna_pattern import RadiationPatternConfig, generate_radiation_pattern


BASE_CFG = RadiationPatternConfig(
    isotropic_probability=0.0,
    max_loss_db=20.0,
    # Keep defaults for everything else — same as production.
)


def _sample(style: str, seed: int):
    from dataclasses import replace
    cfg = replace(BASE_CFG, style_mode=style)
    rng = np.random.default_rng(seed)
    return generate_radiation_pattern(rng, cfg)


def _annotation(sample) -> str:
    info = sample.function_info or {}
    K = len(info.get("k", []))
    d = sample.complexity_dim
    sym = sample.symmetry
    eff = info.get("eff_max_db", 0.0)
    parts = [
        f"K={K}  d={d}  sym={sym}",
        f"eff={eff:.1f}dB  az={sample.azimuth_deg:.0f}°",
    ]
    return "\n".join(parts)


def _plot_one(ax, sample, max_loss: float):
    gain_db = np.asarray(sample.losses_db, dtype=np.float64)
    loss_db = -gain_db
    strength = np.clip(max_loss - loss_db, 0.0, max_loss)

    theta = np.deg2rad(np.arange(strength.size, dtype=np.float64))
    theta = np.concatenate([theta, theta[:1]])
    strength = np.concatenate([strength, strength[:1]])

    ax.plot(theta, strength, lw=1.2, color="#0d3b66")
    az = np.deg2rad(float(sample.azimuth_deg))
    ax.plot([az, az], [0.0, max_loss], lw=0.7, ls="--", color="#ee964b")

    ax.set_ylim(0.0, max_loss)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["0", "90", "180", "270"], fontsize=5)
    ax.set_yticks([])
    ax.grid(alpha=0.25)

    ax.text(
        0.5, -0.13,
        _annotation(sample),
        transform=ax.transAxes,
        fontsize=5,
        ha="center", va="top",
        color="#333333",
        family="monospace",
    )


def generate_style_sheet(style: str, out_dir: str, seed: int, dpi: int, n: int = 20):
    nrows, ncols = 4, 5
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(16.0, 13.5),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    max_loss = float(BASE_CFG.max_loss_db)

    for i in range(n):
        r, c = divmod(i, ncols)
        sample = _sample(style, seed + i)
        _plot_one(axes[r, c], sample, max_loss)

    fig.suptitle(
        f"latent_fourier  style={style}  —  {n} random samples (default config)",
        fontsize=13,
    )
    out_path = os.path.join(out_dir, f"pattern_samples_{style}.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Random sample sheets per style")
    parser.add_argument(
        "--out-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pattern_catalog")),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("-n", type=int, default=20, help="Samples per style")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    styles = ["front_back", "bidirectional", "petal", "ripple"]
    for style in styles:
        path = generate_style_sheet(
            style, out_dir,
            seed=args.seed, dpi=args.dpi, n=args.n,
        )
        print(path)


if __name__ == "__main__":
    main()
