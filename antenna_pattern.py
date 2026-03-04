from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RadiationPatternConfig:
    num_angles: int = 360
    isotropic_probability: float = 1.0
    min_loss_db: float = 0.0
    max_loss_db: float = 20.0

    # Main generator model: latent_fourier (default) or gaussian_lobes (legacy fallback).
    pattern_model: str = "latent_fourier"

    # Latent Fourier model controls.
    latent_dim: int = 10
    latent_dim_min: int = 10
    latent_dim_max: int = 10
    fourier_order: int = 14
    dynamic_range_min_fraction: float = 0.35
    style_front_back_prob: float = 0.25
    style_bidirectional_prob: float = 0.25
    style_petal_prob: float = 0.30
    style_ripple_prob: float = 0.20
    petal_order_min: int = 3
    petal_order_max: int = 8

    # Legacy gaussian-lobe controls.
    lobe_count_min: int = 2
    lobe_count_max: int = 6
    lobe_width_deg_min: float = 18.0
    lobe_width_deg_max: float = 80.0
    smooth_sigma_deg_min: float = 2.0
    smooth_sigma_deg_max: float = 8.0

    # Symmetry controls.
    symmetry_probability: float = 0.35
    symmetry_mode: str = "random"  # random | none | x | y | xy
    azimuth_quantization_deg: float = 0.0  # 0 => continuous


@dataclass(frozen=True)
class RadiationPatternSample:
    losses_db: np.ndarray
    is_isotropic: bool
    azimuth_deg: float
    symmetry: str
    model: str
    style: str
    complexity_dim: int


def _wrap_deg(delta: np.ndarray) -> np.ndarray:
    return (delta + 180.0) % 360.0 - 180.0


def _circular_interp(arr: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    n = arr.shape[0]
    pos = (angles_deg % 360.0) * (n / 360.0)
    i0 = np.floor(pos).astype(np.int64) % n
    i1 = (i0 + 1) % n
    t = pos - np.floor(pos)
    return (1.0 - t) * arr[i0] + t * arr[i1]


def evaluate_pattern_db(losses_db: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    return _circular_interp(np.asarray(losses_db, dtype=np.float64), np.asarray(angles_deg, dtype=np.float64))


def _apply_symmetry(losses: np.ndarray, axis_deg: float) -> np.ndarray:
    n = losses.shape[0]
    theta = np.arange(n, dtype=np.float64) * (360.0 / n)
    mirrored = (2.0 * axis_deg - theta) % 360.0
    mirror_vals = _circular_interp(losses, mirrored)
    return 0.5 * (losses + mirror_vals)


def _gaussian_kernel_circular(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0.0:
        return np.array([1.0], dtype=np.float64)
    radius = int(max(1, np.ceil(4.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    s = float(np.sum(k))
    if s <= 0:
        return np.array([1.0], dtype=np.float64)
    return k / s


def _smooth_circular(losses: np.ndarray, sigma_deg: float) -> np.ndarray:
    n = losses.shape[0]
    sigma_bins = float(sigma_deg) * n / 360.0
    kernel = _gaussian_kernel_circular(sigma_bins)
    if kernel.shape[0] == 1:
        return losses
    radius = (kernel.shape[0] - 1) // 2
    out = np.zeros_like(losses, dtype=np.float64)
    for offset, w in enumerate(kernel):
        shift = offset - radius
        out += w * np.roll(losses, shift)
    return out


def _resolve_symmetry_mode(rng: np.random.Generator, cfg: RadiationPatternConfig) -> str:
    mode = cfg.symmetry_mode.lower().strip()
    if mode == "random":
        if rng.random() < float(np.clip(cfg.symmetry_probability, 0.0, 1.0)):
            return str(rng.choice(np.array(["x", "y", "xy"], dtype=object)))
        return "none"
    if mode not in {"none", "x", "y", "xy"}:
        return "none"
    return mode


def _sample_style(rng: np.random.Generator, cfg: RadiationPatternConfig) -> str:
    styles = np.array(["front_back", "bidirectional", "petal", "ripple"], dtype=object)
    probs = np.array(
        [
            float(max(0.0, cfg.style_front_back_prob)),
            float(max(0.0, cfg.style_bidirectional_prob)),
            float(max(0.0, cfg.style_petal_prob)),
            float(max(0.0, cfg.style_ripple_prob)),
        ],
        dtype=np.float64,
    )
    s = float(np.sum(probs))
    if s <= 0.0:
        probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    else:
        probs /= s
    return str(rng.choice(styles, p=probs))


def _style_envelope(rng: np.random.Generator, cfg: RadiationPatternConfig, style: str, k: np.ndarray) -> np.ndarray:
    if style == "front_back":
        env = 1.0 / (1.0 + 0.6 * (k - 1.0) ** 2)
        env[0] *= 1.6
        return env
    if style == "bidirectional":
        env = np.exp(-0.12 * (k - 1.0))
        odd = (k % 2 == 1)
        env[odd] *= 0.18
        env[~odd] *= 1.3
        return env
    if style == "petal":
        m_min = int(max(2, cfg.petal_order_min))
        m_max = int(max(m_min, cfg.petal_order_max))
        m = int(rng.integers(m_min, m_max + 1))
        sigma = float(rng.uniform(0.8, 2.2))
        env = np.exp(-0.5 * ((k - float(m)) / sigma) ** 2)
        env += 0.25 * np.exp(-0.5 * ((k - 2.0 * float(m)) / max(1.0, 1.6 * sigma)) ** 2)
        return env
    # ripple
    return 1.0 / (k ** float(rng.uniform(0.7, 1.3)))


def _map_raw_to_losses(raw: np.ndarray, cfg: RadiationPatternConfig, rng: np.random.Generator) -> np.ndarray:
    min_loss = float(np.clip(cfg.min_loss_db, 0.0, cfg.max_loss_db))
    max_loss = float(max(min_loss + 1e-6, cfg.max_loss_db))
    dyn_lo = float(np.clip(cfg.dynamic_range_min_fraction, 0.05, 1.0))
    dyn = float(rng.uniform(dyn_lo, 1.0))
    eff_max = min_loss + dyn * (max_loss - min_loss)

    raw = np.tanh(raw)
    rmin = float(np.min(raw))
    rmax = float(np.max(raw))
    if rmax > rmin:
        g = (raw - rmin) / (rmax - rmin)
    else:
        g = np.zeros_like(raw, dtype=np.float64) + 0.5
    losses = eff_max - g * (eff_max - min_loss)
    return np.clip(losses, min_loss, max_loss)


def _generate_latent_fourier_pattern(
    rng: np.random.Generator,
    cfg: RadiationPatternConfig,
    azimuth_deg: float,
    symmetry: str,
    latent_dim: int,
) -> tuple[np.ndarray, str]:
    n = int(max(8, cfg.num_angles))
    d = int(max(2, latent_dim))
    K = int(max(1, cfg.fourier_order))
    theta = np.arange(n, dtype=np.float64) * (360.0 / n)

    style = _sample_style(rng, cfg)
    k = np.arange(1, K + 1, dtype=np.float64)
    env = _style_envelope(rng, cfg, style, k)
    env = env / max(1e-12, float(np.linalg.norm(env)))

    j = np.arange(1, d + 1, dtype=np.float64)
    z = rng.normal(0.0, 1.0, size=d)
    z2 = rng.normal(0.0, 1.0, size=d)
    # Deterministic low-rank basis maps latent vectors to Fourier coefficients.
    M_cos = np.cos(2.0 * np.pi * np.outer(k, j) / (2.0 * d + 1.0))
    M_sin = np.sin(2.0 * np.pi * np.outer(k, j + 0.5) / (2.0 * d + 1.0))
    a = (M_cos @ z) / np.sqrt(d)
    b = (M_sin @ z2) / np.sqrt(d)
    a *= env
    b *= env

    rel = np.deg2rad(theta - azimuth_deg)
    if symmetry == "x":
        raw = np.cos(np.outer(rel, k)) @ a
    elif symmetry == "y":
        rel_y = np.deg2rad(theta - ((azimuth_deg + 90.0) % 360.0))
        raw = np.cos(np.outer(rel_y, k)) @ a
    elif symmetry == "xy":
        even_mask = (k % 2.0) == 0.0
        kk = k[even_mask]
        aa = a[even_mask]
        if kk.size == 0:
            kk = np.array([2.0], dtype=np.float64)
            aa = np.array([0.0], dtype=np.float64)
        raw = np.cos(np.outer(rel, kk)) @ aa
    else:
        raw = np.cos(np.outer(rel, k)) @ a + np.sin(np.outer(rel, k)) @ b

    losses = _map_raw_to_losses(raw, cfg, rng)
    return losses, style


def _generate_gaussian_lobe_pattern(rng: np.random.Generator, cfg: RadiationPatternConfig, azimuth_deg: float, symmetry: str) -> tuple[np.ndarray, str]:
    n = int(max(8, cfg.num_angles))
    theta = np.arange(n, dtype=np.float64) * (360.0 / n)

    lmin = int(max(1, cfg.lobe_count_min))
    lmax = int(max(lmin, cfg.lobe_count_max))
    n_lobes = int(rng.integers(lmin, lmax + 1))
    width_min = float(max(1e-3, cfg.lobe_width_deg_min))
    width_max = float(max(width_min, cfg.lobe_width_deg_max))

    gain = np.zeros(n, dtype=np.float64)
    centers = (azimuth_deg + rng.uniform(-180.0, 180.0, size=n_lobes)) % 360.0
    amps = rng.uniform(0.5, 1.0, size=n_lobes)
    widths = rng.uniform(width_min, width_max, size=n_lobes)
    for c, a, w in zip(centers, amps, widths):
        d = _wrap_deg(theta - float(c))
        gain += float(a) * np.exp(-0.5 * (d / float(w)) ** 2)

    gmin = float(np.min(gain))
    gmax = float(np.max(gain))
    if gmax > gmin:
        gain = (gain - gmin) / (gmax - gmin)
    else:
        gain.fill(0.0)

    min_loss = float(np.clip(cfg.min_loss_db, 0.0, cfg.max_loss_db))
    max_loss = float(max(min_loss + 1e-6, cfg.max_loss_db))
    dyn_lo = float(np.clip(cfg.dynamic_range_min_fraction, 0.05, 1.0))
    dyn = float(rng.uniform(dyn_lo, 1.0))
    eff_max = min_loss + dyn * (max_loss - min_loss)
    losses = eff_max - gain * (eff_max - min_loss)

    if symmetry in {"x", "xy"}:
        losses = _apply_symmetry(losses, azimuth_deg)
    if symmetry in {"y", "xy"}:
        losses = _apply_symmetry(losses, (azimuth_deg + 90.0) % 360.0)

    sigma_min = float(max(0.0, cfg.smooth_sigma_deg_min))
    sigma_max = float(max(sigma_min, cfg.smooth_sigma_deg_max))
    losses = _smooth_circular(losses, float(rng.uniform(sigma_min, sigma_max)))
    return np.clip(losses, min_loss, max_loss), "gaussian_lobes"


def generate_radiation_pattern(rng: np.random.Generator, cfg: RadiationPatternConfig) -> RadiationPatternSample:
    p_iso = float(np.clip(cfg.isotropic_probability, 0.0, 1.0))
    n = int(max(8, cfg.num_angles))
    model = str(cfg.pattern_model).strip().lower()
    if model not in {"latent_fourier", "gaussian_lobes"}:
        model = "latent_fourier"
    dmin = int(max(2, cfg.latent_dim_min))
    dmax = int(max(dmin, cfg.latent_dim_max))
    sampled_d = int(rng.integers(dmin, dmax + 1))

    if rng.random() < p_iso:
        return RadiationPatternSample(
            losses_db=np.zeros(n, dtype=np.float32),
            is_isotropic=True,
            azimuth_deg=0.0,
            symmetry="none",
            model=model,
            style="isotropic",
            complexity_dim=sampled_d,
        )

    az = float(rng.uniform(0.0, 360.0))
    if cfg.azimuth_quantization_deg > 0:
        q = float(cfg.azimuth_quantization_deg)
        az = float((np.round(az / q) * q) % 360.0)

    symmetry = _resolve_symmetry_mode(rng, cfg)
    if model == "gaussian_lobes":
        losses, style = _generate_gaussian_lobe_pattern(rng, cfg, azimuth_deg=az, symmetry=symmetry)
        complexity_dim = int(max(1, cfg.lobe_count_max))
    else:
        losses, style = _generate_latent_fourier_pattern(
            rng,
            cfg,
            azimuth_deg=az,
            symmetry=symmetry,
            latent_dim=sampled_d,
        )
        complexity_dim = sampled_d

    return RadiationPatternSample(
        losses_db=np.asarray(losses, dtype=np.float32),
        is_isotropic=False,
        azimuth_deg=az,
        symmetry=symmetry,
        model=model,
        style=style,
        complexity_dim=complexity_dim,
    )
