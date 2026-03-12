from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RadiationPatternConfig:
    isotropic_probability: float = 0.25
    max_loss_db: float = 40.0

    # Latent Fourier model controls.
    latent_dim_min: int = 8
    latent_dim_max: int = 20
    fourier_order_min: int = 10
    fourier_order_max: int = 24
    style_mode: str = "random"  # random | front_back | bidirectional | petal | ripple
    petal_order_min: int = 3
    petal_order_max: int = 12

    # Symmetry controls.
    symmetry_mode: str = "random"  # random | none | x | y | xy


@dataclass(frozen=True)
class RadiationPatternSample:
    losses_db: np.ndarray
    is_isotropic: bool
    azimuth_deg: float
    symmetry: str
    model: str
    style: str
    complexity_dim: int
    function_info: dict | None = None


def _circular_interp(arr: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    n = arr.shape[0]
    pos = (angles_deg % 360.0) * (n / 360.0)
    i0 = np.floor(pos).astype(np.int64) % n
    i1 = (i0 + 1) % n
    t = pos - np.floor(pos)
    return (1.0 - t) * arr[i0] + t * arr[i1]


def evaluate_pattern_db(losses_db: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    return _circular_interp(np.asarray(losses_db, dtype=np.float64), np.asarray(angles_deg, dtype=np.float64))


def _resolve_symmetry_mode(rng: np.random.Generator, cfg: RadiationPatternConfig) -> str:
    mode = cfg.symmetry_mode.lower().strip()
    if mode == "random":
        return str(rng.choice(np.array(["none", "x", "y", "xy"], dtype=object)))
    if mode not in {"none", "x", "y", "xy"}:
        return "none"
    return mode


def _sample_style(rng: np.random.Generator, cfg: RadiationPatternConfig) -> str:
    styles = ("front_back", "bidirectional", "petal", "ripple")
    mode = str(cfg.style_mode).strip().lower()
    if mode == "random":
        return str(rng.choice(np.array(styles, dtype=object)))
    if mode in styles:
        return mode
    return str(rng.choice(np.array(styles, dtype=object)))


def _style_envelope(rng: np.random.Generator, cfg: RadiationPatternConfig, style: str, k: np.ndarray) -> np.ndarray:
    if style == "front_back":
        # Gentle decay keeps higher harmonics alive for sidelobes.
        decay = float(rng.uniform(0.05, 0.16))
        env = np.exp(-decay * (k - 1.0))
        env[0] *= float(rng.uniform(1.2, 1.8))
        # Secondary bump adds a sidelobe cluster at a random harmonic.
        bump_k = float(rng.uniform(3.0, max(4.0, 0.45 * len(k))))
        bump_sigma = float(rng.uniform(1.0, 3.0))
        bump_amp = float(rng.uniform(0.4, 1.0))
        env += bump_amp * np.exp(-0.5 * ((k - bump_k) / bump_sigma) ** 2)
        return env
    if style == "bidirectional":
        decay = float(rng.uniform(0.06, 0.18))
        env = np.exp(-decay * (k - 1.0))
        odd = (k % 2 == 1)
        odd_suppress = float(rng.uniform(0.10, 0.35))
        env[odd] *= odd_suppress
        env[~odd] *= float(rng.uniform(1.1, 1.5))
        return env
    if style == "petal":
        m_min = int(max(2, cfg.petal_order_min))
        m_max = int(max(m_min, cfg.petal_order_max))
        m = int(rng.integers(m_min, m_max + 1))
        sigma = float(rng.uniform(0.8, 2.2))
        env = np.exp(-0.5 * ((k - float(m)) / sigma) ** 2)
        env += 0.25 * np.exp(-0.5 * ((k - 2.0 * float(m)) / max(1.0, 1.6 * sigma)) ** 2)
        return env
    # ripple — power-law base plus a random mid-frequency bump for structure.
    alpha = float(rng.uniform(0.4, 1.4))
    env = 1.0 / (k ** alpha)
    # Add a bump so even low-K patterns have visible angular structure.
    bump_k = float(rng.uniform(2.0, max(3.0, 0.35 * len(k))))
    bump_sigma = float(rng.uniform(1.2, 3.5))
    bump_amp = float(rng.uniform(0.2, 0.7))
    env += bump_amp * np.exp(-0.5 * ((k - bump_k) / bump_sigma) ** 2)
    return env


def _map_raw_to_losses(raw: np.ndarray, cfg: RadiationPatternConfig, rng: np.random.Generator) -> tuple[np.ndarray, float, float, float]:
    max_loss = float(max(0.0, cfg.max_loss_db))
    eff_max = float(rng.uniform(0.3 * max_loss, max_loss))

    raw = np.tanh(raw)
    rmin = float(np.min(raw))
    rmax = float(np.max(raw))
    if rmax > rmin:
        g = (raw - rmin) / (rmax - rmin)
    else:
        g = np.zeros_like(raw, dtype=np.float64) + 0.5
    losses = eff_max * (1.0 - g)
    return np.clip(losses, 0.0, max_loss), eff_max, rmin, rmax


def _generate_latent_fourier_pattern(
    rng: np.random.Generator,
    cfg: RadiationPatternConfig,
    azimuth_deg: float,
    symmetry: str,
    latent_dim: int,
    fourier_order: int,
) -> tuple[np.ndarray, str, dict]:
    d = int(max(2, latent_dim))
    K = int(max(1, fourier_order))
    # For xy symmetry, ensure K >= 2 so at least one even harmonic exists.
    if symmetry == "xy":
        K = max(K, 2)
    theta = np.arange(360, dtype=np.float64)

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

    losses, eff_max_db, rmin, rmax = _map_raw_to_losses(raw, cfg, rng)
    gains = -losses

    function_info = {
        "version": 1,
        "type": "latent_fourier",
        "azimuth_deg": float(azimuth_deg),
        "symmetry": symmetry,
        "max_loss_db": float(max(0.0, cfg.max_loss_db)),
        "output_units": "db_gain_negative_pathloss",
        "k": k.tolist(),
        "a": a.tolist(),
        "b": b.tolist(),
        "eff_max_db": eff_max_db,
        "rmin": rmin,
        "rmax": rmax,
    }

    return gains, style, function_info


def generate_radiation_pattern(rng: np.random.Generator, cfg: RadiationPatternConfig) -> RadiationPatternSample:
    p_iso = float(np.clip(cfg.isotropic_probability, 0.0, 1.0))
    model = "latent_fourier"
    dmin = int(max(2, cfg.latent_dim_min))
    dmax = int(max(dmin, cfg.latent_dim_max))
    sampled_d = int(rng.integers(dmin, dmax + 1))
    kmin = int(max(1, cfg.fourier_order_min))
    kmax = int(max(kmin, cfg.fourier_order_max))
    sampled_k = int(rng.integers(kmin, kmax + 1))

    if rng.random() < p_iso:
        iso_function_info = {
            "version": 1,
            "type": "isotropic",
            "azimuth_deg": 0.0,
            "symmetry": "none",
            "max_loss_db": float(max(0.0, cfg.max_loss_db)),
            "output_units": "db_gain_negative_pathloss",
        }
        return RadiationPatternSample(
            losses_db=np.zeros(360, dtype=np.float32),
            is_isotropic=True,
            azimuth_deg=0.0,
            symmetry="none",
            model=model,
            style="isotropic",
            complexity_dim=sampled_d,
            function_info=iso_function_info,
        )

    az = float(rng.uniform(0.0, 360.0))

    symmetry = _resolve_symmetry_mode(rng, cfg)
    losses, style, function_info = _generate_latent_fourier_pattern(
        rng,
        cfg,
        azimuth_deg=az,
        symmetry=symmetry,
        latent_dim=sampled_d,
        fourier_order=sampled_k,
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
        function_info=function_info,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_finite_float(info: dict, key: str) -> float:
    val = info.get(key)
    if val is None:
        raise ValueError(f"missing required field: {key}")
    val = float(val)
    if not np.isfinite(val):
        raise ValueError(f"non-finite value for field: {key}")
    return val


def _validate_finite_array(info: dict, key: str) -> np.ndarray:
    val = info.get(key)
    if val is None:
        raise ValueError(f"missing required field: {key}")
    arr = np.asarray(val, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected 1D array for field: {key}")
    if arr.size == 0:
        raise ValueError(f"empty array for field: {key}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"non-finite values in field: {key}")
    return arr


def validate_pattern_function_info(function_info: dict) -> dict:
    if not isinstance(function_info, dict):
        raise ValueError("function_info must be a dict")

    version = function_info.get("version")
    if version != 1:
        raise ValueError(f"unsupported version: {version}")

    ptype = function_info.get("type")
    if ptype not in {"isotropic", "latent_fourier"}:
        raise ValueError(f"invalid type: {ptype}")

    _validate_finite_float(function_info, "azimuth_deg")

    symmetry = function_info.get("symmetry")
    if symmetry not in {"none", "x", "y", "xy"}:
        raise ValueError(f"invalid symmetry: {symmetry}")

    max_loss_db = _validate_finite_float(function_info, "max_loss_db")
    if max_loss_db < 0:
        raise ValueError(f"max_loss_db must be >= 0, got {max_loss_db}")

    output_units = function_info.get("output_units")
    if output_units != "db_gain_negative_pathloss":
        raise ValueError(f"invalid output_units: {output_units}")

    if ptype == "latent_fourier":
        k = _validate_finite_array(function_info, "k")
        a = _validate_finite_array(function_info, "a")
        b = _validate_finite_array(function_info, "b")
        if not (k.size == a.size == b.size):
            raise ValueError("k, a, b must have equal length")
        if np.any(k <= 0):
            raise ValueError("k values must be strictly positive")
        if len(set(k.tolist())) != k.size:
            raise ValueError("k values must be unique")
        eff = _validate_finite_float(function_info, "eff_max_db")
        if eff < 0 or eff > max_loss_db:
            raise ValueError(f"eff_max_db must be in [0, max_loss_db], got {eff}")
        rmin = _validate_finite_float(function_info, "rmin")
        rmax = _validate_finite_float(function_info, "rmax")
        if rmax < rmin:
            raise ValueError(f"rmax ({rmax}) < rmin ({rmin})")
        if symmetry == "xy":
            k_int = np.round(k).astype(np.int64)
            if not np.any(k_int % 2 == 0):
                raise ValueError("xy symmetry requires at least one even harmonic in k")

    return function_info


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def _eval_isotropic(function_info: dict, angles_deg: np.ndarray) -> np.ndarray:
    return np.zeros(angles_deg.shape, dtype=np.float64)


def _eval_latent_fourier(function_info: dict, angles_deg: np.ndarray) -> np.ndarray:
    azimuth = function_info["azimuth_deg"]
    symmetry = function_info["symmetry"]
    max_loss_db = function_info["max_loss_db"]
    k = np.asarray(function_info["k"], dtype=np.float64)
    a = np.asarray(function_info["a"], dtype=np.float64)
    b = np.asarray(function_info["b"], dtype=np.float64)
    eff_max_db = function_info["eff_max_db"]
    rmin = function_info["rmin"]
    rmax = function_info["rmax"]

    theta = angles_deg.ravel()

    if symmetry == "x":
        phi = np.deg2rad((theta - azimuth) % 360.0)
        raw = np.cos(np.outer(phi, k)) @ a
    elif symmetry == "y":
        phi_y = np.deg2rad((theta - (azimuth + 90.0)) % 360.0)
        raw = np.cos(np.outer(phi_y, k)) @ a
    elif symmetry == "xy":
        phi = np.deg2rad((theta - azimuth) % 360.0)
        even_mask = (np.round(k).astype(np.int64) % 2) == 0
        kk = k[even_mask]
        aa = a[even_mask]
        raw = np.cos(np.outer(phi, kk)) @ aa
    else:
        phi = np.deg2rad((theta - azimuth) % 360.0)
        raw = np.cos(np.outer(phi, k)) @ a + np.sin(np.outer(phi, k)) @ b

    raw_tanh = np.tanh(raw)
    if rmax > rmin:
        g = (raw_tanh - rmin) / (rmax - rmin)
    else:
        g = np.full_like(raw_tanh, 0.5)
    loss_db = eff_max_db * (1.0 - g)
    result = -np.clip(loss_db, 0.0, max_loss_db)
    return result.reshape(angles_deg.shape)


def evaluate_pattern_function_db(function_info: dict, angles_deg) -> np.ndarray:
    validate_pattern_function_info(function_info)
    angles = np.asarray(angles_deg, dtype=np.float64)
    ptype = function_info["type"]
    if ptype == "isotropic":
        return _eval_isotropic(function_info, angles)
    elif ptype == "latent_fourier":
        return _eval_latent_fourier(function_info, angles)
    raise ValueError(f"unknown type: {ptype}")
