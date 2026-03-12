"""Tests for radiation pattern generation and its integration into ray-tracing.

Covers:
  A. Sign convention – pattern gains (negative) become non-negative losses
  B. Angle convention – azimuth is baked into the pattern, not applied again at lookup
  C. Pixel-ray consistency – pixel map and ray array agree for the same direction
  D. End-to-end – Approx.approximate produces valid output with non-isotropic patterns
  E. latent_dim sampling – sampled uniformly from latent_dim_min..latent_dim_max
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import pytest

from models import RadarSample
from antenna_pattern import (
    RadiationPatternConfig,
    generate_radiation_pattern,
)
from approx import _build_ray_initial_losses, _build_pixel_initial_loss_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ISO_FN_INFO = {
    "version": 1, "type": "isotropic", "azimuth_deg": 0.0,
    "symmetry": "none", "max_loss_db": 40.0,
    "output_units": "db_gain_negative_pathloss",
}


def _make_sample(
    H=64, W=64, x_ant=32.0, y_ant=32.0, freq_MHz=2400.0,
    azimuth=0.0, pattern=None, ref=None, trans=None,
    fn_info=None,
):
    """Build a minimal RadarSample for testing pattern lookup functions."""
    if ref is None:
        ref = torch.zeros(H, W, dtype=torch.float64)
    if trans is None:
        trans = torch.zeros(H, W, dtype=torch.float64)
    if pattern is None:
        pattern = torch.zeros(360, dtype=torch.float64)
    if fn_info is None:
        fn_info = _ISO_FN_INFO
    return RadarSample(
        H=H, W=W,
        x_ant=x_ant, y_ant=y_ant,
        azimuth=azimuth,
        freq_MHz=freq_MHz,
        reflectance=ref,
        transmittance=trans,
        dist_map=torch.zeros(H, W),
        pathloss=torch.zeros(H, W),
        radiation_pattern=pattern,
        radiation_pattern_fn_info=fn_info,
    )


_NON_ISO_CFG = RadiationPatternConfig(
    isotropic_probability=0.0,
    max_loss_db=20.0,
    latent_dim_min=8,
    latent_dim_max=8,
)


@pytest.fixture
def non_iso_pattern():
    rng = np.random.default_rng(42)
    return generate_radiation_pattern(rng, _NON_ISO_CFG)


# ---------------------------------------------------------------------------
# Test A: Sign convention
# ---------------------------------------------------------------------------

class TestSignConvention:
    def test_pattern_stores_gains(self, non_iso_pattern):
        """The pattern generator returns gains (all <= 0)."""
        assert np.all(non_iso_pattern.losses_db <= 0)

    def test_ray_init_non_negative(self, non_iso_pattern):
        sample = _make_sample(
            pattern=torch.from_numpy(non_iso_pattern.losses_db.astype(np.float64)),
            azimuth=non_iso_pattern.azimuth_deg,
            fn_info=non_iso_pattern.function_info,
        )
        ray_init = _build_ray_initial_losses(sample, 360)
        assert np.all(ray_init >= 0), f"min={ray_init.min()}"

    def test_pixel_init_non_negative(self, non_iso_pattern):
        sample = _make_sample(
            pattern=torch.from_numpy(non_iso_pattern.losses_db.astype(np.float64)),
            azimuth=non_iso_pattern.azimuth_deg,
            fn_info=non_iso_pattern.function_info,
        )
        pix_init = _build_pixel_initial_loss_map(sample)
        assert np.all(pix_init >= 0), f"min={pix_init.min()}"

    def test_main_beam_lower_loss_than_null(self, non_iso_pattern):
        """Main beam (gain ~ 0) -> low loss; null (gain ~ -20) -> high loss."""
        sample = _make_sample(
            pattern=torch.from_numpy(non_iso_pattern.losses_db.astype(np.float64)),
            azimuth=non_iso_pattern.azimuth_deg,
            fn_info=non_iso_pattern.function_info,
        )
        ray_init = _build_ray_initial_losses(sample, 360)
        pat_np = non_iso_pattern.losses_db.astype(np.float64)
        main_beam_idx = int(np.argmax(pat_np))
        null_idx = int(np.argmin(pat_np))
        assert ray_init[main_beam_idx] < ray_init[null_idx]

    def test_isotropic_pattern_gives_zero_loss(self):
        """An all-zero (isotropic) pattern should produce zero additional loss."""
        sample = _make_sample(pattern=torch.zeros(360, dtype=torch.float64))
        ray_init = _build_ray_initial_losses(sample, 360)
        np.testing.assert_array_equal(ray_init, 0.0)


# ---------------------------------------------------------------------------
# Test B: Angle convention – no double azimuth application
# ---------------------------------------------------------------------------

class TestAngleConvention:
    def test_same_seed_same_ray_init(self):
        """Same seed -> same pattern array -> same ray_init (azimuth not re-applied)."""
        rng1 = np.random.default_rng(99)
        pat1 = generate_radiation_pattern(rng1, _NON_ISO_CFG)
        rng2 = np.random.default_rng(99)
        pat2 = generate_radiation_pattern(rng2, _NON_ISO_CFG)

        s1 = _make_sample(pattern=torch.from_numpy(pat1.losses_db.astype(np.float64)),
                          azimuth=pat1.azimuth_deg, fn_info=pat1.function_info)
        s2 = _make_sample(pattern=torch.from_numpy(pat2.losses_db.astype(np.float64)),
                          azimuth=pat2.azimuth_deg, fn_info=pat2.function_info)

        ray1 = _build_ray_initial_losses(s1, 360)
        ray2 = _build_ray_initial_losses(s2, 360)
        np.testing.assert_array_almost_equal(ray1, ray2)

    def test_ray_peak_aligns_with_pattern_peak(self, non_iso_pattern):
        """Minimum ray loss should be near the function evaluation's max gain."""
        from antenna_pattern import evaluate_pattern_function_db
        sample = _make_sample(
            pattern=torch.from_numpy(non_iso_pattern.losses_db.astype(np.float64)),
            azimuth=non_iso_pattern.azimuth_deg,
            fn_info=non_iso_pattern.function_info,
        )
        ray_init = _build_ray_initial_losses(sample, 360)
        # Compare against the function evaluation, not the float32 stored array,
        # to avoid degenerate-peak tie-breaking differences.
        angles = np.arange(360, dtype=np.float64)
        fn_gains = evaluate_pattern_function_db(non_iso_pattern.function_info, angles)
        pattern_peak_idx = int(np.argmax(fn_gains))
        ray_min_idx = int(np.argmin(ray_init))
        diff = abs(pattern_peak_idx - ray_min_idx) % 360
        diff = min(diff, 360 - diff)
        assert diff <= 2, f"ray peak {ray_min_idx} vs pattern peak {pattern_peak_idx}"

    def test_different_azimuth_rotates_pixel_map(self):
        """Two patterns with different azimuths should produce rotated pixel maps."""
        rng = np.random.default_rng(55)
        pat = generate_radiation_pattern(rng, _NON_ISO_CFG)

        # Use a second pattern with different azimuth
        rng2 = np.random.default_rng(77)
        pat2 = generate_radiation_pattern(rng2, _NON_ISO_CFG)

        s1 = _make_sample(pattern=torch.from_numpy(pat.losses_db.astype(np.float64)),
                          fn_info=pat.function_info)
        s2 = _make_sample(pattern=torch.from_numpy(pat2.losses_db.astype(np.float64)),
                          fn_info=pat2.function_info)

        pix1 = _build_pixel_initial_loss_map(s1)
        pix2 = _build_pixel_initial_loss_map(s2)

        # They should differ
        assert not np.allclose(pix1, pix2), "Different patterns should give different pixel map"


# ---------------------------------------------------------------------------
# Test C: Pixel lookup matches ray lookup
# ---------------------------------------------------------------------------

class TestPixelRayConsistency:
    def test_right_direction(self, non_iso_pattern):
        """Pixel to the right of antenna (atan2=0°) matches ray at index 0."""
        H, W = 128, 128
        sample = _make_sample(
            H=H, W=W, x_ant=64.0, y_ant=64.0,
            pattern=torch.from_numpy(non_iso_pattern.losses_db.astype(np.float64)),
            azimuth=non_iso_pattern.azimuth_deg,
            fn_info=non_iso_pattern.function_info,
        )
        ray_init = _build_ray_initial_losses(sample, 360)
        pix_init = _build_pixel_initial_loss_map(sample)

        # px=120, py=64 -> dx=56, dy=0 -> atan2(0,56)=0°
        assert abs(pix_init[64, 120] - ray_init[0]) < 0.5

    def test_down_direction(self, non_iso_pattern):
        """Pixel below antenna (atan2=90°) matches ray at index 90."""
        H, W = 128, 128
        sample = _make_sample(
            H=H, W=W, x_ant=64.0, y_ant=64.0,
            pattern=torch.from_numpy(non_iso_pattern.losses_db.astype(np.float64)),
            azimuth=non_iso_pattern.azimuth_deg,
            fn_info=non_iso_pattern.function_info,
        )
        ray_init = _build_ray_initial_losses(sample, 360)
        pix_init = _build_pixel_initial_loss_map(sample)

        # px=64, py=120 -> dx=0, dy=56 -> atan2(56,0)=90°
        assert abs(pix_init[120, 64] - ray_init[90]) < 0.5

    def test_left_direction(self, non_iso_pattern):
        """Pixel to the left (atan2=180°) matches ray at index 180."""
        H, W = 128, 128
        sample = _make_sample(
            H=H, W=W, x_ant=64.0, y_ant=64.0,
            pattern=torch.from_numpy(non_iso_pattern.losses_db.astype(np.float64)),
            azimuth=non_iso_pattern.azimuth_deg,
            fn_info=non_iso_pattern.function_info,
        )
        ray_init = _build_ray_initial_losses(sample, 360)
        pix_init = _build_pixel_initial_loss_map(sample)

        # px=8, py=64 -> dx=-56, dy=0 -> atan2(0,-56)=180°
        assert abs(pix_init[64, 8] - ray_init[180]) < 0.5

    def test_up_direction(self, non_iso_pattern):
        """Pixel above antenna (atan2=-90°=270°) matches ray at index 270."""
        H, W = 128, 128
        sample = _make_sample(
            H=H, W=W, x_ant=64.0, y_ant=64.0,
            pattern=torch.from_numpy(non_iso_pattern.losses_db.astype(np.float64)),
            azimuth=non_iso_pattern.azimuth_deg,
            fn_info=non_iso_pattern.function_info,
        )
        ray_init = _build_ray_initial_losses(sample, 360)
        pix_init = _build_pixel_initial_loss_map(sample)

        # px=64, py=8 -> dx=0, dy=-56 -> atan2(-56,0)=-90° -> 270°
        assert abs(pix_init[8, 64] - ray_init[270]) < 0.5


# ---------------------------------------------------------------------------
# Test D: End-to-end with Approx
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_no_nan_inf(self, non_iso_pattern):
        from approx import Approx

        H, W = 32, 32
        trans = torch.zeros(H, W, dtype=torch.float64)
        ref = torch.zeros(H, W, dtype=torch.float64)
        trans[10:22, 15] = 8.0
        ref[10:22, 15] = 3.0

        sample = _make_sample(
            H=H, W=W, x_ant=5.0, y_ant=16.0,
            pattern=torch.from_numpy(non_iso_pattern.losses_db.astype(np.float64)),
            azimuth=non_iso_pattern.azimuth_deg,
            ref=ref, trans=trans,
            fn_info=non_iso_pattern.function_info,
        )

        result = Approx(method='combined').approximate(sample, max_trans=3, max_refl=2)
        r = result.numpy()
        assert not np.any(np.isnan(r))
        assert not np.any(np.isinf(r))
        assert np.all(r >= 0)

    def test_directional_pattern_breaks_symmetry(self):
        """A non-isotropic pattern should produce asymmetric output vs isotropic."""
        from approx import Approx

        H, W = 32, 32
        trans = torch.zeros(H, W, dtype=torch.float64)
        ref = torch.zeros(H, W, dtype=torch.float64)

        rng = np.random.default_rng(200)
        pat = generate_radiation_pattern(rng, _NON_ISO_CFG)

        sample_dir = _make_sample(
            H=H, W=W, x_ant=16.0, y_ant=16.0,
            pattern=torch.from_numpy(pat.losses_db.astype(np.float64)),
            azimuth=pat.azimuth_deg,
            ref=ref, trans=trans,
            fn_info=pat.function_info,
        )
        sample_iso = _make_sample(
            H=H, W=W, x_ant=16.0, y_ant=16.0,
            pattern=torch.zeros(360, dtype=torch.float64),
            azimuth=0.0,
            ref=ref, trans=trans,
        )

        approx = Approx(method='combined')
        r_dir = approx.approximate(sample_dir).numpy()
        r_iso = approx.approximate(sample_iso).numpy()

        assert not np.allclose(r_dir, r_iso), "Directional pattern should differ from isotropic"


# ---------------------------------------------------------------------------
# Test E: latent_dim respected
# ---------------------------------------------------------------------------

class TestLatentDim:
    def test_degenerate_range_uses_that_value(self):
        """When latent_dim_min == latent_dim_max, complexity_dim is that fixed value."""
        cfg = RadiationPatternConfig(
            isotropic_probability=0.0,
        
            latent_dim_min=10,
            latent_dim_max=10,
        )
        rng = np.random.default_rng(42)
        pat = generate_radiation_pattern(rng, cfg)
        assert pat.complexity_dim == 10

    def test_range_sampling(self):
        """When latent_dim_min != latent_dim_max, sample from [min, max]."""
        cfg = RadiationPatternConfig(
            isotropic_probability=0.0,
        
            latent_dim_min=4,
            latent_dim_max=8,
        )
        dims = set()
        for seed in range(50):
            rng = np.random.default_rng(seed)
            p = generate_radiation_pattern(rng, cfg)
            assert 4 <= p.complexity_dim <= 8
            dims.add(p.complexity_dim)
        assert len(dims) > 1, f"Should sample different dims, got {dims}"


