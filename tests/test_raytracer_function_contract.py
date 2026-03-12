"""Ray and pixel source-of-truth tests.

Verifies that the tracer uses the continuous function, not the discretized array.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import pytest
from models import RadarSample
from antenna_pattern import (
    RadiationPatternConfig, generate_radiation_pattern,
    evaluate_pattern_function_db,
)
from approx import _build_ray_initial_losses, _build_pixel_initial_loss_map


def _make_sample(fn_info, H=64, W=64, x_ant=32.0, y_ant=32.0,
                 pattern=None):
    if pattern is None:
        pattern = torch.zeros(360, dtype=torch.float32)
    return RadarSample(
        H=H, W=W, x_ant=x_ant, y_ant=y_ant, azimuth=0.0,
        freq_MHz=2400.0,
        reflectance=torch.zeros(H, W),
        transmittance=torch.zeros(H, W),
        dist_map=torch.zeros(H, W),
        pathloss=torch.zeros(H, W),
        radiation_pattern=pattern,
        radiation_pattern_fn_info=fn_info,
    )


class TestRayUsesFunction:
    def test_basic_ray_evaluation(self):
        cfg = RadiationPatternConfig(isotropic_probability=0.0)
        rng = np.random.default_rng(42)
        pat = generate_radiation_pattern(rng, cfg)
        sample = _make_sample(pat.function_info)
        ray_init = _build_ray_initial_losses(sample, 360)
        # Should be non-negative (negated gains)
        assert np.all(ray_init >= -1e-10)
        # Should not be all zeros for a directional pattern
        assert np.any(ray_init > 0.1)

    def test_poisoned_array_ignored(self):
        """Poisoning radiation_pattern doesn't affect result when fn_info is valid."""
        cfg = RadiationPatternConfig(isotropic_probability=0.0)
        rng = np.random.default_rng(42)
        pat = generate_radiation_pattern(rng, cfg)

        # Good sample
        good = _make_sample(pat.function_info)
        ray_good = _build_ray_initial_losses(good, 360)

        # Poisoned: fill radiation_pattern with garbage
        poison = torch.full((360,), 9999.0)
        poisoned = _make_sample(pat.function_info, pattern=poison)
        ray_poison = _build_ray_initial_losses(poisoned, 360)

        np.testing.assert_array_equal(ray_good, ray_poison)


class TestPixelUsesFunction:
    def test_basic_pixel_evaluation(self):
        cfg = RadiationPatternConfig(isotropic_probability=0.0)
        rng = np.random.default_rng(42)
        pat = generate_radiation_pattern(rng, cfg)
        sample = _make_sample(pat.function_info)
        pix = _build_pixel_initial_loss_map(sample)
        assert pix.shape == (64, 64)
        assert np.all(pix >= -1e-10)

    def test_poisoned_array_ignored(self):
        cfg = RadiationPatternConfig(isotropic_probability=0.0)
        rng = np.random.default_rng(42)
        pat = generate_radiation_pattern(rng, cfg)

        good = _make_sample(pat.function_info)
        pix_good = _build_pixel_initial_loss_map(good)

        poison = torch.full((360,), -9999.0)
        poisoned = _make_sample(pat.function_info, pattern=poison)
        pix_poison = _build_pixel_initial_loss_map(poisoned)

        np.testing.assert_array_equal(pix_good, pix_poison)


class TestMissingFnInfo:
    def test_ray_raises_on_none(self):
        sample = _make_sample(fn_info=None)
        # Override to None
        sample.radiation_pattern_fn_info = None
        with pytest.raises(ValueError, match="radiation_pattern_fn_info"):
            _build_ray_initial_losses(sample, 360)

    def test_pixel_raises_on_none(self):
        sample = _make_sample(fn_info=None)
        sample.radiation_pattern_fn_info = None
        with pytest.raises(ValueError, match="radiation_pattern_fn_info"):
            _build_pixel_initial_loss_map(sample)


class TestMalformedFnInfo:
    def test_ray_raises_on_bad_version(self):
        bad_info = {
            "version": 99, "type": "isotropic", "azimuth_deg": 0.0,
            "symmetry": "none", "max_loss_db": 20.0,
            "output_units": "db_gain_negative_pathloss",
        }
        sample = _make_sample(fn_info=bad_info)
        with pytest.raises(ValueError, match="version"):
            _build_ray_initial_losses(sample, 360)

    def test_pixel_raises_on_bad_type(self):
        bad_info = {
            "version": 1, "type": "invalid", "azimuth_deg": 0.0,
            "symmetry": "none", "max_loss_db": 20.0,
            "output_units": "db_gain_negative_pathloss",
        }
        sample = _make_sample(fn_info=bad_info)
        with pytest.raises(ValueError, match="type"):
            _build_pixel_initial_loss_map(sample)
