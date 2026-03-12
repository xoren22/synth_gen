"""Geometry realism tests.

Validates angle convention and downstream usage on actual geometry.
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


def _make_sample(fn_info, H=128, W=128, x_ant=64.0, y_ant=64.0):
    return RadarSample(
        H=H, W=W, x_ant=x_ant, y_ant=y_ant, azimuth=0.0,
        freq_MHz=2400.0,
        reflectance=torch.zeros(H, W),
        transmittance=torch.zeros(H, W),
        dist_map=torch.zeros(H, W),
        pathloss=torch.zeros(H, W),
        radiation_pattern=torch.zeros(360),
        radiation_pattern_fn_info=fn_info,
    )


class TestDirectionalAgreement:
    """Pixel at cardinal direction matches corresponding ray angle."""

    @pytest.fixture
    def directional_sample(self):
        cfg = RadiationPatternConfig(
            isotropic_probability=0.0,
            symmetry_mode="none",
        )
        rng = np.random.default_rng(42)
        pat = generate_radiation_pattern(rng, cfg)
        return _make_sample(pat.function_info), pat

    def test_right(self, directional_sample):
        sample, pat = directional_sample
        ray = _build_ray_initial_losses(sample, 360)
        pix = _build_pixel_initial_loss_map(sample)
        # px=120, py=64 -> atan2(0, 56)=0 degrees
        assert abs(pix[64, 120] - ray[0]) < 0.5

    def test_down(self, directional_sample):
        sample, pat = directional_sample
        ray = _build_ray_initial_losses(sample, 360)
        pix = _build_pixel_initial_loss_map(sample)
        # px=64, py=120 -> atan2(56, 0)=90 degrees
        assert abs(pix[120, 64] - ray[90]) < 0.5

    def test_left(self, directional_sample):
        sample, pat = directional_sample
        ray = _build_ray_initial_losses(sample, 360)
        pix = _build_pixel_initial_loss_map(sample)
        # px=8, py=64 -> atan2(0, -56)=180 degrees
        assert abs(pix[64, 8] - ray[180]) < 0.5

    def test_up(self, directional_sample):
        sample, pat = directional_sample
        ray = _build_ray_initial_losses(sample, 360)
        pix = _build_pixel_initial_loss_map(sample)
        # px=64, py=8 -> atan2(-56, 0)=-90 -> 270 degrees
        assert abs(pix[8, 64] - ray[270]) < 0.5


class TestBeamVsNull:
    def test_main_beam_lower_loss(self):
        cfg = RadiationPatternConfig(
            isotropic_probability=0.0,
        )
        rng = np.random.default_rng(42)
        pat = generate_radiation_pattern(rng, cfg)
        sample = _make_sample(pat.function_info)
        ray = _build_ray_initial_losses(sample, 360)
        # Evaluate gains from function
        angles = np.arange(360, dtype=np.float64)
        gains = evaluate_pattern_function_db(pat.function_info, angles)
        beam_idx = int(np.argmax(gains))
        null_idx = int(np.argmin(gains))
        assert ray[beam_idx] < ray[null_idx]


class TestIsotropicZeroLoss:
    def test_zero_additional_loss(self):
        iso_info = {
            "version": 1, "type": "isotropic", "azimuth_deg": 0.0,
            "symmetry": "none", "max_loss_db": 20.0,
            "output_units": "db_gain_negative_pathloss",
        }
        sample = _make_sample(iso_info)
        ray = _build_ray_initial_losses(sample, 360)
        np.testing.assert_array_equal(ray, 0.0)
        pix = _build_pixel_initial_loss_map(sample)
        np.testing.assert_array_equal(pix, 0.0)


class TestDirectionalBreaksIsotropicSymmetry:
    def test_approx_differs(self):
        from approx import Approx
        H, W = 32, 32
        trans = torch.zeros(H, W, dtype=torch.float64)
        ref = torch.zeros(H, W, dtype=torch.float64)

        cfg = RadiationPatternConfig(
            isotropic_probability=0.0,
        )
        rng = np.random.default_rng(200)
        pat = generate_radiation_pattern(rng, cfg)

        iso_info = {
            "version": 1, "type": "isotropic", "azimuth_deg": 0.0,
            "symmetry": "none", "max_loss_db": 40.0,
            "output_units": "db_gain_negative_pathloss",
        }

        s_dir = RadarSample(
            H=H, W=W, x_ant=16.0, y_ant=16.0, azimuth=0.0,
            freq_MHz=2400.0, reflectance=ref, transmittance=trans,
            dist_map=torch.zeros(H, W), pathloss=torch.zeros(H, W),
            radiation_pattern=torch.zeros(360),
            radiation_pattern_fn_info=pat.function_info,
        )
        s_iso = RadarSample(
            H=H, W=W, x_ant=16.0, y_ant=16.0, azimuth=0.0,
            freq_MHz=2400.0, reflectance=ref, transmittance=trans,
            dist_map=torch.zeros(H, W), pathloss=torch.zeros(H, W),
            radiation_pattern=torch.zeros(360),
            radiation_pattern_fn_info=iso_info,
        )

        approx = Approx(method='combined')
        r_dir = approx.approximate(s_dir).numpy()
        r_iso = approx.approximate(s_iso).numpy()
        assert not np.allclose(r_dir, r_iso)
