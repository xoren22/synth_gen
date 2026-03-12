"""Evaluator oracle tests for latent Fourier pattern.

Hand-computed oracles using the explicit Fourier formula from the spec.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from antenna_pattern import evaluate_pattern_function_db


def _make_fourier_info(*, k, a, b, azimuth_deg=0.0, symmetry="none",
                       max_loss_db=20.0, eff_max_db=10.0, rmin=-0.5, rmax=0.5):
    return {
        "version": 1, "type": "latent_fourier",
        "azimuth_deg": azimuth_deg, "symmetry": symmetry,
        "max_loss_db": max_loss_db,
        "output_units": "db_gain_negative_pathloss",
        "k": k, "a": a, "b": b,
        "eff_max_db": eff_max_db, "rmin": rmin, "rmax": rmax,
    }


def _oracle_none(angles_deg, azimuth, k, a, b, eff_max, rmin, rmax, max_loss):
    """Hand-compute expected output for symmetry='none'."""
    phi = np.deg2rad((angles_deg - azimuth) % 360.0)
    k = np.asarray(k, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    raw = np.cos(np.outer(phi, k)) @ a + np.sin(np.outer(phi, k)) @ b
    raw_t = np.tanh(raw)
    if rmax > rmin:
        g = (raw_t - rmin) / (rmax - rmin)
    else:
        g = np.full_like(raw_t, 0.5)
    loss = eff_max * (1.0 - g)
    return -np.clip(loss, 0.0, max_loss)


class TestSingleHarmonicCosine:
    """k=[1], a=[1], b=[0]: pure cosine shape."""

    def test_at_azimuth(self):
        info = _make_fourier_info(k=[1.0], a=[1.0], b=[0.0], azimuth_deg=0.0)
        result = evaluate_pattern_function_db(info, np.array([0.0]))
        expected = _oracle_none(np.array([0.0]), 0.0, [1.0], [1.0], [0.0],
                                10.0, -0.5, 0.5, 20.0)
        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_at_180(self):
        info = _make_fourier_info(k=[1.0], a=[1.0], b=[0.0], azimuth_deg=0.0)
        result = evaluate_pattern_function_db(info, np.array([180.0]))
        expected = _oracle_none(np.array([180.0]), 0.0, [1.0], [1.0], [0.0],
                                10.0, -0.5, 0.5, 20.0)
        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_full_circle(self):
        info = _make_fourier_info(k=[1.0], a=[1.0], b=[0.0], azimuth_deg=45.0)
        angles = np.arange(360, dtype=np.float64)
        result = evaluate_pattern_function_db(info, angles)
        expected = _oracle_none(angles, 45.0, [1.0], [1.0], [0.0],
                                10.0, -0.5, 0.5, 20.0)
        np.testing.assert_allclose(result, expected, atol=1e-9)


class TestSingleHarmonicSine:
    """k=[1], a=[0], b=[1]: pure sine shape."""

    def test_full_circle(self):
        info = _make_fourier_info(k=[1.0], a=[0.0], b=[1.0], azimuth_deg=90.0)
        angles = np.linspace(0, 359, 1000)
        result = evaluate_pattern_function_db(info, angles)
        expected = _oracle_none(angles, 90.0, [1.0], [0.0], [1.0],
                                10.0, -0.5, 0.5, 20.0)
        np.testing.assert_allclose(result, expected, atol=1e-9)


class TestMultiHarmonic:
    def test_two_harmonics(self):
        info = _make_fourier_info(
            k=[1.0, 3.0], a=[0.8, 0.3], b=[-0.2, 0.5],
            azimuth_deg=120.0, eff_max_db=15.0, rmin=-0.7, rmax=0.6,
            max_loss_db=30.0,
        )
        angles = np.array([0.0, 60.0, 120.0, 240.0, 359.0])
        result = evaluate_pattern_function_db(info, angles)
        expected = _oracle_none(angles, 120.0, [1.0, 3.0], [0.8, 0.3], [-0.2, 0.5],
                                15.0, -0.7, 0.6, 30.0)
        np.testing.assert_allclose(result, expected, atol=1e-9)


class TestSymmetryX:
    def test_x_symmetry_uses_cos_only(self):
        """x symmetry: raw = cos(k*phi) @ a, b is ignored."""
        info = _make_fourier_info(
            k=[1.0, 2.0], a=[0.5, 0.3], b=[0.9, 0.7],
            azimuth_deg=30.0, symmetry="x",
        )
        angles = np.arange(360, dtype=np.float64)
        result = evaluate_pattern_function_db(info, angles)
        # Mirror symmetry: f(az + d) = f(az - d)
        d = 50.0
        val_plus = evaluate_pattern_function_db(info, np.array([30.0 + d]))
        val_minus = evaluate_pattern_function_db(info, np.array([30.0 - d]))
        np.testing.assert_allclose(val_plus, val_minus, atol=1e-9)


class TestSymmetryY:
    def test_y_symmetry(self):
        """y symmetry: raw = cos(k*phi_y) @ a, axis is az+90."""
        info = _make_fourier_info(
            k=[1.0, 2.0], a=[0.5, 0.3], b=[0.9, 0.7],
            azimuth_deg=30.0, symmetry="y",
        )
        # Mirror about az+90=120: f(120+d) = f(120-d)
        d = 40.0
        val_plus = evaluate_pattern_function_db(info, np.array([120.0 + d]))
        val_minus = evaluate_pattern_function_db(info, np.array([120.0 - d]))
        np.testing.assert_allclose(val_plus, val_minus, atol=1e-9)


class TestSymmetryXY:
    def test_xy_symmetry_even_harmonics(self):
        """xy symmetry: uses only even harmonics with cosine."""
        info = _make_fourier_info(
            k=[2.0, 4.0], a=[0.5, 0.3], b=[0.9, 0.7],
            azimuth_deg=0.0, symmetry="xy",
        )
        angles = np.arange(360, dtype=np.float64)
        result = evaluate_pattern_function_db(info, angles)
        # Both x and y mirror symmetry should hold
        d = 37.0
        vp_x = evaluate_pattern_function_db(info, np.array([0.0 + d]))
        vm_x = evaluate_pattern_function_db(info, np.array([0.0 - d]))
        np.testing.assert_allclose(vp_x, vm_x, atol=1e-9)
        vp_y = evaluate_pattern_function_db(info, np.array([90.0 + d]))
        vm_y = evaluate_pattern_function_db(info, np.array([90.0 - d]))
        np.testing.assert_allclose(vp_y, vm_y, atol=1e-9)


class TestWraparound:
    def test_360_equiv_0(self):
        info = _make_fourier_info(k=[1.0, 2.0], a=[0.5, 0.3], b=[0.2, -0.1])
        v0 = evaluate_pattern_function_db(info, np.array([0.0]))
        v360 = evaluate_pattern_function_db(info, np.array([360.0]))
        np.testing.assert_allclose(v0, v360, atol=1e-12)

    def test_negative_angle(self):
        info = _make_fourier_info(k=[1.0], a=[1.0], b=[0.0])
        v_neg = evaluate_pattern_function_db(info, np.array([-90.0]))
        v_pos = evaluate_pattern_function_db(info, np.array([270.0]))
        np.testing.assert_allclose(v_neg, v_pos, atol=1e-12)


class TestRangePreservation:
    def test_output_in_valid_range(self):
        info = _make_fourier_info(
            k=[1.0, 2.0, 3.0], a=[0.8, -0.5, 0.3], b=[0.2, 0.6, -0.4],
            max_loss_db=25.0, eff_max_db=20.0,
        )
        angles = np.linspace(0, 360, 3600)
        result = evaluate_pattern_function_db(info, angles)
        assert np.all(result <= 0), f"max={result.max()}"
        assert np.all(result >= -25.0), f"min={result.min()}"
