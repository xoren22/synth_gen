"""Evaluator oracle tests for isotropic pattern."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from antenna_pattern import evaluate_pattern_function_db


_ISO = {
    "version": 1, "type": "isotropic", "azimuth_deg": 0.0,
    "symmetry": "none", "max_loss_db": 20.0,
    "output_units": "db_gain_negative_pathloss",
}


class TestIsotropicEval:
    def test_scalar_angle(self):
        result = evaluate_pattern_function_db(_ISO, np.array([45.0]))
        np.testing.assert_array_equal(result, 0.0)

    def test_vector_angles(self):
        angles = np.linspace(0, 359, 360)
        result = evaluate_pattern_function_db(_ISO, angles)
        np.testing.assert_array_equal(result, 0.0)
        assert result.shape == (360,)

    def test_wrapped_angles(self):
        angles = np.array([-720.0, -1.0, 359.0, 721.0])
        result = evaluate_pattern_function_db(_ISO, angles)
        np.testing.assert_array_equal(result, 0.0)

    def test_empty_array(self):
        result = evaluate_pattern_function_db(_ISO, np.array([], dtype=np.float64))
        assert result.shape == (0,)
        assert result.dtype == np.float64

    def test_output_dtype(self):
        result = evaluate_pattern_function_db(_ISO, np.array([0.0, 90.0, 180.0]))
        assert result.dtype == np.float64
