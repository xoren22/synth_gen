"""Failure-surface tests: one hard failure per invalid class."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from antenna_pattern import evaluate_pattern_function_db


class TestUnsupportedVersion:
    def test_version_2(self):
        info = {
            "version": 2, "type": "isotropic", "azimuth_deg": 0.0,
            "symmetry": "none", "max_loss_db": 20.0,
            "output_units": "db_gain_negative_pathloss",
        }
        with pytest.raises(ValueError, match="version"):
            evaluate_pattern_function_db(info, np.array([0.0]))


class TestMissingModelFields:
    def test_fourier_missing_k(self):
        info = {
            "version": 1, "type": "latent_fourier", "azimuth_deg": 0.0,
            "symmetry": "none", "max_loss_db": 20.0,
            "output_units": "db_gain_negative_pathloss",
            "a": [1.0], "b": [0.0],
            "eff_max_db": 10.0, "rmin": -0.5, "rmax": 0.5,
        }
        with pytest.raises(ValueError, match="k"):
            evaluate_pattern_function_db(info, np.array([0.0]))

class TestNaNCoefficients:
    def test_nan_in_fourier_a(self):
        info = {
            "version": 1, "type": "latent_fourier", "azimuth_deg": 0.0,
            "symmetry": "none", "max_loss_db": 20.0,
            "output_units": "db_gain_negative_pathloss",
            "k": [1.0], "a": [float("nan")], "b": [0.0],
            "eff_max_db": 10.0, "rmin": -0.5, "rmax": 0.5,
        }
        with pytest.raises(ValueError, match="a"):
            evaluate_pattern_function_db(info, np.array([0.0]))

class TestInfValues:
    def test_inf_eff_max(self):
        info = {
            "version": 1, "type": "latent_fourier", "azimuth_deg": 0.0,
            "symmetry": "none", "max_loss_db": 20.0,
            "output_units": "db_gain_negative_pathloss",
            "k": [1.0], "a": [1.0], "b": [0.0],
            "eff_max_db": float("inf"), "rmin": -0.5, "rmax": 0.5,
        }
        with pytest.raises(ValueError, match="eff_max_db"):
            evaluate_pattern_function_db(info, np.array([0.0]))


class TestEmptyArrays:
    def test_empty_k(self):
        info = {
            "version": 1, "type": "latent_fourier", "azimuth_deg": 0.0,
            "symmetry": "none", "max_loss_db": 20.0,
            "output_units": "db_gain_negative_pathloss",
            "k": [], "a": [], "b": [],
            "eff_max_db": 10.0, "rmin": -0.5, "rmax": 0.5,
        }
        with pytest.raises(ValueError, match="k"):
            evaluate_pattern_function_db(info, np.array([0.0]))


class TestInvalidSymmetryLabel:
    def test_symmetry_z(self):
        info = {
            "version": 1, "type": "isotropic", "azimuth_deg": 0.0,
            "symmetry": "z", "max_loss_db": 20.0,
            "output_units": "db_gain_negative_pathloss",
        }
        with pytest.raises(ValueError, match="symmetry"):
            evaluate_pattern_function_db(info, np.array([0.0]))


class TestInvalidUnits:
    def test_wrong_units(self):
        info = {
            "version": 1, "type": "isotropic", "azimuth_deg": 0.0,
            "symmetry": "none", "max_loss_db": 20.0,
            "output_units": "linear_power",
        }
        with pytest.raises(ValueError, match="output_units"):
            evaluate_pattern_function_db(info, np.array([0.0]))
