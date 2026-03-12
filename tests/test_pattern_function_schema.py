"""Schema validation tests for validate_pattern_function_info."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy
import pytest
import numpy as np
from antenna_pattern import validate_pattern_function_info


# ---------------------------------------------------------------------------
# Valid base payloads
# ---------------------------------------------------------------------------

_VALID_ISO = {
    "version": 1, "type": "isotropic", "azimuth_deg": 0.0,
    "symmetry": "none", "max_loss_db": 20.0,
    "output_units": "db_gain_negative_pathloss",
}

_VALID_FOURIER = {
    "version": 1, "type": "latent_fourier", "azimuth_deg": 45.0,
    "symmetry": "none", "max_loss_db": 30.0,
    "output_units": "db_gain_negative_pathloss",
    "k": [1.0, 2.0, 3.0], "a": [0.5, -0.3, 0.1], "b": [0.2, 0.0, -0.1],
    "eff_max_db": 15.0, "rmin": -0.5, "rmax": 0.5,
}

class TestValidPayloads:
    def test_isotropic_valid(self):
        result = validate_pattern_function_info(_VALID_ISO)
        assert result is _VALID_ISO

    def test_fourier_valid(self):
        result = validate_pattern_function_info(_VALID_FOURIER)
        assert result is _VALID_FOURIER


class TestMissingRequiredKeys:
    @pytest.mark.parametrize("key", ["version", "type", "azimuth_deg",
                                     "symmetry", "max_loss_db", "output_units"])
    def test_missing_common_key(self, key):
        info = copy.deepcopy(_VALID_ISO)
        del info[key]
        with pytest.raises(ValueError):
            validate_pattern_function_info(info)


class TestVersionAndType:
    def test_unsupported_version(self):
        info = {**_VALID_ISO, "version": 2}
        with pytest.raises(ValueError, match="version"):
            validate_pattern_function_info(info)

    def test_invalid_type(self):
        info = {**_VALID_ISO, "type": "dipole"}
        with pytest.raises(ValueError, match="type"):
            validate_pattern_function_info(info)


class TestSymmetry:
    def test_invalid_symmetry(self):
        info = {**_VALID_ISO, "symmetry": "z"}
        with pytest.raises(ValueError, match="symmetry"):
            validate_pattern_function_info(info)


class TestOutputUnits:
    def test_invalid_units(self):
        info = {**_VALID_ISO, "output_units": "watts"}
        with pytest.raises(ValueError, match="output_units"):
            validate_pattern_function_info(info)


class TestNonFiniteFields:
    def test_nan_azimuth(self):
        info = {**_VALID_ISO, "azimuth_deg": float("nan")}
        with pytest.raises(ValueError, match="azimuth_deg"):
            validate_pattern_function_info(info)

    def test_inf_max_loss(self):
        info = {**_VALID_ISO, "max_loss_db": float("inf")}
        with pytest.raises(ValueError, match="max_loss_db"):
            validate_pattern_function_info(info)

    def test_negative_max_loss(self):
        info = {**_VALID_ISO, "max_loss_db": -1.0}
        with pytest.raises(ValueError, match="max_loss_db"):
            validate_pattern_function_info(info)


class TestLatentFourierValidation:
    def test_mismatched_lengths(self):
        info = copy.deepcopy(_VALID_FOURIER)
        info["b"] = [0.1, 0.2]
        with pytest.raises(ValueError, match="equal length"):
            validate_pattern_function_info(info)

    def test_empty_k(self):
        info = copy.deepcopy(_VALID_FOURIER)
        info["k"] = []
        info["a"] = []
        info["b"] = []
        with pytest.raises(ValueError, match="k"):
            validate_pattern_function_info(info)

    def test_non_positive_k(self):
        info = copy.deepcopy(_VALID_FOURIER)
        info["k"] = [0.0, 1.0, 2.0]
        with pytest.raises(ValueError, match="k"):
            validate_pattern_function_info(info)

    def test_duplicate_k(self):
        info = copy.deepcopy(_VALID_FOURIER)
        info["k"] = [1.0, 1.0, 2.0]
        with pytest.raises(ValueError, match="k"):
            validate_pattern_function_info(info)

    def test_nan_coefficients(self):
        info = copy.deepcopy(_VALID_FOURIER)
        info["a"] = [float("nan"), 0.0, 0.0]
        with pytest.raises(ValueError, match="a"):
            validate_pattern_function_info(info)

    def test_rmax_less_than_rmin(self):
        info = copy.deepcopy(_VALID_FOURIER)
        info["rmin"] = 1.0
        info["rmax"] = 0.0
        with pytest.raises(ValueError, match="rmax"):
            validate_pattern_function_info(info)

    def test_eff_max_exceeds_max_loss(self):
        info = copy.deepcopy(_VALID_FOURIER)
        info["eff_max_db"] = 50.0  # > max_loss_db=30
        with pytest.raises(ValueError, match="eff_max_db"):
            validate_pattern_function_info(info)

    def test_xy_symmetry_no_even_harmonics(self):
        info = copy.deepcopy(_VALID_FOURIER)
        info["symmetry"] = "xy"
        info["k"] = [1.0, 3.0, 5.0]
        with pytest.raises(ValueError, match="even harmonic"):
            validate_pattern_function_info(info)


class TestNotADict:
    def test_none(self):
        with pytest.raises(ValueError, match="dict"):
            validate_pattern_function_info(None)

    def test_list(self):
        with pytest.raises(ValueError, match="dict"):
            validate_pattern_function_info([1, 2, 3])
