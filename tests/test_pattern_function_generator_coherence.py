"""Generator-to-function coherence tests.

Verifies that the exported losses_db array matches the canonical function
evaluation on the export grid.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from antenna_pattern import (
    RadiationPatternConfig, generate_radiation_pattern,
    evaluate_pattern_function_db,
)

_EXPORT_GRID = np.arange(360, dtype=np.float64)


class TestLatentFourierCoherence:
    @pytest.mark.parametrize("seed", [0, 7, 42, 99, 200])
    def test_array_matches_function(self, seed):
        cfg = RadiationPatternConfig(
            isotropic_probability=0.0,
        )
        rng = np.random.default_rng(seed)
        pat = generate_radiation_pattern(rng, cfg)
        assert pat.function_info is not None
        assert pat.function_info["type"] == "latent_fourier"
        # Evaluate function at export grid
        fn_gains = evaluate_pattern_function_db(pat.function_info, _EXPORT_GRID)
        # Generator stores float32; compare against float16-quantized canonical
        # evaluation since NPZ export uses float16.
        stored = pat.losses_db.astype(np.float64)
        # The stored array is float32 of the generator output.
        # The function evaluation at integer degrees should match to float32 precision.
        np.testing.assert_allclose(
            stored, fn_gains,
            atol=1e-5,  # float32 precision limit
            rtol=1e-5,
        )


class TestIsotropicCoherence:
    def test_isotropic_array_matches(self):
        cfg = RadiationPatternConfig(isotropic_probability=1.0)
        rng = np.random.default_rng(42)
        pat = generate_radiation_pattern(rng, cfg)
        assert pat.function_info is not None
        assert pat.function_info["type"] == "isotropic"
        fn_gains = evaluate_pattern_function_db(pat.function_info, _EXPORT_GRID)
        np.testing.assert_array_equal(fn_gains, 0.0)
        np.testing.assert_array_equal(pat.losses_db, 0.0)


class TestMetadataConsistency:
    @pytest.mark.parametrize("seed", [0, 42, 99])
    def test_summary_matches_payload(self, seed):
        cfg = RadiationPatternConfig(isotropic_probability=0.0)
        rng = np.random.default_rng(seed)
        pat = generate_radiation_pattern(rng, cfg)
        fi = pat.function_info
        assert fi["symmetry"] == pat.symmetry
        assert fi["type"] == "latent_fourier"
        assert pat.model == "latent_fourier"


class TestAllSymmetries:
    @pytest.mark.parametrize("symmetry", ["none", "x", "y", "xy"])
    def test_coherence_per_symmetry(self, symmetry):
        cfg = RadiationPatternConfig(
            isotropic_probability=0.0,

            symmetry_mode=symmetry,
        )
        rng = np.random.default_rng(42)
        pat = generate_radiation_pattern(rng, cfg)
        assert pat.symmetry == symmetry
        fn_gains = evaluate_pattern_function_db(pat.function_info, _EXPORT_GRID)
        stored = pat.losses_db.astype(np.float64)
        np.testing.assert_allclose(stored, fn_gains, atol=1e-5, rtol=1e-5)
