"""Performance sanity benchmark for pattern evaluator.

Marked as benchmark so it doesn't make normal test suite flaky.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import numpy as np
import pytest
from antenna_pattern import evaluate_pattern_function_db


_FOURIER_INFO = {
    "version": 1, "type": "latent_fourier", "azimuth_deg": 45.0,
    "symmetry": "none", "max_loss_db": 40.0,
    "output_units": "db_gain_negative_pathloss",
    "k": list(range(1, 21)),
    "a": [float(x) for x in np.random.default_rng(0).normal(0, 1, 20)],
    "b": [float(x) for x in np.random.default_rng(1).normal(0, 1, 20)],
    "eff_max_db": 20.0, "rmin": -0.8, "rmax": 0.8,
}

N_ANGLES = 46080  # matches tracer resolution


@pytest.mark.benchmark
class TestPerfSanity:
    def test_latent_fourier_under_200ms(self):
        angles = np.linspace(0, 360, N_ANGLES, endpoint=False)
        # Warmup
        evaluate_pattern_function_db(_FOURIER_INFO, angles)
        # Timed run
        t0 = time.perf_counter()
        result = evaluate_pattern_function_db(_FOURIER_INFO, angles)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert result.shape == (N_ANGLES,)
        assert elapsed_ms < 200, f"Fourier eval took {elapsed_ms:.1f}ms (limit 200ms)"

    def test_isotropic_under_5ms(self):
        iso = {
            "version": 1, "type": "isotropic", "azimuth_deg": 0.0,
            "symmetry": "none", "max_loss_db": 40.0,
            "output_units": "db_gain_negative_pathloss",
        }
        angles = np.linspace(0, 360, N_ANGLES, endpoint=False)
        evaluate_pattern_function_db(iso, angles)
        t0 = time.perf_counter()
        result = evaluate_pattern_function_db(iso, angles)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert result.shape == (N_ANGLES,)
        assert elapsed_ms < 5, f"Isotropic eval took {elapsed_ms:.1f}ms (limit 5ms)"
