"""Real NPZ roundtrip tests.

Uses actual disk files with np.savez_compressed and np.load.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import tempfile
import numpy as np
import torch
import pytest
from models import RadarSample
from antenna_pattern import (
    RadiationPatternConfig, generate_radiation_pattern,
    evaluate_pattern_function_db,
)
from pattern_function_io import load_metadata_from_npz, radar_sample_from_npz


def _build_and_export(tmpdir, seed=42):
    """Generate a sample and export to NPZ, returning the path."""
    from generate import build_sample_from_generated

    cfg = RadiationPatternConfig(
        isotropic_probability=0.0,
    )
    H, W = 32, 32
    mask = np.ones((H, W), dtype=np.uint8)
    normals = np.zeros((H, W, 2), dtype=np.float32)
    scene = {
        "antenna": {"x": 16, "y": 16},
        "frequency_mhz": 2400,
        "canvas": {"width_m": 8.0, "height_m": 8.0},
    }
    refl = np.zeros((H, W), dtype=np.float32)
    trans = np.zeros((H, W), dtype=np.float32)
    trans[10:22, 15] = 5.0

    sample = build_sample_from_generated(
        mask, normals, scene, refl, trans,
        np.zeros((H, W), dtype=np.float32),
        pattern_cfg=cfg, pattern_seed=seed,
    )

    # Export
    arrays = {
        'normals': normals.astype(np.float16),
        'reflectance': refl.astype(np.float16),
        'transmittance': trans.astype(np.float16),
        'mask': mask,
        'pathloss': np.zeros((H, W), dtype=np.float16),
        'radiation_pattern_db': sample.radiation_pattern.numpy().astype(np.float16),
    }
    ant_pattern = scene.get('antenna_pattern', {})
    metadata = {
        'sample_name': 'test',
        'shape_hw': [H, W],
        'pixel_size_m': 0.25,
        'antenna': {
            'x_px': int(sample.x_ant),
            'y_px': int(sample.y_ant),
            'azimuth_deg': float(sample.azimuth),
            'pattern_units': 'db_gain_negative_pathloss',
            'is_isotropic': bool(ant_pattern.get('is_isotropic', False)),
            'pattern_symmetry': str(ant_pattern.get('symmetry', 'none')),
            'pattern_model': str(ant_pattern.get('model', 'latent_fourier')),
            'pattern_style': str(ant_pattern.get('style', 'unknown')),
            'pattern_complexity_dim': int(ant_pattern.get('complexity_dim', 0)),
            'pattern_function': ant_pattern.get('pattern_function'),
        },
        'frequency_mhz': 2400,
    }
    npz_path = os.path.join(tmpdir, 'test.npz')
    meta_json = json.dumps(metadata, separators=(",", ":"))
    np.savez_compressed(npz_path, **arrays, meta_json=np.asarray(meta_json))
    return npz_path, sample


class TestRoundtrip:
    def test_load_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, _ = _build_and_export(tmpdir)
            meta = load_metadata_from_npz(npz_path)
            assert "antenna" in meta
            assert "pattern_function" in meta["antenna"]
            assert meta["antenna"]["pattern_function"]["version"] == 1

    def test_reconstruct_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, original = _build_and_export(tmpdir)
            loaded = radar_sample_from_npz(npz_path)
            assert loaded.radiation_pattern_fn_info is not None
            assert loaded.H == original.H
            assert loaded.W == original.W

    def test_dense_evaluation_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, original = _build_and_export(tmpdir)
            loaded = radar_sample_from_npz(npz_path)
            angles = np.linspace(0, 359, 1000)
            gains_orig = original.evaluate_radiation_pattern_db(angles)
            gains_loaded = loaded.evaluate_radiation_pattern_db(angles)
            np.testing.assert_array_equal(gains_orig, gains_loaded)

    def test_poisoned_array_ignored(self):
        """Reconstructed sample ignores the stored float16 array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, original = _build_and_export(tmpdir)
            loaded = radar_sample_from_npz(npz_path)
            # The loaded radiation_pattern is float16 from disk,
            # but evaluate_radiation_pattern_db uses fn_info.
            angles = np.arange(360, dtype=np.float64)
            from_fn = loaded.evaluate_radiation_pattern_db(angles)
            # Compare against original function evaluation (float64)
            from_orig_fn = original.evaluate_radiation_pattern_db(angles)
            np.testing.assert_array_equal(from_fn, from_orig_fn)


# ---------------------------------------------------------------------------
# Per-style function_info JSON roundtrip: generate → serialize → deserialize
# → evaluate at non-trivial angles → assert bit-identical gains.
# ---------------------------------------------------------------------------

# Test angles: mix of integer degrees, fractional, negative, and >360 to
# exercise wrapping.  NOT the export grid — these are arbitrary query angles.
_TEST_ANGLES = np.array(
    [-10.0, 0.0, 0.5, 45.0, 89.999, 90.0, 179.5, 180.0, 270.0, 359.999, 360.0, 721.0],
    dtype=np.float64,
)


def _roundtrip_function_info(function_info: dict) -> dict:
    """Simulate the NPZ JSON roundtrip: dict → JSON string → dict."""
    return json.loads(json.dumps(function_info, separators=(",", ":")))


class TestPerStyleFunctionRoundtrip:
    """For each Fourier style: generate, JSON-roundtrip function_info, compare gains."""

    @pytest.mark.parametrize("style", ["front_back", "bidirectional", "petal", "ripple"])
    def test_style_roundtrip(self, style):
        cfg = RadiationPatternConfig(
            isotropic_probability=0.0,
            style_mode=style,
        )
        rng = np.random.default_rng(42)
        pat = generate_radiation_pattern(rng, cfg)
        assert pat.function_info is not None
        assert pat.function_info["type"] == "latent_fourier"
        assert pat.style == style

        gains_before = evaluate_pattern_function_db(pat.function_info, _TEST_ANGLES)
        reloaded_info = _roundtrip_function_info(pat.function_info)
        gains_after = evaluate_pattern_function_db(reloaded_info, _TEST_ANGLES)

        np.testing.assert_array_equal(gains_before, gains_after)

    def test_isotropic_roundtrip(self):
        cfg = RadiationPatternConfig(isotropic_probability=1.0)
        rng = np.random.default_rng(42)
        pat = generate_radiation_pattern(rng, cfg)
        assert pat.function_info["type"] == "isotropic"

        gains_before = evaluate_pattern_function_db(pat.function_info, _TEST_ANGLES)
        reloaded_info = _roundtrip_function_info(pat.function_info)
        gains_after = evaluate_pattern_function_db(reloaded_info, _TEST_ANGLES)

        np.testing.assert_array_equal(gains_before, gains_after)

    @pytest.mark.parametrize("symmetry", ["none", "x", "y", "xy"])
    def test_symmetry_roundtrip(self, symmetry):
        """Each symmetry mode survives JSON roundtrip with identical gains."""
        cfg = RadiationPatternConfig(
            isotropic_probability=0.0,
            symmetry_mode=symmetry,
        )
        rng = np.random.default_rng(99)
        pat = generate_radiation_pattern(rng, cfg)
        assert pat.symmetry == symmetry

        gains_before = evaluate_pattern_function_db(pat.function_info, _TEST_ANGLES)
        reloaded_info = _roundtrip_function_info(pat.function_info)
        gains_after = evaluate_pattern_function_db(reloaded_info, _TEST_ANGLES)

        np.testing.assert_array_equal(gains_before, gains_after)


class TestOldFormatRejection:
    def test_missing_pattern_function_raises(self):
        """NPZ without pattern_function metadata fails loudly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            arrays = {
                'reflectance': np.zeros((8, 8), dtype=np.float16),
                'transmittance': np.zeros((8, 8), dtype=np.float16),
                'pathloss': np.zeros((8, 8), dtype=np.float16),
                'radiation_pattern_db': np.zeros(360, dtype=np.float16),
                'mask': np.zeros((8, 8), dtype=np.uint8),
                'normals': np.zeros((8, 8, 2), dtype=np.float16),
            }
            meta = {
                'shape_hw': [8, 8],
                'antenna': {
                    'x_px': 4, 'y_px': 4, 'azimuth_deg': 0.0,
                    # No pattern_function key
                },
                'frequency_mhz': 1800,
            }
            npz_path = os.path.join(tmpdir, 'old.npz')
            meta_json = json.dumps(meta, separators=(",", ":"))
            np.savez_compressed(npz_path, **arrays,
                                meta_json=np.asarray(meta_json))
            with pytest.raises(ValueError, match="pattern_function"):
                radar_sample_from_npz(npz_path)


class TestCorruptedMetadata:
    def test_bad_version_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            arrays = {
                'reflectance': np.zeros((8, 8), dtype=np.float16),
                'transmittance': np.zeros((8, 8), dtype=np.float16),
                'pathloss': np.zeros((8, 8), dtype=np.float16),
                'radiation_pattern_db': np.zeros(360, dtype=np.float16),
                'mask': np.zeros((8, 8), dtype=np.uint8),
                'normals': np.zeros((8, 8, 2), dtype=np.float16),
            }
            bad_fn = {
                "version": 99, "type": "isotropic", "azimuth_deg": 0.0,
                "symmetry": "none", "max_loss_db": 20.0,
                "output_units": "db_gain_negative_pathloss",
            }
            meta = {
                'shape_hw': [8, 8],
                'antenna': {
                    'x_px': 4, 'y_px': 4, 'azimuth_deg': 0.0,
                    'pattern_function': bad_fn,
                },
                'frequency_mhz': 1800,
            }
            npz_path = os.path.join(tmpdir, 'bad.npz')
            meta_json = json.dumps(meta, separators=(",", ":"))
            np.savez_compressed(npz_path, **arrays,
                                meta_json=np.asarray(meta_json))
            with pytest.raises(ValueError, match="version"):
                radar_sample_from_npz(npz_path)
