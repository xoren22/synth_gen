"""Tests for bucketed output directory structure.

Verifies that samples are placed into bucket subdirectories and that
_count_done correctly counts across buckets.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tempfile
import json
import numpy as np
import pytest

from generate import (
    SAMPLES_PER_BUCKET,
    _bucket_dir,
    _export_sample_npz_json,
    _reserve_sample_name,
)


def _dummy_arrays():
    return {
        'normals': np.zeros((4, 4, 2), dtype=np.float16),
        'reflectance': np.zeros((4, 4), dtype=np.float16),
        'transmittance': np.zeros((4, 4), dtype=np.float16),
        'mask': np.zeros((4, 4), dtype=np.uint8),
        'pathloss': np.zeros((4, 4), dtype=np.float16),
        'radiation_pattern_db': np.zeros(360, dtype=np.float16),
    }


def _dummy_metadata(name):
    return {'sample_name': name, 'shape_hw': [4, 4]}


class TestBucketDir:
    def test_bucket_zero(self):
        with tempfile.TemporaryDirectory() as td:
            d = _bucket_dir(td, 0)
            assert os.path.basename(d) == "000000"
            assert os.path.isdir(d)

    def test_bucket_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            d0 = _bucket_dir(td, SAMPLES_PER_BUCKET - 1)
            d1 = _bucket_dir(td, SAMPLES_PER_BUCKET)
            assert os.path.basename(d0) == "000000"
            assert os.path.basename(d1) == "000001"

    def test_large_index(self):
        with tempfile.TemporaryDirectory() as td:
            d = _bucket_dir(td, 555_555)
            expected = f"{555_555 // SAMPLES_PER_BUCKET:06d}"
            assert os.path.basename(d) == expected


class TestExportToBucket:
    def test_file_lands_in_correct_bucket(self):
        with tempfile.TemporaryDirectory() as td:
            name = "s000000000042"
            path = _export_sample_npz_json(td, name, 42, _dummy_arrays(), _dummy_metadata(name))
            assert os.path.isfile(path)
            assert os.path.basename(os.path.dirname(path)) == "000000"
            assert os.path.basename(path) == f"{name}.npz"

    def test_second_bucket(self):
        with tempfile.TemporaryDirectory() as td:
            idx = SAMPLES_PER_BUCKET + 5
            name = f"s{idx:012d}"
            path = _export_sample_npz_json(td, name, idx, _dummy_arrays(), _dummy_metadata(name))
            assert os.path.basename(os.path.dirname(path)) == "000001"

    def test_npz_is_loadable(self):
        with tempfile.TemporaryDirectory() as td:
            name = "s000000000000"
            path = _export_sample_npz_json(td, name, 0, _dummy_arrays(), _dummy_metadata(name))
            data = np.load(path)
            meta = json.loads(str(data['meta_json']))
            assert meta['sample_name'] == name


class TestReserveSampleName:
    def test_first_reservation(self):
        with tempfile.TemporaryDirectory() as td:
            name, idx = _reserve_sample_name(td, 0)
            assert name == "s000000000000"
            assert idx == 0

    def test_skips_existing(self):
        with tempfile.TemporaryDirectory() as td:
            # Create first sample
            _export_sample_npz_json(td, "s000000000000", 0, _dummy_arrays(), _dummy_metadata("s000000000000"))
            name, idx = _reserve_sample_name(td, 0)
            assert name == "s000000000001"
            assert idx == 1

    def test_cross_bucket_skip(self):
        with tempfile.TemporaryDirectory() as td:
            # Fill the last slot of bucket 0
            last_idx = SAMPLES_PER_BUCKET - 1
            last_name = f"s{last_idx:012d}"
            _export_sample_npz_json(td, last_name, last_idx, _dummy_arrays(), _dummy_metadata(last_name))
            # Reserving at that index should skip to SAMPLES_PER_BUCKET
            name, idx = _reserve_sample_name(td, last_idx)
            assert idx == SAMPLES_PER_BUCKET
            expected_bucket = f"{SAMPLES_PER_BUCKET // SAMPLES_PER_BUCKET:06d}"
            bucket_path = os.path.join(td, expected_bucket)
            assert os.path.isdir(bucket_path)


class TestCountDone:
    def test_empty_dir(self):
        # Import here to avoid numba issues at module level
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from unified_runner import _count_done
        with tempfile.TemporaryDirectory() as td:
            assert _count_done(td) == 0

    def test_nonexistent_dir(self):
        from unified_runner import _count_done
        assert _count_done("/nonexistent/path") == 0

    def test_counts_across_buckets(self):
        from unified_runner import _count_done
        with tempfile.TemporaryDirectory() as td:
            # Put 3 samples in bucket 000000
            for i in range(3):
                name = f"s{i:012d}"
                _export_sample_npz_json(td, name, i, _dummy_arrays(), _dummy_metadata(name))
            # Put 2 samples in bucket 000001
            for i in range(SAMPLES_PER_BUCKET, SAMPLES_PER_BUCKET + 2):
                name = f"s{i:012d}"
                _export_sample_npz_json(td, name, i, _dummy_arrays(), _dummy_metadata(name))
            assert _count_done(td) == 5

    def test_ignores_empty_npz(self):
        from unified_runner import _count_done
        with tempfile.TemporaryDirectory() as td:
            bucket = os.path.join(td, "000000")
            os.makedirs(bucket)
            # Create a valid sample
            name = "s000000000000"
            _export_sample_npz_json(td, name, 0, _dummy_arrays(), _dummy_metadata(name))
            # Create an empty file
            open(os.path.join(bucket, "s000000000001.npz"), "w").close()
            assert _count_done(td) == 1
