# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synthetic radio-frequency pathloss dataset generator. Generates random 2D floor plans, places an antenna with a randomly-sampled radiation pattern, then ray-traces transmission/reflection losses to produce per-pixel pathloss maps. Output is compressed `.npz` files with JSON metadata.

## Commands

### Run tests
```bash
python -m pytest tests/                          # all tests
python -m pytest tests/test_radiation_pattern.py  # single file
python -m pytest tests/ -m "not benchmark"        # skip perf benchmarks
python -m pytest tests/ -m benchmark              # only benchmarks
```

### Generate samples
```bash
# Single-process (small runs / debugging)
python generate.py --num 5 --seed 42

# Multi-worker with orchestrator + Streamlit dashboard
python unified_runner.py start --workers 4 --num-samples 20
python unified_runner.py status                   # check progress
```

### Visualization
```bash
python scripts/visualize_samples.py               # view generated samples
python scripts/visualize_pattern_samples.py        # view radiation patterns
python scripts/visualize_pattern_catalog.py        # pattern style catalog
```

## Architecture

### Pipeline flow
`generate.py:main` → for each sample:
1. `room_generator.generate_floor_scene()` — random 2D floor plan (walls, doors, materials)
2. `antenna_pattern.generate_radiation_pattern()` — sample a radiation pattern (isotropic or latent Fourier)
3. `approx.Approx.approximate()` — Numba-JIT ray tracer computes pathloss
4. Export to `{run_dir}/{sample_name}/{sample_name}.npz`

### Core modules
- **`models.py`** — `RadarSample` dataclass: the central data structure passed between all pipeline stages. Contains grid dimensions, antenna position, material maps, and the `radiation_pattern_fn_info` dict.
- **`antenna_pattern.py`** — Radiation pattern generation and evaluation. `function_info` dict is the **sole source of truth** for patterns (not the discretized 360-element array). Two pattern types: `isotropic` (0 dB everywhere) and `latent_fourier` (parameterized by Fourier coefficients). Four styles: `front_back`, `bidirectional`, `petal`, `ripple`. Includes `validate_pattern_function_info()` for schema validation.
- **`approx.py`** — Numba-accelerated ray tracer. `calculate_combined_loss_with_normals` traces rays with reflection/transmission. `_backfill_direct_los` fills uncovered pixels via LOS. First call triggers JIT warmup.
- **`room_generator.py`** — Procedural floor plan generator using BSP partitioning. Produces wall mask, normals, reflectance/transmittance maps, and distance map.
- **`normal_parser.py`** — PCA-based wall normal estimation from binary mask. Multi-scale with trimmed fitting for robustness.
- **`pattern_function_io.py`** — NPZ loader that reconstructs `RadarSample` from saved files using `function_info` metadata.
- **`unified_runner.py`** — Multi-worker orchestrator. Spawns subprocess workers, monitors heartbeats, auto-restarts dead workers, serves Streamlit dashboard.

### Key design decisions
- **`function_info` is the source of truth** for radiation patterns. The 360-element `radiation_pattern` array is a derived export convenience, not the authoritative representation. The ray tracer and pixel-map builder call `evaluate_pattern_function_db()` to get continuous gains at arbitrary angles.
- **No try/except in feature-scope code**: `pattern_function_io.py` and pattern evaluation functions in `antenna_pattern.py` must not contain try/except blocks. This is enforced by `test_no_try_except_feature_scope.py`.
- **Numba threading**: Set via `NUMBA_THREADING_LAYER=tbb` and `NUMBA_NUM_THREADS` env vars before import. Default is single-threaded.
- Pattern output units are `db_gain_negative_pathloss` (gains are <= 0 dB, representing signal loss).

## Test conventions
- Tests use `sys.path.insert(0, ...)` to find the repo root (no package install needed).
- Benchmark tests are marked `@pytest.mark.benchmark` — excluded from normal runs.
- Tests validate schema correctness, generator-evaluator coherence, geometric regressions, and the ray tracer contract (function-based evaluation, not array lookup).
