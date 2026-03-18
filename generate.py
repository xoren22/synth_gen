import os
# Force Numba threading layer for all runs (set before importing approx/numba)
os.environ.setdefault("NUMBA_THREADING_LAYER", "tbb")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")  # default: disable multi-threaded numba speedup
import logging
import numpy as np
import torch

from tqdm import tqdm
import time

import json
import datetime

from approx import Approx
from models import RadarSample
from room_generator import generate_floor_scene
from antenna_pattern import RadiationPatternConfig, generate_radiation_pattern, evaluate_pattern_function_db



SAMPLES_PER_BUCKET = 1000


def _bucket_dir(out_root: str, idx: int) -> str:
	"""Return the bucket subdirectory for sample index *idx* and ensure it exists."""
	bucket = idx // SAMPLES_PER_BUCKET
	d = os.path.join(out_root, f"{bucket:06d}")
	os.makedirs(d, exist_ok=True)
	return d


def _export_sample_npz_json(out_root: str, sample_name: str, idx: int, arrays: dict, metadata: dict) -> str:
	"""
	Save arrays and metadata to {out_root}/{bucket}/{sample_name}.npz (compressed).
	Metadata is stored under the 'meta_json' key as a JSON string.
	Returns npz_path.
	"""
	bucket = _bucket_dir(out_root, idx)
	npz_path = os.path.join(bucket, f"{sample_name}.npz")

	# Normalize dtypes for compact storage; preserve float16 as-is
	np_arrays = {}
	for key, val in arrays.items():
		if isinstance(val, torch.Tensor):
			val = val.detach().cpu().numpy()
		if isinstance(val, np.ndarray):
			if val.dtype == np.float16:
				np_arrays[key] = val
			elif val.dtype == np.float64:
				np_arrays[key] = val.astype(np.float32, copy=False)
			else:
				np_arrays[key] = val
		else:
			raise TypeError(f"Array value for key '{key}' must be a numpy array or tensor, got {type(val)}")

	with open(npz_path, "wb") as f:
		meta_json = json.dumps(metadata, separators=(",", ":"))
		np.savez_compressed(f, **np_arrays, meta_json=np.asarray(meta_json))
	return npz_path


def _ensure_unique_run_dir(base_root: str, desired_run_id: str | None) -> tuple[str, str]:
	"""
	Return (out_dir, run_id) such that out_dir does not exist.
	If desired_run_id is None, use yyyy_mm_dd_hh_mm_ss; if exists, append _001, _002, ...
	If desired_run_id is provided and exists, append numeric suffix similarly.
	"""
	if desired_run_id and len(str(desired_run_id)) > 0:
		run_id = str(desired_run_id)
	else:
		now = datetime.datetime.now()
		run_id = now.strftime("%Y_%m_%d_%H_%M_%S")
	out_dir = os.path.join(base_root, run_id)
	if not os.path.exists(out_dir):
		return out_dir, run_id
	# Add incremental suffix
	i = 1
	while True:
		sfx = f"_{i:03d}"
		cand = os.path.join(base_root, run_id + sfx)
		if not os.path.exists(cand):
			return cand, run_id + sfx
		i += 1



def _reserve_sample_name(samples_dir: str, start_idx: int) -> tuple[str, int]:
    """
    Reserve a unique sample name by checking that no .npz with that name
    exists in its bucket yet. Returns (sample_name, idx).
    """
    idx = int(max(0, start_idx))
    while True:
        name = f"s{idx:012d}"
        bucket = _bucket_dir(samples_dir, idx)
        path = os.path.join(bucket, f"{name}.npz")
        if not os.path.exists(path):
            return name, idx
        idx += 1


def build_sample_from_generated(
	mask,
	normals,
	scene,
	reflectance,
	transmittance,
	dist_map,
	building_id: int = 0,
	pattern_cfg: RadiationPatternConfig | None = None,
	pattern_seed: int | None = None,
):
	H, W = mask.shape
	reflectance_t = torch.from_numpy(np.ascontiguousarray(reflectance, dtype=np.float32))
	transmittance_t = torch.from_numpy(np.ascontiguousarray(transmittance, dtype=np.float32))
	dist_map_t = torch.from_numpy(np.ascontiguousarray(dist_map, dtype=np.float32))
	# Pathloss is unknown before approximation; keep placeholder.
	pathloss_t = torch.zeros((H, W), dtype=torch.float32)

	ant = scene.get("antenna", {})
	x_ant = float(ant.get("x", 0))
	y_ant = float(ant.get("y", 0))
	freq_mhz = float(scene.get("frequency_mhz", 1800))

	cfg = pattern_cfg if pattern_cfg is not None else RadiationPatternConfig()
	pattern_rng = np.random.default_rng(pattern_seed)
	pattern_sample = generate_radiation_pattern(pattern_rng, cfg)

	# Canonical: evaluate the function on the export grid to produce the compatibility array.
	export_grid = np.arange(360, dtype=np.float64)
	canonical_gains = evaluate_pattern_function_db(pattern_sample.function_info, export_grid)
	radiation_pattern = torch.from_numpy(canonical_gains.astype(np.float32))

	scene["antenna_pattern"] = {
		"is_isotropic": bool(pattern_sample.is_isotropic),
		"azimuth_deg": float(pattern_sample.azimuth_deg),
		"symmetry": str(pattern_sample.symmetry),
		"model": str(pattern_sample.model),
		"style": str(pattern_sample.style),
		"complexity_dim": int(pattern_sample.complexity_dim),
		"units": "db_gain_negative_pathloss",
		"pattern_function": pattern_sample.function_info,
	}

	return RadarSample(
		H=H,
		W=W,
		x_ant=x_ant,
		y_ant=y_ant,
		azimuth=float(pattern_sample.azimuth_deg),
		freq_MHz=freq_mhz,
		reflectance=reflectance_t,
		transmittance=transmittance_t,
		dist_map=dist_map_t,
		pathloss=pathloss_t,
		radiation_pattern=radiation_pattern,
		pixel_size=0.25,
		mask=torch.from_numpy(mask.astype(np.float32)),
		normals=normals,
		radiation_pattern_fn_info=pattern_sample.function_info,
	)



def _generate_one(seed_i, freq_min, freq_max):
	"""Generate a single floor scene (sequential helper)."""
	if seed_i is not None:
		np.random.seed(seed_i)
	return generate_floor_scene(seed=seed_i, freq_min=freq_min, freq_max=freq_max)


def _export_one(sample, mask, normals, refl, trans, scene, gidx, pred_t, samples_dir, pbar=None, timing=None, extra_meta=None):
	"""Export a single predicted sample to disk (shared by threaded and sequential modes)."""
	pred = pred_t.cpu().numpy() if hasattr(pred_t, 'cpu') else np.array(pred_t)
	sample_name, sample_idx = _reserve_sample_name(samples_dir, gidx)
	arrays = {
		'normals': normals.astype(np.float16, copy=False),
		'reflectance': refl.astype(np.float16, copy=False),
		'transmittance': trans.astype(np.float16, copy=False),
		'mask': mask.astype(np.uint8, copy=False),
		'pathloss': pred.astype(np.float16, copy=False),
		'radiation_pattern_db': sample.radiation_pattern.detach().cpu().numpy().astype(np.float16, copy=False),
	}
	canvas = scene.get('canvas', {})
	ant_pattern = scene.get('antenna_pattern', {})
	metadata = {
		'sample_name': sample_name,
		'shape_hw': [int(sample.H), int(sample.W)],
		'pixel_size_m': float(sample.pixel_size),
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
		'frequency_mhz': int(scene.get('frequency_mhz', 1800)),
		'canvas': {'width_m': float(canvas.get('width_m', 0.0)), 'height_m': float(canvas.get('height_m', 0.0))},
		'created_at_unix_s': float(time.time()),
	}
	if timing is not None:
		metadata['timing_s'] = timing
	if extra_meta is not None:
		metadata.update(extra_meta)
	_export_sample_npz_json(samples_dir, sample_name, sample_idx, arrays, metadata)
	if pbar is not None:
		pbar.update(1)



def main():
	import argparse
	logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

	parser = argparse.ArgumentParser("Synthetic data generation pipeline")
	parser.add_argument('--num', type=int, default=5, help='Number of rooms to generate and approximate')
	parser.add_argument('--numba_threads', type=int, default=1, help='Numba threads for prange kernels (default 1 disables multi-thread speedup)')
	parser.add_argument('--seed', type=int, default=None, help='Base seed for deterministic generation (per-sample: seed+index)')
	parser.add_argument('--run_id', type=str, default=None, help='Unique run identifier; auto-generated if omitted')
	parser.add_argument('--out_root', type=str, default=None, help='Root directory for streamed outputs (default: <repo>/data)')
	parser.add_argument('--freq_min', type=int, default=400, help='Minimum frequency in MHz for uniform sampling')
	parser.add_argument('--freq_max', type=int, default=10_000, help='Maximum frequency in MHz for uniform sampling')
	parser.add_argument('--ant_iso_prob', type=float, default=0.25, help='Probability of isotropic antenna pattern')
	parser.add_argument('--ant_latent_dim_min', type=int, default=8, help='Minimum latent dimension d (uniformly sampled per sample)')
	parser.add_argument('--ant_latent_dim_max', type=int, default=20, help='Maximum latent dimension d (uniformly sampled per sample)')
	parser.add_argument('--ant_fourier_order_min', type=int, default=10, help='Minimum Fourier harmonics K (uniform per sample)')
	parser.add_argument('--ant_fourier_order_max', type=int, default=24, help='Maximum Fourier harmonics K (uniform per sample)')
	parser.add_argument('--ant_petal_order_min', type=int, default=3, help='Minimum petal count for petal style')
	parser.add_argument('--ant_petal_order_max', type=int, default=12, help='Maximum petal count for petal style')
	parser.add_argument('--ant_db_max', type=float, default=40.0, help='Maximum dB loss in radiation pattern')
	parser.add_argument('--ant_symmetry_mode', type=str, default='random', choices=['random', 'none', 'x', 'y', 'xy'], help='Symmetry mode; random samples uniformly among none/x/y/xy')
	parser.add_argument('--max_refl', type=int, default=None, help='Max reflections for ray tracer (default: use approx.MAX_REFL)')
	parser.add_argument('--verbose', action='store_true', help='Enable detailed per-step timing logs (DEBUG level)')
	args = parser.parse_args()

	if args.verbose:
		logging.getLogger().setLevel(logging.DEBUG)
		# Suppress noisy third-party debug logs
		for name in ('numba', 'llvmlite', 'matplotlib'):
			logging.getLogger(name).setLevel(logging.WARNING)

	N = int(max(1, args.num))
	chosen_numba_threads = max(1, int(args.numba_threads)) if args.numba_threads is not None else 1

	# Resolve output root for streaming pipeline.
	# Default is repo_root/data so each run writes to data/{run_id}.
	if args.out_root and len(str(args.out_root)) > 0:
		base_out_root = args.out_root if os.path.isabs(args.out_root) else os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out_root)
	else:
		base_out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
	os.makedirs(base_out_root, exist_ok=True)

	# Unique run identifier and output dir (non-overwriting)
	out_dir, run_id = _ensure_unique_run_dir(base_out_root, args.run_id)
	os.makedirs(out_dir, exist_ok=True)
	logging.info(f"Run ID: {run_id}; outputs -> {out_dir}")

	# Resolve max_refl / max_trans
	from approx import MAX_REFL as DEFAULT_MAX_REFL, MAX_TRANS as DEFAULT_MAX_TRANS
	max_refl = args.max_refl if args.max_refl is not None else DEFAULT_MAX_REFL
	max_trans = DEFAULT_MAX_TRANS
	logging.info(f"Processing {N} samples sequentially in one process... (max_refl={max_refl}, max_trans={max_trans})")
	model = Approx()
	samples_dir = out_dir
	os.makedirs(samples_dir, exist_ok=True)

	# Determine seed base once
	if args.seed is not None:
		seed_base = int(args.seed)
	else:
		seed_base = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])

	logging.info(f"Numba threads: {chosen_numba_threads} (1 = effectively no numba multithreading)")
	try:
		import numba as _nb
		_nb.set_num_threads(chosen_numba_threads)
	except Exception:
		pass
	pattern_cfg = RadiationPatternConfig(
			latent_dim_min=int(args.ant_latent_dim_min),
			latent_dim_max=int(args.ant_latent_dim_max),
			fourier_order_min=int(args.ant_fourier_order_min),
			fourier_order_max=int(args.ant_fourier_order_max),
			petal_order_min=int(args.ant_petal_order_min),
			petal_order_max=int(args.ant_petal_order_max),
			isotropic_probability=float(args.ant_iso_prob),
			max_loss_db=float(args.ant_db_max),
			symmetry_mode=str(args.ant_symmetry_mode),
		)
	logging.info(
		"Antenna pattern config: d=[%d,%d] K=[%d,%d] p_iso=%.3f db_max=%.2f "
		"(effective per-sample max is uniform in [0, db_max]) petals=[%d,%d] symmetry=%s",
		pattern_cfg.latent_dim_min,
		pattern_cfg.latent_dim_max,
		pattern_cfg.fourier_order_min,
		pattern_cfg.fourier_order_max,
		pattern_cfg.isotropic_probability,
		pattern_cfg.max_loss_db,
		pattern_cfg.petal_order_min,
		pattern_cfg.petal_order_max,
		pattern_cfg.symmetry_mode,
	)

	pbar = tqdm(total=N, desc='Samples')
	t_start = time.perf_counter()
	for gidx in range(N):
		seed_i = int(seed_base) + gidx
		t0 = time.perf_counter()
		mask, normals, scene, refl, trans, dist = _generate_one(seed_i, args.freq_min, args.freq_max)
		sample = build_sample_from_generated(
			mask,
			normals,
			scene,
			refl,
			trans,
			dist,
			building_id=gidx,
			pattern_cfg=pattern_cfg,
			pattern_seed=seed_i + 777_777,
		)
		t1 = time.perf_counter()
		pred = model.approximate(sample, max_refl=max_refl)
		t2 = time.perf_counter()
		_export_one(sample, mask, normals, refl, trans, scene, gidx, pred, samples_dir, pbar,
			timing={'scene_s': round(t1 - t0, 6), 'raytrace_s': round(t2 - t1, 6)},
			extra_meta={'max_refl': max_refl, 'max_trans': max_trans})

	pbar.close()

	elapsed = time.perf_counter() - t_start
	logging.info(
		f"Done: {N} samples in {elapsed:.1f}s ({N/max(elapsed,0.001):.1f} samples/sec)"
	)


if __name__ == "__main__":
	main()
