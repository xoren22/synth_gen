import os
# Force Numba threading layer for all runs (set before importing approx/numba)
os.environ.setdefault("NUMBA_THREADING_LAYER", "tbb")
import logging
import numpy as np
import torch

from tqdm import tqdm
import time

import json
import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from approx import Approx
from models import RadarSample
from room_generator import generate_floor_scene



def _export_sample_npz_json(out_root: str, sample_name: str, arrays: dict, metadata: dict) -> tuple[str, str]:
	"""
	Save arrays to {out_root}/{sample_name}/{sample_name}.npz (compressed) and metadata JSON alongside.
	Returns (npz_path, json_path).
	"""
	sample_dir = os.path.join(out_root, sample_name)
	os.makedirs(sample_dir, exist_ok=True)
	npz_path = os.path.join(sample_dir, f"{sample_name}.npz")
	json_path = os.path.join(sample_dir, f"{sample_name}.json")

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
		np.savez_compressed(f, **np_arrays)
	with open(json_path, "w") as f:
		json.dump(metadata, f, indent=2)
	return npz_path, json_path


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



def _reserve_sample_dir(samples_dir: str, start_idx: int) -> str:
    """
    Reserve a unique sample directory by creating it without scanning the
    whole directory. Avoids os.listdir on huge directories.
    Returns the created sample name (e.g., 's000123').
    """
    idx = int(max(0, start_idx))
    while True:
        name = f"s{idx:012d}"
        path = os.path.join(samples_dir, name)
        try:
            os.makedirs(path, exist_ok=False)
            return name
        except FileExistsError:
            idx += 1
            continue
        except OSError:
            # On transient I/O errors, advance and retry without directory listing
            time.sleep(0.01)
            idx += 1


def build_sample_from_generated(mask, normals, scene, reflectance, transmittance, dist_map, building_id: int = 0):
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

	# Radiation pattern placeholder: isotropic 360 values
	radiation_pattern = torch.ones(360, dtype=torch.float32)

	return RadarSample(
		H=H,
		W=W,
		x_ant=x_ant,
		y_ant=y_ant,
		azimuth=0.0,
		freq_MHz=freq_mhz,
		reflectance=reflectance_t,
		transmittance=transmittance_t,
		dist_map=dist_map_t,
		pathloss=pathloss_t,
		radiation_pattern=radiation_pattern,
		pixel_size=0.25,
		mask=torch.from_numpy(mask.astype(np.float32)),
		normals=normals,
	)



def _generate_one(args_tuple):
	"""Top-level function for ProcessPoolExecutor: generate a single floor scene."""
	seed_i, freq_min, freq_max = args_tuple
	if seed_i is not None:
		np.random.seed(seed_i)
	return generate_floor_scene(seed=seed_i, freq_min=freq_min, freq_max=freq_max)


def _submit_generation_batch(gen_pool, batch_indices, seed_base, freq_min, freq_max):
	"""Submit a batch of generation tasks as individual futures for pipelining."""
	futures = []
	for gidx in batch_indices:
		seed_i = int(seed_base) + int(gidx)
		futures.append(gen_pool.submit(_generate_one, (seed_i, freq_min, freq_max)))
	return futures


def _export_batch(batch_data, preds_data, samples_dir_path, start_idx):
	"""Export a batch of samples to disk. Runs in a background thread."""
	export_time = 0.0
	for (sample, mask, normals, refl, trans, dist, scene, gidx), pred_t in zip(batch_data, preds_data):
		pred = pred_t.cpu().numpy() if hasattr(pred_t, 'cpu') else np.array(pred_t)
		freq_mhz = int(scene.get('frequency_mhz', 1800))
		sample_name = _reserve_sample_dir(samples_dir_path, gidx)
		arrays = {
			'normals': normals.astype(np.float16, copy=False),
			'reflectance': refl.astype(np.float16, copy=False),
			'transmittance': trans.astype(np.float16, copy=False),
			'mask': mask.astype(np.uint8, copy=False),
			'pathloss': pred.astype(np.float16, copy=False),
		}
		canvas = scene.get('canvas', {})
		metadata = {
			'sample_name': sample_name,
			'shape_hw': [int(sample.H), int(sample.W)],
			'pixel_size_m': float(sample.pixel_size),
			'antenna': {'x_px': int(sample.x_ant), 'y_px': int(sample.y_ant)},
			'frequency_mhz': int(freq_mhz),
			'canvas': {'width_m': float(canvas.get('width_m', 0.0)), 'height_m': float(canvas.get('height_m', 0.0))},
			'created_at_unix_s': float(time.time()),
		}
		t_e = time.perf_counter()
		_export_sample_npz_json(samples_dir_path, sample_name, arrays, metadata)
		export_time += time.perf_counter() - t_e
	return export_time



def main():
	import argparse
	logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

	parser = argparse.ArgumentParser("Synthetic data generation pipeline")
	parser.add_argument('--num', type=int, default=5, help='Number of rooms to generate and approximate')
	parser.add_argument('--batch_size', type=int, default=96, help='Batch size for generate->predict->save streaming')
	parser.add_argument('--numba_threads', type=int, default=1, help='Numba threads per worker (0 = auto)')
	parser.add_argument('--workers', type=int, default=max((mp.cpu_count() or 1) - 5, 1), help='Number of workers for model.predict (1=sequential)')
	parser.add_argument('--seed', type=int, default=None, help='Base seed for deterministic generation (per-sample: seed+index)')
	parser.add_argument('--run_id', type=str, default=None, help='Unique run identifier; auto-generated if omitted')
	parser.add_argument('--out_root', type=str, default=None, help='Root directory for streamed outputs (default: ~/synth_gen/data/synthetic)')
	parser.add_argument('--freq_min', type=int, default=400, help='Minimum frequency in MHz for uniform sampling')
	parser.add_argument('--freq_max', type=int, default=10_000, help='Maximum frequency in MHz for uniform sampling')
	parser.add_argument('--verbose', action='store_true', help='Enable detailed per-step timing logs (DEBUG level)')
	args = parser.parse_args()

	if args.verbose:
		logging.getLogger().setLevel(logging.DEBUG)
		# Suppress noisy third-party debug logs
		for name in ('numba', 'llvmlite', 'matplotlib'):
			logging.getLogger(name).setLevel(logging.WARNING)

	N = int(max(1, args.num))
	B = int(max(1, args.batch_size))

	# Threads backend is always used
	chosen_numba_threads = int(args.numba_threads) if (args.numba_threads and args.numba_threads > 0) else 0
	chosen_workers = int(max(1, args.workers))

	# Resolve output root for streaming pipeline
	if args.out_root and len(str(args.out_root)) > 0:
		base_out_root = args.out_root if os.path.isabs(args.out_root) else os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out_root)
	else:
		base_out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'synthetic')
	os.makedirs(base_out_root, exist_ok=True)

	# Unique run identifier and output dir (non-overwriting)
	out_dir, run_id = _ensure_unique_run_dir(base_out_root, args.run_id)
	os.makedirs(out_dir, exist_ok=True)
	logging.info(f"Run ID: {run_id}; outputs -> {out_dir}")

	# Generate->predict->save in batches (save NPZ+JSON per sample)
	logging.info(f"Processing {N} samples in batches of {B} (backend=threads, workers={chosen_workers})...")
	model = Approx()
	global_idx = 0
	samples_dir = out_dir
	os.makedirs(samples_dir, exist_ok=True)
	sample_logger = logging.getLogger('generate.sample')
	# Running totals across all batches
	tot_gen = 0.0
	tot_build_sample = 0.0
	tot_predict = 0.0
	tot_export = 0.0

	cpu_count = mp.cpu_count() or 2
	gen_pool_size = min(B, max(1, cpu_count // 2))
	gen_pool = ProcessPoolExecutor(max_workers=gen_pool_size, mp_context=mp.get_context('spawn'))
	export_pool = ThreadPoolExecutor(max_workers=2)

	batch_starts = list(range(0, N, B))
	pending_export_futures = []

	# Determine seed base once
	if args.seed is not None:
		seed_base = int(args.seed)
	else:
		seed_base = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])

	# Pre-submit first batch generation so it overlaps with nothing initially,
	# then subsequent batches overlap generation with predict/export.
	first_end = min(B, N)
	current_gen_futures = _submit_generation_batch(
		gen_pool, list(range(0, first_end)), seed_base, args.freq_min, args.freq_max
	)

	for batch_idx, start in enumerate(tqdm(batch_starts, desc='Batches')):
		end = min(start + B, N)
		batch_build = 0.0
		batch_predict = 0.0
		batch_export = 0.0

		# Collect current batch generation results
		t0 = time.perf_counter()
		gen_results = [f.result() for f in current_gen_futures]
		batch_gen = time.perf_counter() - t0

		# Pre-submit next batch generation before predict/export (true pipelining)
		if batch_idx + 1 < len(batch_starts):
			next_start = batch_starts[batch_idx + 1]
			next_end = min(next_start + B, N)
			current_gen_futures = _submit_generation_batch(
				gen_pool, list(range(next_start, next_end)), seed_base, args.freq_min, args.freq_max
			)
		else:
			current_gen_futures = []

		# Build samples
		batch = []
		t0 = time.perf_counter()
		for i, (mask, normals, scene, refl, trans, dist) in enumerate(gen_results):
			gidx = start + i
			sample = build_sample_from_generated(mask, normals, scene, refl, trans, dist, building_id=gidx)
			sample_logger.debug(
				"sample #%d [%dx%d]: built",
				gidx, mask.shape[1], mask.shape[0]
			)
			batch.append((sample, mask, normals, refl, trans, dist, scene, gidx))
		batch_build = time.perf_counter() - t0

		# Predict for this batch
		samples = [t[0] for t in batch]
		t0 = time.perf_counter()
		preds = model.predict(samples, num_workers=chosen_workers, numba_threads=chosen_numba_threads, backend='threads')
		batch_predict = time.perf_counter() - t0

		# Wait for any prior export to complete and collect its time
		for fut in pending_export_futures:
			batch_export += fut.result()
		pending_export_futures.clear()

		# Submit this batch's export asynchronously
		export_fut = export_pool.submit(_export_batch, batch, preds, samples_dir, start)
		pending_export_futures.append(export_fut)

		global_idx = end
		tot_gen += batch_gen
		tot_build_sample += batch_build
		tot_predict += batch_predict
		tot_export += batch_export
		batch_total = batch_gen + batch_build + batch_predict + batch_export
		batch_count = max(1, end - start)
		total_this_sample = batch_total / batch_count
		total_so_far = tot_gen + tot_build_sample + tot_predict + tot_export
		logging.info(
			f"Batch {batch_idx+1}: gen={batch_gen:.3f}s, build_sample={batch_build:.3f}s, "
			f"predict={batch_predict:.3f}s, export={batch_export:.3f}s, total_this_sample={total_this_sample:.3f}s"
		)
		logging.info(
			f"Totals so far ({end}/{N}): gen={tot_gen:.3f}s, build_sample={tot_build_sample:.3f}s, "
			f"predict={tot_predict:.3f}s, export={tot_export:.3f}s, total_so_far={total_so_far:.3f}s"
		)

	# Wait for final export
	for fut in pending_export_futures:
		tot_export += fut.result()
	pending_export_futures.clear()

	gen_pool.shutdown(wait=False)
	export_pool.shutdown(wait=True)

	logging.info(
		f"Final totals ({N}/{N}): gen={tot_gen:.3f}s, build_sample={tot_build_sample:.3f}s, "
		f"predict={tot_predict:.3f}s, export={tot_export:.3f}s, "
		f"total_so_far={tot_gen + tot_build_sample + tot_predict + tot_export:.3f}s"
	)


if __name__ == "__main__":
	main()
