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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import queue
import threading

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


def _predict_worker(model, gen_queue, export_queue, numba_threads):
	"""Worker thread: pull from gen_queue, run approximate(), push to export_queue."""
	try:
		import numba as _nb
		if numba_threads and numba_threads > 0:
			_nb.set_num_threads(numba_threads)
	except Exception:
		pass
	while True:
		item = gen_queue.get()
		if item is None:
			break
		sample, mask, normals, refl, trans, dist, scene, gidx = item
		pred = model.approximate(sample)
		export_queue.put((sample, mask, normals, refl, trans, dist, scene, gidx, pred))


def _export_worker(export_queue, samples_dir, pbar):
	"""Worker thread: pull from export_queue, write to disk, update progress."""
	while True:
		item = export_queue.get()
		if item is None:
			break
		sample, mask, normals, refl, trans, dist, scene, gidx, pred_t = item
		pred = pred_t.cpu().numpy() if hasattr(pred_t, 'cpu') else np.array(pred_t)
		sample_name = _reserve_sample_dir(samples_dir, gidx)
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
			'frequency_mhz': int(scene.get('frequency_mhz', 1800)),
			'canvas': {'width_m': float(canvas.get('width_m', 0.0)), 'height_m': float(canvas.get('height_m', 0.0))},
			'created_at_unix_s': float(time.time()),
		}
		_export_sample_npz_json(samples_dir, sample_name, arrays, metadata)
		pbar.update(1)



def main():
	import argparse
	logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

	parser = argparse.ArgumentParser("Synthetic data generation pipeline")
	parser.add_argument('--num', type=int, default=5, help='Number of rooms to generate and approximate')
	parser.add_argument('--batch_size', type=int, default=96, help='Batch size for generate->predict->save streaming')
	parser.add_argument('--numba_threads', type=int, default=1, help='Numba threads per worker (0 = auto)')
	parser.add_argument('--workers', type=int, default=1, help='Number of predict worker threads (1=sequential)')
	parser.add_argument('--seed', type=int, default=None, help='Base seed for deterministic generation (per-sample: seed+index)')
	parser.add_argument('--run_id', type=str, default=None, help='Unique run identifier; auto-generated if omitted')
	parser.add_argument('--out_root', type=str, default=None, help='Root directory for streamed outputs (default: ~/synth_gen/data/synthetic)')
	parser.add_argument('--freq_min', type=int, default=400, help='Minimum frequency in MHz for uniform sampling')
	parser.add_argument('--freq_max', type=int, default=10_000, help='Maximum frequency in MHz for uniform sampling')
	parser.add_argument('--gen_workers', type=int, default=1, help='Max workers for room generation pool')
	parser.add_argument('--export_workers', type=int, default=1, help='Max workers for export thread pool')
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

	# Continuous sample-level pipeline: gen -> predict -> export (no batch sync)
	logging.info(f"Processing {N} samples continuously (workers={chosen_workers})...")
	model = Approx()
	samples_dir = out_dir
	os.makedirs(samples_dir, exist_ok=True)

	cpu_count = mp.cpu_count() or 2
	if args.gen_workers and args.gen_workers > 0:
		gen_pool_size = min(N, args.gen_workers)
	else:
		gen_pool_size = min(N, max(1, cpu_count // 2))
	export_worker_count = max(1, args.export_workers) if args.export_workers and args.export_workers > 0 else 2

	# Bounded queues limit memory growth
	gen_queue = queue.Queue(maxsize=chosen_workers * 2)
	export_queue = queue.Queue(maxsize=chosen_workers * 2)

	gen_pool = ProcessPoolExecutor(max_workers=gen_pool_size, mp_context=mp.get_context('spawn'))
	logging.info(f"Pools: gen_workers={gen_pool_size}, export_workers={export_worker_count}, predict_workers={chosen_workers}, numba_threads={chosen_numba_threads or 'auto'}")

	# Determine seed base once
	if args.seed is not None:
		seed_base = int(args.seed)
	else:
		seed_base = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])

	pbar = tqdm(total=N, desc='Samples')
	t_start = time.perf_counter()

	# Start predict worker threads
	predict_threads = []
	for _ in range(chosen_workers):
		t = threading.Thread(
			target=_predict_worker,
			args=(model, gen_queue, export_queue, chosen_numba_threads),
			daemon=True,
		)
		t.start()
		predict_threads.append(t)

	# Start export worker threads
	export_threads = []
	for _ in range(export_worker_count):
		t = threading.Thread(
			target=_export_worker,
			args=(export_queue, samples_dir, pbar),
			daemon=True,
		)
		t.start()
		export_threads.append(t)

	# Submit all generation tasks to the process pool
	gen_futures = {}
	for gidx in range(N):
		seed_i = int(seed_base) + gidx
		fut = gen_pool.submit(_generate_one, (seed_i, args.freq_min, args.freq_max))
		gen_futures[fut] = gidx

	# Feed generated results into predict queue as they complete
	for fut in as_completed(gen_futures):
		gidx = gen_futures[fut]
		mask, normals, scene, refl, trans, dist = fut.result()
		sample = build_sample_from_generated(mask, normals, scene, refl, trans, dist, building_id=gidx)
		gen_queue.put((sample, mask, normals, refl, trans, dist, scene, gidx))

	# Signal predict workers to stop (one sentinel per worker)
	for _ in predict_threads:
		gen_queue.put(None)
	for t in predict_threads:
		t.join()

	# Signal export workers to stop
	for _ in export_threads:
		export_queue.put(None)
	for t in export_threads:
		t.join()

	pbar.close()
	gen_pool.shutdown(wait=False)

	elapsed = time.perf_counter() - t_start
	logging.info(
		f"Done: {N} samples in {elapsed:.1f}s ({N/max(elapsed,0.001):.1f} samples/sec)"
	)


if __name__ == "__main__":
	main()
