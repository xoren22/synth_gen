#!/usr/bin/env python3
"""
Benchmark grid for batch_size, numba_threads, and workers.
Tests the full generate->predict->export pipeline and reports per-sample speed.
"""
import os
os.environ.setdefault("NUMBA_THREADING_LAYER", "tbb")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import shutil
import subprocess
import sys
import multiprocessing as mp
import json
import tempfile

CPU_COUNT = mp.cpu_count()
NUM_SAMPLES = 48  # enough to get stable timings across 2+ batches

# Grid: (workers, numba_threads, batch_size)
# Key insight: workers * numba_threads ≈ CPU_COUNT for optimal utilization
# We also need to leave some headroom for the generation pool

CONFIGS = []

# Phase 1: Vary workers/numba_threads ratio at batch_size=48
for workers, nt in [
    (1, 24),    # sequential prediction, max threads per sample
    (2, 12),    # 2 concurrent samples, 12 threads each
    (3, 8),     # 3 concurrent, 8 threads each
    (4, 6),     # 4 concurrent, 6 threads each
    (6, 4),     # 6 concurrent, 4 threads each
    (8, 3),     # 8 concurrent, 3 threads each
    (12, 2),    # 12 concurrent, 2 threads each
    (24, 1),    # 24 concurrent, 1 thread each (max sample parallelism)
    (19, 1),    # default config (cpu-5 workers, 1 thread)
    (20, 1),    # slightly above default
    (16, 1),    # moderate parallelism
]:
    CONFIGS.append((workers, nt, 48))

# Phase 2: Vary batch_size with the most promising worker/thread ratios
for bs in [16, 24, 32, 64, 96]:
    for workers, nt in [(24, 1), (12, 2), (8, 3)]:
        CONFIGS.append((workers, nt, bs))

# Deduplicate
seen = set()
unique_configs = []
for c in CONFIGS:
    if c not in seen:
        seen.add(c)
        unique_configs.append(c)
CONFIGS = unique_configs

def run_one_config(workers, numba_threads, batch_size, num_samples, run_idx):
    """Run generate.py with given config and return (wall_time, per_sample_time, gen_time, predict_time, export_time)."""
    out_dir = tempfile.mkdtemp(prefix=f"bench_w{workers}_nt{numba_threads}_bs{batch_size}_")

    cmd = [
        sys.executable, "generate.py",
        f"--num={num_samples}",
        f"--batch_size={batch_size}",
        f"--numba_threads={numba_threads}",
        f"--workers={workers}",
        f"--seed=42",
        f"--out_root={out_dir}",
        f"--run_id=bench_{run_idx}",
    ]

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    wall_time = time.perf_counter() - t0

    # Parse timing from output
    gen_time = 0.0
    predict_time = 0.0
    export_time = 0.0
    build_time = 0.0

    for line in result.stderr.split('\n'):
        if 'Final totals' in line:
            # Parse: gen=X.XXXs, build_sample=X.XXXs, predict=X.XXXs, export=X.XXXs
            for part in line.split(','):
                part = part.strip()
                if 'gen=' in part and 'build' not in part:
                    try:
                        gen_time = float(part.split('gen=')[1].split('s')[0])
                    except (ValueError, IndexError):
                        pass
                elif 'predict=' in part:
                    try:
                        predict_time = float(part.split('predict=')[1].split('s')[0])
                    except (ValueError, IndexError):
                        pass
                elif 'export=' in part:
                    try:
                        export_time = float(part.split('export=')[1].split('s')[0])
                    except (ValueError, IndexError):
                        pass
                elif 'build_sample=' in part:
                    try:
                        build_time = float(part.split('build_sample=')[1].split('s')[0])
                    except (ValueError, IndexError):
                        pass

    # Clean up
    try:
        shutil.rmtree(out_dir, ignore_errors=True)
    except Exception:
        pass

    return {
        'workers': workers,
        'numba_threads': numba_threads,
        'batch_size': batch_size,
        'wall_time': wall_time,
        'per_sample_wall': wall_time / num_samples,
        'gen_time': gen_time,
        'predict_time': predict_time,
        'build_time': build_time,
        'export_time': export_time,
        'per_sample_predict': predict_time / num_samples,
        'returncode': result.returncode,
        'stderr_tail': result.stderr[-500:] if result.returncode != 0 else '',
    }


def main():
    print(f"=== Benchmark Grid ===")
    print(f"Machine: {CPU_COUNT} CPUs")
    print(f"Samples per config: {NUM_SAMPLES}")
    print(f"Configs to test: {len(CONFIGS)}")
    print()

    # Warmup run (Numba JIT compilation)
    print("Warmup run (JIT compilation)...")
    warmup = run_one_config(4, 1, 8, 8, 0)
    print(f"  Warmup done in {warmup['wall_time']:.1f}s")
    print()

    results = []
    for i, (workers, nt, bs) in enumerate(CONFIGS):
        label = f"w={workers:2d}, nt={nt:2d}, bs={bs:3d}"
        product = workers * nt
        print(f"[{i+1}/{len(CONFIGS)}] {label} (w*nt={product:3d})...", end=" ", flush=True)

        try:
            r = run_one_config(workers, nt, bs, NUM_SAMPLES, i+1)
            results.append(r)

            if r['returncode'] != 0:
                print(f"FAILED (rc={r['returncode']})")
                if r['stderr_tail']:
                    print(f"  stderr: {r['stderr_tail'][:200]}")
            else:
                print(
                    f"wall={r['wall_time']:6.1f}s  "
                    f"per_sample={r['per_sample_wall']:.3f}s  "
                    f"gen={r['gen_time']:.2f}s  "
                    f"predict={r['predict_time']:.2f}s  "
                    f"export={r['export_time']:.2f}s"
                )
        except subprocess.TimeoutExpired:
            print("TIMEOUT (>300s)")
            results.append({
                'workers': workers, 'numba_threads': nt, 'batch_size': bs,
                'wall_time': 999, 'per_sample_wall': 999, 'gen_time': 0,
                'predict_time': 0, 'build_time': 0, 'export_time': 0,
                'per_sample_predict': 0, 'returncode': -1, 'stderr_tail': 'TIMEOUT',
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'workers': workers, 'numba_threads': nt, 'batch_size': bs,
                'wall_time': 999, 'per_sample_wall': 999, 'gen_time': 0,
                'predict_time': 0, 'build_time': 0, 'export_time': 0,
                'per_sample_predict': 0, 'returncode': -1, 'stderr_tail': str(e),
            })

    # Sort by per-sample wall time
    valid = [r for r in results if r['returncode'] == 0]
    valid.sort(key=lambda r: r['per_sample_wall'])

    print("\n" + "=" * 130)
    print(f"{'Rank':<5} {'Workers':<8} {'NbThreads':<10} {'BatchSz':<8} {'W*NT':<6} "
          f"{'Wall(s)':<9} {'Per-Sample':<11} {'Gen(s)':<9} {'Predict(s)':<11} {'Export(s)':<10} "
          f"{'Pred/Samp':<10}")
    print("-" * 130)

    for rank, r in enumerate(valid, 1):
        product = r['workers'] * r['numba_threads']
        print(
            f"{rank:<5} {r['workers']:<8} {r['numba_threads']:<10} {r['batch_size']:<8} {product:<6} "
            f"{r['wall_time']:<9.2f} {r['per_sample_wall']:<11.4f} {r['gen_time']:<9.2f} "
            f"{r['predict_time']:<11.2f} {r['export_time']:<10.2f} {r['per_sample_predict']:<10.4f}"
        )

    # Save raw results to JSON for analysis
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nRaw results saved to benchmark_results.json")

    # Quick analysis
    if valid:
        best = valid[0]
        print(f"\n=== Best Configuration (on {CPU_COUNT} CPUs) ===")
        print(f"  workers={best['workers']}, numba_threads={best['numba_threads']}, batch_size={best['batch_size']}")
        print(f"  Per-sample wall time: {best['per_sample_wall']:.4f}s")
        print(f"  Per-sample predict time: {best['per_sample_predict']:.4f}s")
        print(f"  Total wall time for {NUM_SAMPLES} samples: {best['wall_time']:.2f}s")


if __name__ == "__main__":
    main()
