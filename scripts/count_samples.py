#!/usr/bin/env python3
"""Count total generated samples across one or more run directories.

Reads worker meta.json files rather than listing .npz files.
Estimates time to 100M by measuring throughput over a short interval.

Usage:
  python scripts/count_samples.py /mnt/weka/xoren/synth_data
  python scripts/count_samples.py /mnt/weka/xoren/synth_data/2026_03_13_06_01_33
"""

import json, os, sys, time

TARGET = 100_000_000


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def fmt(n):
    """Format number with underscore separators."""
    s = str(n)
    groups = []
    while s:
        groups.append(s[-3:])
        s = s[:-3]
    return "_".join(reversed(groups))


def count_run(run_dir):
    workers_dir = os.path.join(run_dir, "workers")
    if not os.path.isdir(workers_dir):
        return None
    total = 0
    workers = 0
    for entry in sorted(os.scandir(workers_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        meta = read_json(os.path.join(entry.path, "meta.json"))
        if meta is None:
            continue
        total += meta.get("generated_samples", 0)
        workers += 1
    return {"run_dir": run_dir, "workers": workers, "generated": total}


def discover_runs(path):
    runs = []
    if os.path.isdir(os.path.join(path, "workers")):
        runs.append(path)
    elif os.path.isdir(path):
        for entry in sorted(os.scandir(path), key=lambda e: e.name):
            if entry.is_dir() and os.path.isdir(os.path.join(entry.path, "workers")):
                runs.append(entry.path)
    return runs


def count_all(all_runs):
    grand = 0
    results = []
    for run_path in all_runs:
        info = count_run(run_path)
        if info is None:
            continue
        results.append(info)
        grand += info["generated"]
    return results, grand


def main():
    DEFAULT_DIR = "/mnt/weka/xoren/synth_data"
    args = sys.argv[1:] if len(sys.argv) > 1 else [DEFAULT_DIR]

    all_runs = []
    for arg in args:
        found = discover_runs(arg)
        if not found:
            print(f"No runs found in {arg}", file=sys.stderr)
        all_runs.extend(found)

    if not all_runs:
        print("No runs found.")
        sys.exit(1)

    # First count (use midpoint time to account for counting duration)
    ta = time.monotonic()
    _, total1 = count_all(all_runs)
    t1 = (ta + time.monotonic()) / 2

    print("Measuring throughput (20s)...", end="", flush=True)
    time.sleep(20)
    print(" done.\n")

    # Second count
    tb = time.monotonic()
    results2, total2 = count_all(all_runs)
    t2 = (tb + time.monotonic()) / 2
    dt = t2 - t1

    for info in results2:
        name = os.path.basename(info["run_dir"])
        print(f"{name:30s}  {fmt(info['generated']):>15s}  [{info['workers']} workers]")

    if len(results2) > 1:
        print(f"{'TOTAL':30s}  {fmt(total2):>15s}")

    delta = total2 - total1
    sps = delta / dt if dt > 0 else 0

    spd = sps * 86400
    print(f"\nThroughput: {sps:.1f} samples/sec  |  {fmt(int(spd))} samples/day")

    remaining = TARGET - total2
    if remaining <= 0:
        print(f"Target {fmt(TARGET)} reached!")
    elif sps > 0:
        eta_s = remaining / sps
        days, rem = divmod(int(eta_s), 86400)
        hours, rem = divmod(rem, 3600)
        mins, _ = divmod(rem, 60)
        parts = []
        if days: parts.append(f"{days}d")
        if hours: parts.append(f"{hours}h")
        if mins: parts.append(f"{mins}m")
        print(f"Remaining: {fmt(remaining)} samples")
        print(f"ETA to {fmt(TARGET)}: {' '.join(parts) or '<1m'}")
    else:
        print(f"Remaining: {fmt(remaining)} samples")
        print("ETA: unknown (no throughput detected)")


if __name__ == "__main__":
    main()
