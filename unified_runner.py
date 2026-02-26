#!/usr/bin/env python3
"""
unified_runner.py — orchestrator + worker + status + dashboard in one script.

Usage:
  # New run (2 workers, 5 samples each):
  python scripts/unified_runner.py start --workers 2 --samples-per-worker 5

  # Dry run:
  python scripts/unified_runner.py start --workers 4 --samples-per-worker 10 --dry-run

  # Reattach to existing run:
  python scripts/unified_runner.py start --run-dir analysis/unified_runs/2026_...

  # Status snapshot:
  python scripts/unified_runner.py status --run-dir analysis/unified_runs/2026_...

  # Dashboard (launched automatically by orchestrator, or manually):
  python -m streamlit run scripts/unified_runner.py -- dashboard --run-dir <path>
"""

import argparse, datetime, json, logging, os, subprocess, sys, time
from pathlib import Path

log = logging.getLogger("runner")

META = "meta.json"
STATE = "runner_state.json"
STALE_S = 60
POLL_S = 1
ST_PORT = 8501

# ── helpers ──────────────────────────────────────────────────

def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def _read(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def _pid_alive(pid):
    try:
        os.kill(pid, 0); return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True

def _worker_alive(meta):
    if not _pid_alive(meta.get("pid", 0)):
        return False
    hb = meta.get("last_heartbeat_utc", "")
    try:
        dt = datetime.datetime.fromisoformat(hb.replace("Z", "+00:00"))
        return (datetime.datetime.now(datetime.timezone.utc) - dt).total_seconds() < STALE_S
    except Exception:
        return False

def _count_done(out_dir):
    if not os.path.isdir(out_dir):
        return 0
    n = 0
    for e in os.scandir(out_dir):
        if e.is_dir() and e.name.startswith("s"):
            if os.path.exists(os.path.join(e.path, e.name + ".npz")) and \
               os.path.exists(os.path.join(e.path, e.name + ".json")):
                n += 1
    return n

# ── worker mode ──────────────────────────────────────────────

def cmd_worker(args):
    os.environ["NUMBA_THREADING_LAYER"] = "tbb"
    os.environ["NUMBA_NUM_THREADS"] = str(args.numba_threads)

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from generate import _generate_one, build_sample_from_generated, _export_one
    from approx import Approx

    wdir = os.path.join(args.run_dir, "workers", args.worker_id)
    out = os.path.join(wdir, "out")
    mp = os.path.join(wdir, META)
    os.makedirs(out, exist_ok=True)

    # Read meta written by orchestrator
    meta = _read(mp) or {}
    gen_idx = meta.get("generation_index", 0)
    restarts = meta.get("restart_count", 0)
    attempt = restarts + 1
    logf = os.path.join(wdir, f"attempt_{attempt:03d}.log")

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(message)s",
        handlers=[logging.FileHandler(logf), logging.StreamHandler()],
    )
    wlog = logging.getLogger(args.worker_id)

    done = _count_done(out)
    target = args.target_samples
    remaining = max(0, target - done)

    def update(status="running", completed=None):
        c = completed if completed is not None else _count_done(out)
        _write(mp, {
            "worker_id": args.worker_id, "target_samples": target,
            "generated_samples": c, "remaining_samples": max(0, target - c),
            "status": status, "pid": os.getpid(),
            "start_time_utc": start_t, "last_heartbeat_utc": _now(),
            "last_exit_code": None, "restart_count": restarts,
            "generation_index": gen_idx, "seed_base": args.seed_base,
            "attempt_log": f"attempt_{attempt:03d}.log",
            "color_key": f"gen_{gen_idx}",
        })

    start_t = _now()
    if remaining <= 0:
        wlog.info(f"Already done {done}/{target}. Exiting.")
        update("completed", done)
        return

    wlog.info(f"target={target} done={done} remaining={remaining} seed_base={args.seed_base}")
    update("running", done)

    try:
        import numba as _nb
        _nb.set_num_threads(max(1, args.numba_threads))
    except Exception:
        pass

    model = Approx()
    completed = done
    for i in range(remaining):
        idx = done + i
        seed = args.seed_base + idx
        try:
            mask, normals, scene, refl, trans, dist = _generate_one(seed, args.freq_min, args.freq_max)
            sample = build_sample_from_generated(mask, normals, scene, refl, trans, dist, building_id=idx)
            pred = model.approximate(sample)
            _export_one(sample, mask, normals, refl, trans, scene, idx, pred, out)
            completed += 1
            wlog.info(f"Sample {completed}/{target} (seed={seed})")
        except Exception as e:
            wlog.error(f"Sample idx={idx} failed: {e}", exc_info=True)
        update("running", completed)

    update("completed", completed)
    wlog.info(f"Done: {completed}/{target} samples.")

# ── orchestrator helpers ─────────────────────────────────────

def _launch_worker(run_dir, wid, target, seed_base, fmin, fmax, nt):
    wdir = os.path.join(run_dir, "workers", wid)
    os.makedirs(wdir, exist_ok=True)

    # Write initial meta so orchestrator doesn't immediately restart
    old = _read(os.path.join(wdir, META))
    rc = (old.get("restart_count", 0) + 1) if old else 0
    gi = (old.get("generation_index", 0) + 1) if old else 0
    if old is None:  # first ever launch
        rc, gi = 0, 0
    done = _count_done(os.path.join(wdir, "out"))

    _write(os.path.join(wdir, META), {
        "worker_id": wid, "target_samples": target,
        "generated_samples": done, "remaining_samples": max(0, target - done),
        "status": "starting", "pid": 0,
        "start_time_utc": _now(), "last_heartbeat_utc": _now(),
        "last_exit_code": None, "restart_count": rc,
        "generation_index": gi, "seed_base": seed_base,
        "attempt_log": "", "color_key": f"gen_{gi}",
    })

    cmd = [
        sys.executable, os.path.abspath(__file__), "worker",
        "--run-dir", run_dir, "--worker-id", wid,
        "--target-samples", str(target), "--seed-base", str(seed_base),
        "--freq-min", str(fmin), "--freq-max", str(fmax),
        "--numba-threads", str(nt),
    ]
    proc = subprocess.Popen(cmd, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Update meta with real PID
    m = _read(os.path.join(wdir, META))
    if m:
        m["pid"] = proc.pid
        _write(os.path.join(wdir, META), m)
    return proc

def _launch_streamlit(run_dir, port):
    return subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__),
        "--server.port", str(port), "--server.headless", "true",
        "--", "dashboard", "--run-dir", run_dir,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ── orchestrator mode ────────────────────────────────────────

def cmd_start(args):
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    nw = args.workers
    spw = args.samples_per_worker
    if nw > 8 and not args.force:
        log.error(f"{nw} workers > soft limit 8. Use --force."); sys.exit(1)

    # Resolve run dir
    if args.run_dir:
        rd = os.path.abspath(args.run_dir)
    else:
        base = os.path.join(Path(__file__).resolve().parent.parent, "analysis", "unified_runs")
        os.makedirs(base, exist_ok=True)
        rd = os.path.join(base, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        i = 1
        while os.path.exists(rd):
            rd = os.path.join(base, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + f"_{i:03d}")
            i += 1

    os.makedirs(os.path.join(rd, "workers"), exist_ok=True)
    sp = os.path.join(rd, STATE)

    # Reattach or new
    existing = _read(sp)
    if existing:
        log.info(f"Reattaching to {rd}")
        nw = existing.get("num_workers", nw)
        spw = existing.get("samples_per_worker", spw)
        seed = existing.get("global_seed", args.seed)
        fmin = existing.get("freq_min", args.freq_min)
        fmax = existing.get("freq_max", args.freq_max)
        nt = existing.get("numba_threads", args.numba_threads)
        existing["orchestrator_pid"] = os.getpid()
        existing["reattached_utc"] = _now()
        _write(sp, existing)
    else:
        import numpy as np
        seed = args.seed if args.seed is not None else int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        fmin, fmax, nt = args.freq_min, args.freq_max, args.numba_threads
        _write(sp, {
            "run_dir": rd, "num_workers": nw, "samples_per_worker": spw,
            "total_target": nw * spw, "global_seed": seed,
            "freq_min": fmin, "freq_max": fmax, "numba_threads": nt,
            "created_utc": _now(), "orchestrator_pid": os.getpid(),
        })

    log.info(f"Run: {rd}  workers={nw}  samples/worker={spw}  seed={seed}")

    if args.dry_run:
        log.info("=== DRY RUN ===")
        for i in range(nw):
            log.info(f"  worker_{i:03d}: seed_base={seed + i * spw}, target={spw}")
        return

    # Reconcile workers
    to_launch = []
    for i in range(nw):
        wid = f"worker_{i:03d}"
        mp = os.path.join(rd, "workers", wid, META)
        m = _read(mp)
        if m is None:
            to_launch.append(i)
        elif m.get("status") == "completed":
            continue
        elif _worker_alive(m):
            log.info(f"  {wid} alive (PID {m['pid']})")
        else:
            to_launch.append(i)

    log.info(f"Launching {len(to_launch)} workers, {nw - len(to_launch)} already alive/completed")
    for i in to_launch:
        wid = f"worker_{i:03d}"
        sb = seed + i * spw
        log.info(f"  Starting {wid} (seed_base={sb})")
        _launch_worker(rd, wid, spw, sb, fmin, fmax, nt)

    # Streamlit
    st_proc = None
    if not args.no_ui:
        try:
            st_proc = _launch_streamlit(rd, args.port)
            log.info(f"Streamlit on port {args.port} (PID {st_proc.pid})")
        except Exception as e:
            log.warning(f"Streamlit launch failed: {e}")

    # Monitor loop
    log_path = os.path.join(rd, "runner.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info("Monitor loop started. Ctrl+C to stop (workers keep running).")
    try:
        while True:
            time.sleep(POLL_S)
            all_done, tg, tr = True, 0, 0
            for i in range(nw):
                wid = f"worker_{i:03d}"
                mp = os.path.join(rd, "workers", wid, META)
                m = _read(mp)
                if m is None:
                    all_done = False; continue
                tg += m.get("generated_samples", 0)
                tr += m.get("remaining_samples", 0)
                if m.get("status") == "completed":
                    continue
                all_done = False
                if not _worker_alive(m):
                    sb = seed + i * spw
                    log.warning(f"{wid} dead (PID {m.get('pid')}). Restarting...")
                    _launch_worker(rd, wid, spw, sb, fmin, fmax, nt)

            log.info(f"Progress: {tg}/{nw * spw} ({tr} remaining)")
            if all_done:
                log.info("All workers completed!"); break
            if st_proc and st_proc.poll() is not None:
                log.warning("Streamlit died, restarting...")
                try:
                    st_proc = _launch_streamlit(rd, args.port)
                except Exception:
                    pass
    except KeyboardInterrupt:
        log.info("Orchestrator stopped. Workers continue independently.")
    finally:
        if st_proc:
            try: st_proc.terminate()
            except Exception: pass

# ── status mode ──────────────────────────────────────────────

def cmd_status(args):
    rd = os.path.abspath(args.run_dir)
    st = _read(os.path.join(rd, STATE))
    if not st:
        print(f"No run at {rd}"); sys.exit(1)

    nw = st.get("num_workers", 0)
    spw = st.get("samples_per_worker", 0)
    print(f"Run: {rd}")
    print(f"Workers: {nw}  Samples/worker: {spw}  Total: {nw * spw}")
    print(f"Seed: {st.get('global_seed')}  Freq: [{st.get('freq_min')}, {st.get('freq_max')}] MHz\n")

    rows, tg = [], 0
    for i in range(nw):
        wid = f"worker_{i:03d}"
        m = _read(os.path.join(rd, "workers", wid, META))
        if not m:
            rows.append((wid, "no-meta", "-", "0", str(spw), "0", "-")); continue
        s = m.get("status", "?")
        if s == "running" and not _worker_alive(m):
            s = "DEAD"
        g = m.get("generated_samples", 0)
        tg += g
        rows.append((wid, s, str(m.get("pid", "?")), str(g),
                      str(m.get("remaining_samples", 0)), str(m.get("restart_count", 0)),
                      m.get("last_heartbeat_utc", "?")[:19]))

    hdr = ("Worker", "Status", "PID", "Done", "Left", "Restarts", "Heartbeat")
    ws = [max(len(h), max((len(r[j]) for r in rows), default=0)) for j, h in enumerate(hdr)]
    fmt = "  ".join(f"{{:<{w}}}" for w in ws)
    print(fmt.format(*hdr))
    print(fmt.format(*("-" * w for w in ws)))
    for r in rows:
        print(fmt.format(*r))
    print(f"\nTotal: {tg}/{nw * spw}")

# ── dashboard mode (streamlit) ───────────────────────────────

def cmd_dashboard(args):
    import streamlit as st

    rd = os.path.abspath(args.run_dir)
    state = _read(os.path.join(rd, STATE))

    st.set_page_config(page_title="Runner Dashboard", layout="wide")
    st.title("Unified Runner Dashboard")

    if not state:
        st.error(f"No run at {rd}"); return

    nw = state.get("num_workers", 0)
    spw = state.get("samples_per_worker", 0)
    total = nw * spw

    cols = st.columns(4)
    cols[0].metric("Workers", nw)
    cols[1].metric("Samples/Worker", spw)
    cols[2].metric("Total Target", total)
    cols[3].metric("Seed", state.get("global_seed"))

    COLORS = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6", "#e74c3c", "#1abc9c", "#f39c12", "#34495e"]
    rows, tg = [], 0
    for i in range(nw):
        wid = f"worker_{i:03d}"
        m = _read(os.path.join(rd, "workers", wid, META))
        if not m:
            rows.append({"w": wid, "s": "unknown", "pid": "-", "g": 0, "r": spw, "rc": 0, "hb": "-", "gi": 0})
            continue
        s = m.get("status", "?")
        alive = _worker_alive(m) if s == "running" else False
        if s == "running" and not alive:
            s = "DEAD"
        g = m.get("generated_samples", 0)
        tg += g
        rows.append({"w": wid, "s": s, "pid": m.get("pid", "-"), "g": g,
                      "r": m.get("remaining_samples", 0), "rc": m.get("restart_count", 0),
                      "hb": m.get("last_heartbeat_utc", "-")[:19], "gi": m.get("generation_index", 0)})

    pct = tg / total if total else 0
    st.progress(min(pct, 1.0))
    st.markdown(f"**Progress: {tg}/{total} ({pct:.0%})**")

    a = sum(1 for r in rows if r["s"] == "running")
    d = sum(1 for r in rows if r["s"] == "DEAD")
    c = sum(1 for r in rows if r["s"] == "completed")
    c1, c2, c3 = st.columns(3)
    c1.metric("Running", a); c2.metric("Dead", d); c3.metric("Completed", c)

    for r in rows:
        color = COLORS[r["gi"] % len(COLORS)]
        icon = {"running": "🟢", "completed": "✅", "DEAD": "🔴"}.get(r["s"], "⚪")
        bg = "#1a1a2e" if r["s"] != "DEAD" else "#2d1b1b"
        t = r["g"] + r["r"]
        st.markdown(
            f'<div style="border-left:4px solid {color};padding:8px;margin:4px 0;background:{bg}">'
            f'<b>{icon} {r["w"]}</b> &nbsp; {r["s"]} &nbsp; PID:{r["pid"]} &nbsp; '
            f'{r["g"]}/{t} samples &nbsp; restarts:{r["rc"]} &nbsp; gen:{r["gi"]} &nbsp; '
            f'hb:{r["hb"]}</div>', unsafe_allow_html=True)

    time.sleep(3)
    st.rerun()

# ── CLI ──────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Unified runner for parallel generation")
    sub = p.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("start", help="Launch/reattach orchestrator")
    s.add_argument("--run-dir", default=None)
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--samples-per-worker", type=int, default=5)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--freq-min", type=int, default=400)
    s.add_argument("--freq-max", type=int, default=10000)
    s.add_argument("--numba-threads", type=int, default=1)
    s.add_argument("--force", action="store_true", help="Allow >8 workers")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--no-ui", action="store_true")
    s.add_argument("--port", type=int, default=ST_PORT)

    s = sub.add_parser("status", help="Print status snapshot")
    s.add_argument("--run-dir", required=True)

    s = sub.add_parser("worker", help="(internal) single worker")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--worker-id", required=True)
    s.add_argument("--target-samples", type=int, required=True)
    s.add_argument("--seed-base", type=int, required=True)
    s.add_argument("--freq-min", type=int, default=400)
    s.add_argument("--freq-max", type=int, default=10000)
    s.add_argument("--numba-threads", type=int, default=1)

    s = sub.add_parser("dashboard", help="(internal) streamlit UI")
    s.add_argument("--run-dir", required=True)

    args = p.parse_args()
    {"start": cmd_start, "status": cmd_status, "worker": cmd_worker, "dashboard": cmd_dashboard}[args.mode](args)

if __name__ == "__main__":
    main()
