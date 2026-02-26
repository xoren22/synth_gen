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

import argparse, datetime, json, logging, os, signal, subprocess, sys, time, uuid
from pathlib import Path

log = logging.getLogger("runner")

META = "meta.json"
STATE = "runner_state.json"
CONTROL = "control.json"
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
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        pass

    # `kill(pid, 0)` is true for zombies. Treat zombies as dead on Linux.
    status_path = f"/proc/{pid}/status"
    if os.path.exists(status_path):
        try:
            with open(status_path) as f:
                for line in f:
                    if line.startswith("State:"):
                        return " Z" not in line
        except Exception:
            pass
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

def _parse_utc(ts):
    if not ts:
        return None
    try:
        return datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None

def _heartbeat_age_seconds(meta):
    dt = _parse_utc(meta.get("last_heartbeat_utc", ""))
    if not dt:
        return None
    return max(0.0, (datetime.datetime.now(datetime.timezone.utc) - dt).total_seconds())

def _format_age(age_s):
    if age_s is None:
        return "-"
    age = int(age_s)
    if age < 60:
        return f"{age}s ago"
    mm, ss = divmod(age, 60)
    if mm < 60:
        return f"{mm}m {ss}s ago"
    hh, mm = divmod(mm, 60)
    return f"{hh}h {mm}m ago"

def _dashboard_status(meta):
    s = str(meta.get("status", "unknown")).lower()
    remaining = int(meta.get("remaining_samples", 0) or 0)
    if s == "completed":
        return "completed" if remaining <= 0 else "incomplete"
    if s in {"incomplete", "retired", "starting"}:
        return s
    if s == "running":
        return "running" if _worker_alive(meta) else "dead"
    if s in {"dead", "failed"}:
        return "dead"
    # Catch stale/missing heartbeats for unknown statuses with a PID.
    if meta.get("pid", 0) and not _worker_alive(meta):
        return "dead"
    return s

def _tail_lines(path, n_lines=60):
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []
    return lines[-max(1, int(n_lines)):]

def _extract_last_error(log_lines):
    if not log_lines:
        return ""
    for i in range(len(log_lines) - 1, -1, -1):
        low = log_lines[i].lower()
        if ("failed:" in low) or ("error" in low) or ("traceback" in low):
            start = max(0, i - 3)
            end = min(len(log_lines), i + 8)
            return "".join(log_lines[start:end]).strip()
    return ""

def _control_path(run_dir, worker_id):
    return os.path.join(run_dir, "workers", worker_id, CONTROL)

def _read_control(run_dir, worker_id):
    c = _read(_control_path(run_dir, worker_id))
    if c is None:
        return {"worker_id": worker_id, "active_instance_id": "", "active_pid": 0, "dead_worker_pids": []}
    c.setdefault("worker_id", worker_id)
    c.setdefault("active_instance_id", "")
    c.setdefault("active_pid", 0)
    c.setdefault("dead_worker_pids", [])
    return c

def _write_control(run_dir, worker_id, data):
    data["updated_utc"] = _now()
    _write(_control_path(run_dir, worker_id), data)

def _worker_should_stop(run_dir, worker_id, instance_id, pid):
    c = _read_control(run_dir, worker_id)
    if c.get("active_instance_id") != instance_id:
        return True, "lease-revoked"
    dead = c.get("dead_worker_pids", [])
    if int(pid) in dead:
        return True, "pid-marked-dead"
    return False, ""

def _mark_pid_dead(run_dir, worker_id, pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return
    if pid <= 0:
        return
    c = _read_control(run_dir, worker_id)
    dead = c.get("dead_worker_pids", [])
    if pid not in dead:
        dead.append(pid)
    c["dead_worker_pids"] = dead[-256:]  # bound file growth
    _write_control(run_dir, worker_id, c)

def _pid_matches_worker(pid, run_dir, worker_id):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    cmdline_path = f"/proc/{pid}/cmdline"
    if not os.path.exists(cmdline_path):
        return False
    try:
        with open(cmdline_path, "rb") as f:
            parts = [p.decode("utf-8", errors="ignore") for p in f.read().split(b"\x00") if p]
    except Exception:
        return False
    if not parts:
        return False
    # Strictly require this process to be our worker invocation.
    return ("unified_runner.py" in " ".join(parts) and
            "worker" in parts and
            "--run-dir" in parts and run_dir in parts and
            "--worker-id" in parts and worker_id in parts)

def _terminate_pid(pid, run_dir=None, worker_id=None):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return
    if pid <= 0 or not _pid_alive(pid):
        return
    if run_dir is not None and worker_id is not None and not _pid_matches_worker(pid, run_dir, worker_id):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        return
    for _ in range(10):
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass

def _set_parent_death_signal():
    # Linux-only best effort: terminate worker when parent (orchestrator) dies.
    try:
        import ctypes
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
        # Race guard: parent may have died before prctl call.
        if os.getppid() == 1:
            os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        pass

def _reap_children():
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        except Exception:
            return
        if pid == 0:
            return

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
    my_pid = os.getpid()

    def update(status="running", completed=None):
        c = completed if completed is not None else _count_done(out)
        _write(mp, {
            "worker_id": args.worker_id, "target_samples": target,
            "generated_samples": c, "remaining_samples": max(0, target - c),
            "status": status, "pid": my_pid,
            "start_time_utc": start_t, "last_heartbeat_utc": _now(),
            "last_exit_code": None, "restart_count": restarts,
            "generation_index": gen_idx, "seed_base": args.seed_base,
            "attempt_log": f"attempt_{attempt:03d}.log",
            "color_key": f"gen_{gen_idx}", "worker_instance_id": args.instance_id,
        })

    def clear_lease_pid():
        c = _read_control(args.run_dir, args.worker_id)
        if c.get("active_instance_id") == args.instance_id and c.get("active_pid") == my_pid:
            c["active_pid"] = 0
            _write_control(args.run_dir, args.worker_id, c)

    start_t = _now()
    c = _read_control(args.run_dir, args.worker_id)
    c["active_pid"] = my_pid
    _write_control(args.run_dir, args.worker_id, c)

    should_stop, reason = _worker_should_stop(args.run_dir, args.worker_id, args.instance_id, my_pid)
    if should_stop:
        wlog.info(f"Stop requested before start ({reason}). Exiting.")
        update("retired", done)
        clear_lease_pid()
        return

    if remaining <= 0:
        wlog.info(f"Already done {done}/{target}. Exiting.")
        update("completed", done)
        clear_lease_pid()
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
        should_stop, reason = _worker_should_stop(args.run_dir, args.worker_id, args.instance_id, my_pid)
        if should_stop:
            wlog.info(f"Stop requested ({reason}). Exiting at {completed}/{target}.")
            update("retired", completed)
            clear_lease_pid()
            return

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

    if completed >= target:
        update("completed", completed)
        wlog.info(f"Done: {completed}/{target} samples.")
    else:
        update("incomplete", completed)
        wlog.warning(f"Incomplete: {completed}/{target} samples.")
    clear_lease_pid()

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
    old_pid = (old or {}).get("pid", 0)
    if old_pid:
        _mark_pid_dead(run_dir, wid, old_pid)
        _terminate_pid(old_pid, run_dir=run_dir, worker_id=wid)

    instance_id = uuid.uuid4().hex
    c = _read_control(run_dir, wid)
    c["active_instance_id"] = instance_id
    c["active_pid"] = 0
    _write_control(run_dir, wid, c)

    _write(os.path.join(wdir, META), {
        "worker_id": wid, "target_samples": target,
        "generated_samples": done, "remaining_samples": max(0, target - done),
        "status": "starting", "pid": 0,
        "start_time_utc": _now(), "last_heartbeat_utc": _now(),
        "last_exit_code": None, "restart_count": rc,
        "generation_index": gi, "seed_base": seed_base,
        "attempt_log": "", "color_key": f"gen_{gi}",
        "worker_instance_id": instance_id,
    })

    cmd = [
        sys.executable, os.path.abspath(__file__), "worker",
        "--run-dir", run_dir, "--worker-id", wid,
        "--target-samples", str(target), "--seed-base", str(seed_base),
        "--freq-min", str(fmin), "--freq-max", str(fmax),
        "--numba-threads", str(nt), "--instance-id", instance_id,
    ]
    proc = subprocess.Popen(cmd, preexec_fn=_set_parent_death_signal,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Update meta with real PID
    m = _read(os.path.join(wdir, META))
    if m:
        m["pid"] = proc.pid
        _write(os.path.join(wdir, META), m)
    c = _read_control(run_dir, wid)
    c["active_pid"] = proc.pid
    _write_control(run_dir, wid, c)
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
    force_takeover = False
    if existing:
        log.info(f"Reattaching to {rd}")
        prev_orch_pid = existing.get("orchestrator_pid", 0)
        nw = existing.get("num_workers", nw)
        spw = existing.get("samples_per_worker", spw)
        seed = existing.get("global_seed", args.seed)
        fmin = existing.get("freq_min", args.freq_min)
        fmax = existing.get("freq_max", args.freq_max)
        nt = existing.get("numba_threads", args.numba_threads)
        force_takeover = bool(prev_orch_pid and prev_orch_pid != os.getpid())
        existing["orchestrator_pid"] = os.getpid()
        existing["reattached_utc"] = _now()
        existing["previous_orchestrator_pid"] = prev_orch_pid
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
            continue

        done = (m.get("status") == "completed") and (m.get("remaining_samples", 1) <= 0)
        if done:
            continue

        if force_takeover:
            old_pid = m.get("pid", 0)
            if _pid_alive(old_pid):
                log.warning(f"  takeover: retiring stale {wid} PID {old_pid}")
                _mark_pid_dead(rd, wid, old_pid)
                _terminate_pid(old_pid, run_dir=rd, worker_id=wid)
            to_launch.append(i)
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

    log.info("Monitor loop started. Ctrl+C to stop (workers terminate with orchestrator).")
    try:
        while True:
            time.sleep(POLL_S)
            _reap_children()
            all_done, tg, tr = True, 0, 0
            for i in range(nw):
                wid = f"worker_{i:03d}"
                mp = os.path.join(rd, "workers", wid, META)
                m = _read(mp)
                if m is None:
                    all_done = False; continue
                tg += m.get("generated_samples", 0)
                tr += m.get("remaining_samples", 0)
                done = (m.get("status") == "completed") and (m.get("remaining_samples", 1) <= 0)
                if done:
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
        log.info("Orchestrator stopping. Terminating live workers...")
        for i in range(nw):
            wid = f"worker_{i:03d}"
            m = _read(os.path.join(rd, "workers", wid, META))
            if not m:
                continue
            pid = m.get("pid", 0)
            if _pid_alive(pid):
                _mark_pid_dead(rd, wid, pid)
                _terminate_pid(pid, run_dir=rd, worker_id=wid)
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
    try:
        import pandas as pd
    except Exception:
        pd = None

    rd = os.path.abspath(args.run_dir)

    st.set_page_config(page_title="Runner Dashboard", layout="wide")
    st.title("Unified Runner Dashboard")
    st.caption(f"Run directory: `{rd}`")

    def _mtime_ns(path):
        try:
            return os.stat(path).st_mtime_ns
        except OSError:
            return 0

    @st.cache_data(ttl=1, show_spinner=False)
    def _read_json_cached(path, mtime_ns):
        del mtime_ns
        return _read(path)

    @st.cache_data(ttl=1, show_spinner=False)
    def _read_tail_cached(path, mtime_ns, n_lines):
        del mtime_ns
        return "".join(_tail_lines(path, n_lines))

    def _load_rows(state):
        nw = state.get("num_workers", 0)
        spw = state.get("samples_per_worker", 0)
        rows, done_total, left_total = [], 0, 0
        for i in range(nw):
            wid = f"worker_{i:03d}"
            meta_path = os.path.join(rd, "workers", wid, META)
            control_path = os.path.join(rd, "workers", wid, CONTROL)
            m = _read_json_cached(meta_path, _mtime_ns(meta_path))
            c = _read_json_cached(control_path, _mtime_ns(control_path)) or {}
            if not m:
                rows.append({
                    "worker_id": wid, "status": "unknown", "pid": 0,
                    "done": 0, "left": spw, "target": spw, "restarts": 0,
                    "generation": 0, "heartbeat_utc": "-", "heartbeat_age_s": None,
                    "heartbeat_age": "-", "instance_id": "",
                    "attempt_log": "", "meta": {}, "control": c,
                })
                left_total += spw
                continue

            status = _dashboard_status(m)
            done = int(m.get("generated_samples", 0) or 0)
            left = int(m.get("remaining_samples", 0) or 0)
            age_s = _heartbeat_age_seconds(m)
            done_total += done
            left_total += left
            rows.append({
                "worker_id": wid,
                "status": status,
                "pid": int(m.get("pid", 0) or 0),
                "done": done,
                "left": left,
                "target": max(0, done + left),
                "restarts": int(m.get("restart_count", 0) or 0),
                "generation": int(m.get("generation_index", 0) or 0),
                "heartbeat_utc": str(m.get("last_heartbeat_utc", "-"))[:19],
                "heartbeat_age_s": age_s,
                "heartbeat_age": _format_age(age_s),
                "instance_id": str(m.get("worker_instance_id", "")),
                "attempt_log": str(m.get("attempt_log", "")),
                "meta": m,
                "control": c,
            })
        return rows, done_total, left_total

    refresh_s = st.sidebar.slider("Refresh interval (s)", min_value=1, max_value=30, value=3)
    trend_minutes = st.sidebar.slider("Trend window (min)", min_value=5, max_value=30, value=10)
    log_tail_lines = st.sidebar.slider("Log tail lines", min_value=20, max_value=200, value=60, step=10)

    search = st.sidebar.text_input("Search worker", value="", placeholder="worker_00")
    sort_field = st.sidebar.selectbox(
        "Sort by",
        options=["worker_id", "status", "done", "left", "restarts", "generation", "heartbeat_age_s"],
        index=0,
    )
    sort_desc = st.sidebar.checkbox("Sort descending", value=False)

    GEN_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999"]
    STATUS_STYLE = {
        "running": ("RUNNING", "#E8F1FB", "#0B3C5D"),
        "starting": ("STARTING", "#EEF3F8", "#334E68"),
        "completed": ("COMPLETED", "#EAF7EC", "#1B5E20"),
        "incomplete": ("INCOMPLETE", "#FFF4E5", "#8A4B08"),
        "dead": ("DEAD", "#FDECEA", "#8E1B10"),
        "retired": ("RETIRED", "#F5F5F5", "#424242"),
        "unknown": ("UNKNOWN", "#F5F7FA", "#37474F"),
    }

    def _render_dashboard_once():
        state_path = os.path.join(rd, STATE)
        state = _read_json_cached(state_path, _mtime_ns(state_path))
        if not state:
            st.error(f"No run at {rd}")
            return

        nw = state.get("num_workers", 0)
        spw = state.get("samples_per_worker", 0)
        total_target = nw * spw

        rows, total_done, total_left = _load_rows(state)
        all_statuses = sorted({r["status"] for r in rows}) or ["unknown"]
        status_filter = st.sidebar.multiselect("Status filter", options=all_statuses, default=all_statuses)

        filtered = []
        needle = search.strip().lower()
        for r in rows:
            if needle and needle not in r["worker_id"].lower():
                continue
            if r["status"] not in status_filter:
                continue
            filtered.append(r)

        def _sort_key(item):
            val = item.get(sort_field)
            if sort_field in {"worker_id", "status"}:
                return str(val or "").lower()
            if val is None:
                return -1
            return val

        filtered.sort(key=_sort_key, reverse=sort_desc)

        running_n = sum(1 for r in rows if r["status"] == "running")
        starting_n = sum(1 for r in rows if r["status"] == "starting")
        dead_n = sum(1 for r in rows if r["status"] == "dead")
        completed_n = sum(1 for r in rows if r["status"] == "completed")
        incomplete_n = sum(1 for r in rows if r["status"] == "incomplete")
        restart_total = sum(r["restarts"] for r in rows)
        pct = (total_done / total_target) if total_target else 0.0

        now_ts = time.time()
        timeline_key = f"timeline::{rd}"
        timeline = st.session_state.get(timeline_key, [])
        prev = timeline[-1] if timeline else None
        inst_sps = 0.0
        if prev:
            dt = max(1e-9, now_ts - prev["ts"])
            inst_sps = max(0.0, (total_done - prev["done"]) / dt)

        point = {
            "ts": now_ts,
            "done": total_done,
            "samples_per_sec": inst_sps,
            "dead_workers": dead_n,
            "restart_total": restart_total,
            "incomplete_workers": incomplete_n,
        }
        if (not prev) or (now_ts - prev["ts"] >= 1.0) or (point["done"] != prev["done"]) or (point["dead_workers"] != prev["dead_workers"]):
            timeline.append(point)
        cutoff = now_ts - (trend_minutes * 60)
        timeline = [p for p in timeline if p["ts"] >= cutoff]
        st.session_state[timeline_key] = timeline

        avg_sps = 0.0
        if len(timeline) >= 2:
            dt = max(1e-9, timeline[-1]["ts"] - timeline[0]["ts"])
            avg_sps = max(0.0, (timeline[-1]["done"] - timeline[0]["done"]) / dt)

        top = st.columns(6)
        top[0].metric("Workers", nw)
        top[1].metric("Total Target", total_target)
        top[2].metric("Done", total_done)
        top[3].metric("Remaining", total_left)
        top[4].metric("Samples/s", f"{avg_sps:.2f}")
        top[5].metric("Restarts", restart_total)

        st.progress(min(pct, 1.0))
        st.markdown(f"**Progress: {total_done}/{total_target} ({pct:.0%})**")

        stat_cols = st.columns(5)
        stat_cols[0].metric("Running", running_n)
        stat_cols[1].metric("Starting", starting_n)
        stat_cols[2].metric("Dead", dead_n)
        stat_cols[3].metric("Completed", completed_n)
        stat_cols[4].metric("Incomplete", incomplete_n)

        st.subheader("Worker Table")
        table_rows = []
        for r in filtered:
            table_rows.append({
                "Worker": r["worker_id"],
                "Status": r["status"],
                "PID": r["pid"],
                "Done": r["done"],
                "Left": r["left"],
                "Restarts": r["restarts"],
                "Gen": r["generation"],
                "Heartbeat": r["heartbeat_utc"],
                "Age": r["heartbeat_age"],
                "Instance": r["instance_id"][:12],
            })

        if pd is not None:
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        else:
            st.table(table_rows)

        st.subheader("Trends")
        if pd is not None and len(timeline) >= 2:
            tdf = pd.DataFrame(timeline)
            tdf["time"] = pd.to_datetime(tdf["ts"], unit="s", utc=True).dt.tz_convert(None)
            tdf = tdf.set_index("time")
            tc1, tc2 = st.columns(2)
            tc1.caption("Completion and throughput")
            tc1.line_chart(tdf[["done", "samples_per_sec"]], use_container_width=True)
            tc2.caption("Health events")
            tc2.line_chart(tdf[["dead_workers", "restart_total", "incomplete_workers"]], use_container_width=True)
        else:
            st.caption("Trend chart needs at least two refresh points.")

        st.subheader("Worker Cards")
        for r in filtered:
            label, bg, fg = STATUS_STYLE.get(r["status"], STATUS_STYLE["unknown"])
            gen_color = GEN_COLORS[r["generation"] % len(GEN_COLORS)]
            st.markdown(
                f'<div style="border-left:6px solid {gen_color};border:1px solid #d0d7de;'
                f'border-radius:6px;padding:10px;margin:6px 0;background:{bg};color:{fg};">'
                f'<b>{r["worker_id"]}</b> &nbsp; [{label}] &nbsp; PID:{r["pid"]} &nbsp; '
                f'{r["done"]}/{r["target"]} samples &nbsp; restarts:{r["restarts"]} &nbsp; '
                f'gen:{r["generation"]} &nbsp; hb:{r["heartbeat_age"]}</div>',
                unsafe_allow_html=True,
            )

        st.subheader("Worker Diagnostics")
        if not filtered:
            st.info("No workers match current filters.")
            return

        worker_choices = [r["worker_id"] for r in filtered]
        selected_worker = st.selectbox("Inspect worker", worker_choices, index=0)
        selected = next(r for r in filtered if r["worker_id"] == selected_worker)

        dcols = st.columns(4)
        dcols[0].metric("Status", selected["status"])
        dcols[1].metric("PID", selected["pid"])
        dcols[2].metric("Heartbeat Age", selected["heartbeat_age"])
        dcols[3].metric("Restarts", selected["restarts"])
        st.caption(f"Instance: `{selected['instance_id']}` | Generation: `{selected['generation']}`")

        control = selected.get("control", {}) or {}
        dead_pids = [int(p) for p in control.get("dead_worker_pids", []) if str(p).isdigit()]
        active_pid = control.get("active_pid", 0)
        pid_history = dead_pids[:]
        if str(active_pid).isdigit() and int(active_pid) > 0:
            pid_history.append(int(active_pid))
        st.markdown("**PID History (old -> new)**")
        st.code(", ".join(str(p) for p in pid_history[-20:]) if pid_history else "(none)")

        worker_dir = os.path.join(rd, "workers", selected_worker)
        attempt_log = selected.get("attempt_log", "")
        if not attempt_log:
            try:
                logs = sorted(n for n in os.listdir(worker_dir) if n.startswith("attempt_") and n.endswith(".log"))
            except FileNotFoundError:
                logs = []
            if logs:
                attempt_log = logs[-1]

        if attempt_log:
            log_path = os.path.join(worker_dir, attempt_log)
            log_text = _read_tail_cached(log_path, _mtime_ns(log_path), log_tail_lines)
            lines = log_text.splitlines(True)
            last_error = _extract_last_error(lines)
            st.markdown(f"**Log Tail: `{attempt_log}`**")
            st.code(log_text if log_text else "(log empty)")
            if last_error:
                st.markdown("**Last Error Snippet**")
                st.code(last_error)
            else:
                st.caption("No recent errors found in this log tail.")
        else:
            st.info("No attempt log available yet for this worker.")

    if hasattr(st, "fragment"):
        @st.fragment(run_every=f"{refresh_s}s")
        def _refreshing_panel():
            _render_dashboard_once()
        _refreshing_panel()
    else:
        st.warning("This Streamlit build has no fragment auto-refresh; use manual refresh.")
        if st.button("Refresh now"):
            st.rerun()
        _render_dashboard_once()

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
    s.add_argument("--instance-id", required=True)

    s = sub.add_parser("dashboard", help="(internal) streamlit UI")
    s.add_argument("--run-dir", required=True)

    args = p.parse_args()
    {"start": cmd_start, "status": cmd_status, "worker": cmd_worker, "dashboard": cmd_dashboard}[args.mode](args)

if __name__ == "__main__":
    main()
