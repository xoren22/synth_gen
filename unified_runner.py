#!/usr/bin/env python3
"""
unified_runner.py — orchestrator + worker + status + dashboard in one script.

Usage:
  # New run (2 workers, 10 total samples):
  python scripts/unified_runner.py start --workers 2 --num-samples 10

  # Dry run:
  python scripts/unified_runner.py start --workers 4 --num-samples 40 --dry-run

  # Reattach to existing run:
  python scripts/unified_runner.py start --run-dir data/2026_...

  # Status snapshot:
  python scripts/unified_runner.py status --run-dir data/2026_...

  # Dashboard (launched automatically by orchestrator, or manually):
  python -m streamlit run scripts/unified_runner.py -- dashboard
"""

import argparse, datetime, json, logging, os, signal, subprocess, sys, time, uuid
from pathlib import Path

log = logging.getLogger("runner")

META = "meta.json"
STATE = "runner_state.json"
CONTROL = "control.json"
STALE_S = 300
POLL_S = 5
ST_PORT = 8501
DEFAULT_WORKERS = 5
DEFAULT_NUM_SAMPLES = 25
DEFAULT_FREQ_MIN = 400
DEFAULT_FREQ_MAX = 10000
DEFAULT_NUMBA_THREADS = 1
DEFAULT_ANT_ISO_PROB = 0.25
DEFAULT_ANT_LATENT_DIM_MIN = 8
DEFAULT_ANT_LATENT_DIM_MAX = 20
DEFAULT_ANT_FOURIER_ORDER_MIN = 10
DEFAULT_ANT_FOURIER_ORDER_MAX = 24
DEFAULT_ANT_PETAL_ORDER_MIN = 3
DEFAULT_ANT_PETAL_ORDER_MAX = 12
DEFAULT_ANT_DB_MAX = 40.0
DEFAULT_ANT_SYMMETRY_MODE = "random"

# ── helpers ──────────────────────────────────────────────────

def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def _default_runs_base():
    return os.path.join(Path(__file__).resolve().parent, "data")

def _latest_run_dir(base_dir):
    if not os.path.isdir(base_dir):
        return None
    cand = []
    for e in os.scandir(base_dir):
        if not e.is_dir():
            continue
        st_path = os.path.join(e.path, STATE)
        if os.path.isfile(st_path):
            try:
                mt = os.path.getmtime(st_path)
            except OSError:
                mt = 0.0
            cand.append((mt, e.path))
    if not cand:
        return None
    cand.sort(key=lambda x: x[0], reverse=True)
    return cand[0][1]

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
    for bucket in os.scandir(out_dir):
        if not bucket.is_dir():
            continue
        for e in os.scandir(bucket.path):
            if e.is_file() and e.name.startswith("s") and e.name.endswith(".npz"):
                try:
                    if e.stat().st_size > 0:
                        n += 1
                except OSError:
                    pass
    return n

def _parse_iso_utc(ts):
    if not ts:
        return None
    try:
        return datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None

def _fmt_elapsed_hms(seconds):
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

def _resolve_targets(num_workers, num_samples=None, samples_per_worker=None):
    nw = int(num_workers or 0)
    if nw <= 0:
        raise ValueError("workers must be > 0")

    ns = None if num_samples is None else int(num_samples)
    spw = None if samples_per_worker is None else int(samples_per_worker)
    if ns is None and spw is None:
        raise ValueError("missing sample target")

    if ns is None:
        if spw < 0:
            raise ValueError("samples_per_worker must be >= 0")
        ns = nw * spw

    if spw is None:
        if ns < 0:
            raise ValueError("num_samples must be >= 0")
        if ns % nw != 0:
            raise ValueError("num_samples must be divisible by workers")
        spw = ns // nw

    if ns != nw * spw:
        raise ValueError("num_samples must equal workers * samples_per_worker")
    return ns, spw

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
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from generate import _generate_one, build_sample_from_generated, _export_one
    from antenna_pattern import RadiationPatternConfig
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
    update("running", done)  # heartbeat after JIT warmup
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
            sample = build_sample_from_generated(
                mask,
                normals,
                scene,
                refl,
                trans,
                dist,
                building_id=idx,
                pattern_cfg=pattern_cfg,
                pattern_seed=seed + 777_777,
            )
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

def _launch_worker(run_dir, wid, target, seed_base, fmin, fmax, nt, ant_cfg):
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
        "--ant-iso-prob", str(ant_cfg["ant_iso_prob"]),
        "--ant-latent-dim-min", str(ant_cfg["ant_latent_dim_min"]),
        "--ant-latent-dim-max", str(ant_cfg["ant_latent_dim_max"]),
        "--ant-fourier-order-min", str(ant_cfg["ant_fourier_order_min"]),
        "--ant-fourier-order-max", str(ant_cfg["ant_fourier_order_max"]),
        "--ant-petal-order-min", str(ant_cfg["ant_petal_order_min"]),
        "--ant-petal-order-max", str(ant_cfg["ant_petal_order_max"]),
        "--ant-db-max", str(ant_cfg["ant_db_max"]),
        "--ant-symmetry-mode", str(ant_cfg["ant_symmetry_mode"]),
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
    env = os.environ.copy()
    env["UNIFIED_RUN_DIR"] = os.path.realpath(os.path.abspath(run_dir))
    slog = os.path.join(run_dir, "streamlit.log")
    sfh = open(slog, "a", encoding="utf-8", buffering=1)
    try:
        proc = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__),
            "--server.port", str(port), "--server.headless", "true",
            "--", "dashboard",
        ], stdout=sfh, stderr=subprocess.STDOUT, env=env)
    except Exception:
        try:
            sfh.close()
        except Exception:
            pass
        raise
    proc._streamlit_log_handle = sfh
    return proc

def _close_streamlit_log_handle(proc):
    fh = getattr(proc, "_streamlit_log_handle", None)
    if fh:
        try:
            fh.flush()
        except Exception:
            pass
        try:
            fh.close()
        except Exception:
            pass

# ── orchestrator mode ────────────────────────────────────────

def cmd_start(args):
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    nw = args.workers
    total_samples = args.num_samples
    try:
        total_samples, spw = _resolve_targets(nw, num_samples=total_samples)
    except ValueError as e:
        raise SystemExit(f"Invalid start arguments: {e}")
    port = ST_PORT

    # Resolve run dir
    if args.run_dir:
        rd = os.path.abspath(args.run_dir)
    else:
        base = _default_runs_base()
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
        try:
            total_samples, spw = _resolve_targets(
                nw,
                num_samples=existing.get("num_samples"),
                samples_per_worker=existing.get("samples_per_worker"),
            )
        except ValueError:
            # Backward compatibility for old states that may only have total_target.
            total_samples = int(existing.get("total_target", total_samples))
            total_samples, spw = _resolve_targets(
                nw,
                num_samples=total_samples,
                samples_per_worker=existing.get("samples_per_worker"),
            )
        seed = existing.get("global_seed")
        if seed is None:
            import numpy as np
            seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        fmin = existing.get("freq_min", DEFAULT_FREQ_MIN)
        fmax = existing.get("freq_max", DEFAULT_FREQ_MAX)
        nt = existing.get("numba_threads", DEFAULT_NUMBA_THREADS)
        ant_cfg = {
            "ant_iso_prob": existing.get(
                "ant_iso_prob",
                0.5 * (
                    float(existing.get("ant_iso_prob_min", DEFAULT_ANT_ISO_PROB))
                    + float(existing.get("ant_iso_prob_max", DEFAULT_ANT_ISO_PROB))
                ),
            ),
            "ant_latent_dim_min": existing.get("ant_latent_dim_min", DEFAULT_ANT_LATENT_DIM_MIN),
            "ant_latent_dim_max": existing.get("ant_latent_dim_max", DEFAULT_ANT_LATENT_DIM_MAX),
            "ant_fourier_order_min": existing.get("ant_fourier_order_min", existing.get("ant_fourier_order", DEFAULT_ANT_FOURIER_ORDER_MIN)),
            "ant_fourier_order_max": existing.get("ant_fourier_order_max", existing.get("ant_fourier_order", DEFAULT_ANT_FOURIER_ORDER_MAX)),
            "ant_petal_order_min": existing.get("ant_petal_order_min", DEFAULT_ANT_PETAL_ORDER_MIN),
            "ant_petal_order_max": existing.get("ant_petal_order_max", DEFAULT_ANT_PETAL_ORDER_MAX),
            "ant_db_max": existing.get("ant_db_max", DEFAULT_ANT_DB_MAX),
            "ant_symmetry_mode": existing.get("ant_symmetry_mode", DEFAULT_ANT_SYMMETRY_MODE),
        }
        force_takeover = bool(prev_orch_pid and prev_orch_pid != os.getpid())
        existing["orchestrator_pid"] = os.getpid()
        existing["reattached_utc"] = _now()
        existing["previous_orchestrator_pid"] = prev_orch_pid
        existing["num_samples"] = total_samples
        existing["samples_per_worker"] = spw
        existing["total_target"] = total_samples
        for k, v in ant_cfg.items():
            existing[k] = v
        _write(sp, existing)
    else:
        import numpy as np
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        fmin, fmax, nt = DEFAULT_FREQ_MIN, DEFAULT_FREQ_MAX, DEFAULT_NUMBA_THREADS
        ant_cfg = {
            "ant_iso_prob": float(args.ant_iso_prob),
            "ant_latent_dim_min": int(args.ant_latent_dim_min),
            "ant_latent_dim_max": int(args.ant_latent_dim_max),
            "ant_fourier_order_min": int(args.ant_fourier_order_min),
            "ant_fourier_order_max": int(args.ant_fourier_order_max),
            "ant_petal_order_min": int(args.ant_petal_order_min),
            "ant_petal_order_max": int(args.ant_petal_order_max),
            "ant_db_max": float(args.ant_db_max),
            "ant_symmetry_mode": str(args.ant_symmetry_mode),
        }
        _write(sp, {
            "run_dir": rd, "num_workers": nw, "num_samples": total_samples,
            "samples_per_worker": spw, "total_target": total_samples, "global_seed": seed,
            "freq_min": fmin, "freq_max": fmax, "numba_threads": nt,
            "ant_iso_prob": ant_cfg["ant_iso_prob"],
            "ant_latent_dim_min": ant_cfg["ant_latent_dim_min"],
            "ant_latent_dim_max": ant_cfg["ant_latent_dim_max"],
            "ant_fourier_order_min": ant_cfg["ant_fourier_order_min"],
            "ant_fourier_order_max": ant_cfg["ant_fourier_order_max"],
            "ant_petal_order_min": ant_cfg["ant_petal_order_min"],
            "ant_petal_order_max": ant_cfg["ant_petal_order_max"],
            "ant_db_max": ant_cfg["ant_db_max"],
            "ant_symmetry_mode": ant_cfg["ant_symmetry_mode"],
            "created_utc": _now(), "orchestrator_pid": os.getpid(),
        })

    log.info(f"Run: {rd}  workers={nw}  num_samples={total_samples}  samples/worker={spw}  seed={seed}")
    log.info(
        "Antenna: d=[%s,%s] K=[%s,%s] p_iso=%s db_max=%s (effective per-sample max uniform in [0, db_max]) "
        "petals=[%s,%s] sym=%s",
        ant_cfg["ant_latent_dim_min"],
        ant_cfg["ant_latent_dim_max"],
        ant_cfg["ant_fourier_order_min"],
        ant_cfg["ant_fourier_order_max"],
        ant_cfg["ant_iso_prob"],
        ant_cfg["ant_db_max"],
        ant_cfg["ant_petal_order_min"],
        ant_cfg["ant_petal_order_max"],
        ant_cfg["ant_symmetry_mode"],
    )
    if not args.no_ui:
        log.info(f"Streamlit URL: http://localhost:{port}")

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
        _launch_worker(rd, wid, spw, sb, fmin, fmax, nt, ant_cfg)

    # Streamlit
    st_proc = None
    if not args.no_ui:
        try:
            st_proc = _launch_streamlit(rd, port)
            log.info(f"Streamlit on port {port} (PID {st_proc.pid})")
        except Exception as e:
            log.warning(f"Streamlit launch failed: {e}")

    # Monitor loop
    log_path = os.path.join(rd, "runner.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info("Monitor loop started. Ctrl+C to stop (workers terminate with orchestrator).")
    inline_progress = sys.stdout.isatty()
    progress_line_len = 0
    loop_started_at = time.monotonic()
    prev_progress_t = None
    prev_progress_done = None

    def _clear_inline_progress():
        nonlocal progress_line_len
        if inline_progress and progress_line_len > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
            progress_line_len = 0

    def _emit_progress(done, remaining, total_target):
        nonlocal progress_line_len, prev_progress_t, prev_progress_done
        now_t = time.monotonic()
        elapsed_s = max(0, int(now_t - loop_started_at))
        hh, rem = divmod(elapsed_s, 3600)
        mm, ss = divmod(rem, 60)
        elapsed = f"{hh:02d}:{mm:02d}:{ss:02d}"
        if prev_progress_t is None:
            combined_sps = 0.0
        else:
            dt = now_t - prev_progress_t
            dd = done - prev_progress_done
            combined_sps = (dd / dt) if dt > 0 else 0.0
            if combined_sps < 0:
                combined_sps = 0.0
        prev_progress_t = now_t
        prev_progress_done = done
        msg = (
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}] "
            f"Progress: {done}/{total_target} ({remaining} remaining) "
            f"elapsed={elapsed} sps={combined_sps:.2f}"
        )
        if inline_progress:
            pad = max(0, progress_line_len - len(msg))
            sys.stdout.write("\r" + msg + (" " * pad))
            sys.stdout.flush()
            progress_line_len = len(msg)
        else:
            log.info(msg)

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
                    _clear_inline_progress()
                    log.warning(f"{wid} dead (PID {m.get('pid')}). Restarting...")
                    _launch_worker(rd, wid, spw, sb, fmin, fmax, nt, ant_cfg)

            _emit_progress(tg, tr, total_samples)
            if all_done:
                _clear_inline_progress()
                log.info("All workers completed!"); break
            if st_proc and st_proc.poll() is not None:
                _clear_inline_progress()
                log.warning("Streamlit died, restarting...")
                _close_streamlit_log_handle(st_proc)
                try:
                    log.info(f"Streamlit URL: http://localhost:{port}")
                    st_proc = _launch_streamlit(rd, port)
                except Exception:
                    pass
    except KeyboardInterrupt:
        _clear_inline_progress()
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
        _clear_inline_progress()
        if st_proc:
            try:
                st_proc.terminate()
            except Exception:
                pass
            try:
                st_proc.wait(timeout=5)
            except Exception:
                try:
                    st_proc.kill()
                except Exception:
                    pass
            _close_streamlit_log_handle(st_proc)

# ── status mode ──────────────────────────────────────────────

def cmd_status(args):
    run_dir = args.run_dir or _latest_run_dir(_default_runs_base())
    if not run_dir:
        print(f"No run found in default base: {_default_runs_base()}"); sys.exit(1)
    rd = os.path.realpath(os.path.abspath(run_dir))
    st = _read(os.path.join(rd, STATE))
    if not st:
        print(f"No run at {rd}"); sys.exit(1)

    nw = st.get("num_workers", 0)
    try:
        total_samples, spw = _resolve_targets(
            nw,
            num_samples=st.get("num_samples"),
            samples_per_worker=st.get("samples_per_worker"),
        )
    except ValueError:
        total_samples = int(st.get("total_target", 0) or 0)
        spw = int(st.get("samples_per_worker", 0) or 0)
    print(f"Run: {rd}")
    print(f"Workers: {nw}  Samples/worker: {spw}  Total: {total_samples}")
    print(f"Seed: {st.get('global_seed')}  Freq: [{st.get('freq_min')}, {st.get('freq_max')}] MHz\n")
    print(
        "Antenna: "
        f"d_range=[{st.get('ant_latent_dim_min', DEFAULT_ANT_LATENT_DIM_MIN)},{st.get('ant_latent_dim_max', DEFAULT_ANT_LATENT_DIM_MAX)}] "
        f"K_range=[{st.get('ant_fourier_order_min', st.get('ant_fourier_order', DEFAULT_ANT_FOURIER_ORDER_MIN))},{st.get('ant_fourier_order_max', st.get('ant_fourier_order', DEFAULT_ANT_FOURIER_ORDER_MAX))}] "
        f"p_iso={st.get('ant_iso_prob', DEFAULT_ANT_ISO_PROB)} "
        f"db_max={st.get('ant_db_max', DEFAULT_ANT_DB_MAX)} "
        f"sym={st.get('ant_symmetry_mode', DEFAULT_ANT_SYMMETRY_MODE)}\n"
    )

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
    print(f"\nTotal: {tg}/{total_samples}")

# ── dashboard mode (streamlit) ───────────────────────────────

def cmd_dashboard(args):
    import streamlit as st

    run_dir = os.environ.get("UNIFIED_RUN_DIR")
    if not run_dir:
        run_dir = _latest_run_dir(_default_runs_base())
    if not run_dir:
        st.set_page_config(page_title="Runner Dashboard", layout="wide")
        st.title("Unified Runner Dashboard")
        st.error(f"No run found in default base: {_default_runs_base()}")
        return

    rd = os.path.realpath(os.path.abspath(run_dir))
    state = _read(os.path.join(rd, STATE))

    st.set_page_config(page_title="Runner Dashboard", layout="wide")
    st.title("Unified Runner Dashboard")
    st.caption(f"Run dir: `{rd}`")

    if not state:
        st.error(f"No run at {rd}"); return

    nw = state.get("num_workers", 0)
    try:
        total, spw = _resolve_targets(
            nw,
            num_samples=state.get("num_samples"),
            samples_per_worker=state.get("samples_per_worker"),
        )
    except ValueError:
        total = int(state.get("total_target", 0) or 0)
        spw = int(state.get("samples_per_worker", 0) or 0)

    cols = st.columns(5)
    cols[0].metric("Workers", nw)
    cols[1].metric("Samples/Worker", spw)

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
    cols[2].metric("Generated", f"{tg}/{total}")
    cols[3].metric("Progress", f"{pct:.0%}")
    started_dt = _parse_iso_utc(state.get("created_utc")) or _parse_iso_utc(state.get("reattached_utc"))
    if started_dt is not None:
        elapsed_s = (datetime.datetime.now(datetime.timezone.utc) - started_dt).total_seconds()
        cols[4].metric("Elapsed", _fmt_elapsed_hms(elapsed_s))
    else:
        cols[4].metric("Elapsed", "-")
    st.progress(min(pct, 1.0))
    st.markdown(f"**Progress: {tg}/{total} ({pct:.0%})**")

    # One line per worker: generated samples over dashboard refreshes.
    hist_key = f"worker_generated_history::{rd}"
    now_s = time.time()
    point = {"ts": now_s}
    for r in rows:
        point[r["w"]] = int(r["g"])
    hist = st.session_state.get(hist_key, [])
    if (not hist) or (now_s - float(hist[-1].get("ts", 0)) >= 0.5):
        hist.append(point)
    hist = hist[-300:]
    st.session_state[hist_key] = hist

    st.markdown("**Generated Samples Per Worker**")
    if len(hist) >= 2:
        try:
            import pandas as pd
            df = pd.DataFrame(hist)
            df["time"] = pd.to_datetime(df["ts"], unit="s").dt.strftime("%H:%M:%S")
            df = df.drop(columns=["ts"]).set_index("time")
            st.line_chart(df)
        except Exception:
            chart_rows = [{k: v for k, v in h.items() if k != "ts"} for h in hist]
            st.line_chart(chart_rows)
    else:
        st.line_chart([{r["w"]: int(r["g"]) for r in rows}])

    # Combined throughput (samples/sec) across all workers.
    sps_hist = []
    for i in range(1, len(hist)):
        prev = hist[i - 1]
        cur = hist[i]
        dt = float(cur.get("ts", 0.0)) - float(prev.get("ts", 0.0))
        if dt <= 0:
            continue
        prev_total = sum(float(v) for k, v in prev.items() if k != "ts")
        cur_total = sum(float(v) for k, v in cur.items() if k != "ts")
        sps = max(0.0, (cur_total - prev_total) / dt)
        sps_hist.append({"ts": float(cur.get("ts", 0.0)), "combined_sps": sps})

    st.markdown("**Combined Samples/sec (All Workers)**")
    if len(sps_hist) >= 2:
        try:
            import pandas as pd
            sdf = pd.DataFrame(sps_hist)
            sdf["time"] = pd.to_datetime(sdf["ts"], unit="s").dt.strftime("%H:%M:%S")
            sdf = sdf.drop(columns=["ts"]).set_index("time")
            st.line_chart(sdf)
        except Exception:
            st.line_chart([{"combined_sps": p["combined_sps"]} for p in sps_hist])
    else:
        st.line_chart([{"combined_sps": 0.0}])

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
    s.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    s.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--no-ui", action="store_true")
    s.add_argument("--ant-iso-prob", type=float, default=DEFAULT_ANT_ISO_PROB)
    s.add_argument("--ant-latent-dim-min", type=int, default=DEFAULT_ANT_LATENT_DIM_MIN)
    s.add_argument("--ant-latent-dim-max", type=int, default=DEFAULT_ANT_LATENT_DIM_MAX)
    s.add_argument("--ant-fourier-order-min", type=int, default=DEFAULT_ANT_FOURIER_ORDER_MIN)
    s.add_argument("--ant-fourier-order-max", type=int, default=DEFAULT_ANT_FOURIER_ORDER_MAX)
    s.add_argument("--ant-petal-order-min", type=int, default=DEFAULT_ANT_PETAL_ORDER_MIN)
    s.add_argument("--ant-petal-order-max", type=int, default=DEFAULT_ANT_PETAL_ORDER_MAX)
    s.add_argument("--ant-db-max", type=float, default=DEFAULT_ANT_DB_MAX)
    s.add_argument("--ant-symmetry-mode", type=str, default=DEFAULT_ANT_SYMMETRY_MODE, choices=["random", "none", "x", "y", "xy"])

    s = sub.add_parser("status", help="Print status snapshot")
    s.add_argument("--run-dir", default=None)

    s = sub.add_parser("worker", help="(internal) single worker")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--worker-id", required=True)
    s.add_argument("--target-samples", type=int, required=True)
    s.add_argument("--seed-base", type=int, required=True)
    s.add_argument("--freq-min", type=int, default=400)
    s.add_argument("--freq-max", type=int, default=10000)
    s.add_argument("--numba-threads", type=int, default=1)
    s.add_argument("--instance-id", required=True)
    s.add_argument("--ant-iso-prob", type=float, default=DEFAULT_ANT_ISO_PROB)
    s.add_argument("--ant-latent-dim-min", type=int, default=DEFAULT_ANT_LATENT_DIM_MIN)
    s.add_argument("--ant-latent-dim-max", type=int, default=DEFAULT_ANT_LATENT_DIM_MAX)
    s.add_argument("--ant-fourier-order-min", type=int, default=DEFAULT_ANT_FOURIER_ORDER_MIN)
    s.add_argument("--ant-fourier-order-max", type=int, default=DEFAULT_ANT_FOURIER_ORDER_MAX)
    s.add_argument("--ant-petal-order-min", type=int, default=DEFAULT_ANT_PETAL_ORDER_MIN)
    s.add_argument("--ant-petal-order-max", type=int, default=DEFAULT_ANT_PETAL_ORDER_MAX)
    s.add_argument("--ant-db-max", type=float, default=DEFAULT_ANT_DB_MAX)
    s.add_argument("--ant-symmetry-mode", type=str, default=DEFAULT_ANT_SYMMETRY_MODE, choices=["random", "none", "x", "y", "xy"])

    s = sub.add_parser("dashboard", help="(internal) streamlit UI")

    args = p.parse_args()
    {"start": cmd_start, "status": cmd_status, "worker": cmd_worker, "dashboard": cmd_dashboard}[args.mode](args)

if __name__ == "__main__":
    main()
