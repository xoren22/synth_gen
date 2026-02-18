import os
os.environ.setdefault("NUMBA_THREADING_LAYER", "tbb")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from tqdm import tqdm
import torch, numpy as np
from numba import njit, prange
from models import RadarSample
from normal_parser import precompute_wall_angles_pca
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as _mp

_WARMED_UP = False


def _normals_from_sample(sample, ref=None, trans=None):
    """Extract (nx_img, ny_img) float64 arrays from a RadarSample.

    Uses sample.normals if available, otherwise computes via PCA from the
    reflectance/transmittance wall mask.
    """
    if sample.normals is not None:
        n = sample.normals
        nx = np.ascontiguousarray(n[..., 0], dtype=np.float64)
        ny = np.ascontiguousarray(n[..., 1], dtype=np.float64)
        return nx, ny
    if ref is None or trans is None:
        ref, trans, _ = sample.input_img.cpu().numpy()
    building_mask = (ref + trans > 0).astype(np.uint8)
    angles = precompute_wall_angles_pca(building_mask)
    rad = np.deg2rad(angles + 90.0)
    nx = np.cos(rad).astype(np.float64)
    ny = np.sin(rad).astype(np.float64)
    invalid = angles < 0
    if np.any(invalid):
        nx = nx.copy(); ny = ny.copy()
        nx[invalid] = 0.0; ny[invalid] = 0.0
    return nx, ny


# ---------------------------------------------------------------------#
#  GLOBALS                                                             #
# ---------------------------------------------------------------------#
MAX_REFL  = 5            # reflection budget for normal runs
MAX_TRANS = 10           # transmission (wall) budget
N_ANGLES  = 360*128      # single place to control angular resolution for combined method

# Backfill configuration
BACKFILL_METHOD = "los"     # options: "los", "diffuse", or "fspl"
BACKFILL_PARAMS = {
    "iters": 60,           # for diffuse
    "lambda": 0.05,        # for diffuse
    "alpha": 1.0,          # for diffuse
}
# ---------------------------------------------------------------------#
#  NUMERIC BASICS                                                      #
# ---------------------------------------------------------------------#
def calculate_fspl(dist_m, freq_MHz, min_dist_m=0.125):
    dist_clamped = np.maximum(dist_m, min_dist_m)
    return 20.0*np.log10(dist_clamped) + 20.0*np.log10(freq_MHz) - 27.55

@njit(inline='always')
def _fspl(dist_m: float, freq_MHz: float, min_dist_m: float = 0.125) -> float:
    d = dist_m if dist_m > min_dist_m else min_dist_m
    return 20.0*np.log10(d) + 20.0*np.log10(freq_MHz) - 27.55

# Fast FSPL lookup table to avoid log10 in inner loops
@njit(cache=False)
def _build_fspl_lut(max_steps: int, pixel_size: float, freq_MHz: float, min_dist_m: float = 0.125):
    lut = np.empty(max_steps + 1, np.float64)
    for k in range(max_steps + 1):
        d = k * pixel_size
        if d < min_dist_m:
            d = min_dist_m
        lut[k] = 20.0*np.log10(d) + 20.0*np.log10(freq_MHz) - 27.55
    return lut

@njit(inline='always')
def _fspl_from_lut(lut: np.ndarray, step_index: int) -> float:
    if step_index < 0:
        step_index = 0
    n = lut.shape[0]
    if step_index >= n:
        step_index = n - 1
    return lut[step_index]

@njit(inline='always')
def _euclidean_distance(px, py, x_ant, y_ant, pixel_size=0.25):
    return np.hypot(px - x_ant, py - y_ant) * pixel_size

# ---------------------------------------------------------------------#
#  BACKFILL (CPU, numpy)                                               #
# ---------------------------------------------------------------------#

def _compute_fspl_field(h: int, w: int, x_ant: float, y_ant: float, pixel_size: float, freq_MHz: float, max_loss: float):
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - x_ant, yy - y_ant) * pixel_size
    d = np.maximum(d, 0.125)
    F = 20.0*np.log10(d) + 20.0*np.log10(freq_MHz) - 27.55
    return np.minimum(F, max_loss).astype(np.float32)


def _backfill_fspl(out: np.ndarray, cnt: np.ndarray, x_ant: float, y_ant: float, pixel_size: float, freq_MHz: float, max_loss: float) -> np.ndarray:
    mask0 = (cnt == 0)
    if not np.any(mask0):
        return out
    F = _compute_fspl_field(out.shape[0], out.shape[1], x_ant, y_ant, pixel_size, freq_MHz, max_loss)
    out2 = out.copy()
    out2[mask0] = F[mask0]
    return np.minimum(out2, max_loss)


def _backfill_diffuse_residual(out: np.ndarray, cnt: np.ndarray, x_ant: float, y_ant: float, pixel_size: float, freq_MHz: float, max_loss: float, *, iters: int = 60, lam: float = 0.05, alpha: float = 1.0) -> np.ndarray:
    h, w = out.shape
    mask0 = (cnt == 0)
    if not np.any(mask0):
        return out

    F = _compute_fspl_field(h, w, x_ant, y_ant, pixel_size, freq_MHz, max_loss)
    R = (out.astype(np.float32) - F).astype(np.float32)

    cnt_f = cnt.astype(np.float32)
    cmax = float(cnt_f.max()) if cnt_f.size > 0 else 1.0
    norm = cnt_f / (cmax + 1e-6)
    wL = 1.0 + alpha * np.pad(norm[:, 1:], ((0,0),(0,1)), mode='constant')
    wR = 1.0 + alpha * np.pad(norm[:, :-1], ((0,0),(1,0)), mode='constant')
    wU = 1.0 + alpha * np.pad(norm[1:, :], ((0,1),(0,0)), mode='constant')
    wD = 1.0 + alpha * np.pad(norm[:-1, :], ((1,0),(0,0)), mode='constant')

    for _ in range(iters):
        R_left  = np.pad(R[:, :-1], ((0,0),(1,0)), mode='edge')
        R_right = np.pad(R[:, 1:],  ((0,0),(0,1)), mode='edge')
        R_up    = np.pad(R[:-1, :], ((1,0),(0,0)), mode='edge')
        R_down  = np.pad(R[1:, :],  ((0,1),(0,0)), mode='edge')

        num = wL * R_left + wR * R_right + wU * R_up + wD * R_down
        den = (wL + wR + wU + wD) + lam
        R_new = R.copy()
        R_new[mask0] = (num[mask0] / den[mask0]).astype(np.float32)
        R = R_new

    out2 = F + R
    out2 = np.maximum(out2, F)
    out2 = np.minimum(out2, max_loss)
    out2[~mask0] = out[~mask0]
    return out2.astype(np.float32)


@njit(cache=False)
def _backfill_direct_los(out: np.ndarray, cnt: np.ndarray, trans_mat: np.ndarray, x_ant: float, y_ant: float, pixel_size: float, freq_MHz: float, max_loss: float) -> np.ndarray:
    h, w = out.shape
    out2 = out.copy()
    mask0 = (cnt == 0)

    for py in range(h):
        for px in range(w):
            if not mask0[py, px]:
                continue

            x0 = x_ant
            y0 = y_ant
            x1 = float(px)
            y1 = float(py)
            ddx = x1 - x0
            ddy = y1 - y0
            adx = ddx if ddx >= 0.0 else -ddx
            ady = ddy if ddy >= 0.0 else -ddy
            steps = int(adx if adx >= ady else ady)

            if steps <= 0:
                dxp = float(px) - x_ant
                dyp = float(py) - y_ant
                dist = np.hypot(dxp, dyp) * pixel_size
                if dist < 0.125:
                    dist = 0.125
                fspl = 20.0 * np.log10(dist) + 20.0 * np.log10(freq_MHz) - 27.55
                tot = fspl
                if tot > max_loss:
                    tot = max_loss
                out2[py, px] = tot
                cnt[py, px] = 1.0
                continue

            sx = ddx / steps
            sy = ddy / steps
            x = x0
            y = y0

            ix0 = int(np.rint(x))
            iy0 = int(np.rint(y))
            if 0 <= ix0 < w and 0 <= iy0 < h:
                last_val = float(trans_mat[iy0, ix0])
            else:
                last_val = 0.0
            sum_loss = 0.0

            for s in range(steps + 1):
                ix = int(np.rint(x))
                iy = int(np.rint(y))
                if 0 <= ix < w and 0 <= iy < h:
                    val = float(trans_mat[iy, ix])
                    if s == 0:
                        last_val = val
                    if val != last_val and last_val > 0.0 and val == 0.0:
                        sum_loss += last_val
                        if sum_loss >= max_loss:
                            sum_loss = max_loss
                            break
                    last_val = val
                x += sx
                y += sy

            dxp = float(px) - x_ant
            dyp = float(py) - y_ant
            dist = np.hypot(dxp, dyp) * pixel_size
            if dist < 0.125:
                dist = 0.125
            fspl = 20.0 * np.log10(dist) + 20.0 * np.log10(freq_MHz) - 27.55
            tot = sum_loss + fspl
            if tot > max_loss:
                tot = max_loss
            out2[py, px] = tot
            cnt[py, px] = 1.0

    return out2.astype(np.float32)


def apply_backfill(out: np.ndarray, cnt: np.ndarray, x_ant: float, y_ant: float, pixel_size: float, freq_MHz: float, max_loss: float, method: str = BACKFILL_METHOD, params: dict | None = None, *, trans_mat: np.ndarray | None = None) -> np.ndarray:
    if params is None:
        params = BACKFILL_PARAMS
    if method == "fspl":
        return _backfill_fspl(out, cnt, x_ant, y_ant, pixel_size, freq_MHz, max_loss)
    elif method == "diffuse":
        iters = int(params.get("iters", 60))
        lam   = float(params.get("lambda", 0.05))
        alpha = float(params.get("alpha", 1.0))
        return _backfill_diffuse_residual(out, cnt, x_ant, y_ant, pixel_size, freq_MHz, max_loss, iters=iters, lam=lam, alpha=alpha)
    elif method == "los":
        if trans_mat is None:
            raise ValueError("LOS backfill requires trans_mat")
        return _backfill_direct_los(out, cnt, trans_mat, x_ant, y_ant, pixel_size, freq_MHz, max_loss)
    else:
        return _backfill_fspl(out, cnt, x_ant, y_ant, pixel_size, freq_MHz, max_loss)

# ---------------------------------------------------------------------#
#  STEP-UNTIL-WALL                                                     #
# ---------------------------------------------------------------------#
@njit(inline='always')
def _step_until_wall(mat, x0, y0, dx, dy, radial_step, max_dist):
    h, w = mat.shape
    x, y = x0, y0
    px_prev = int(round(x0));  py_prev = int(round(y0))
    last_val = mat[py_prev, px_prev]
    travelled = 0.0

    while travelled <= max_dist:
        x += dx * radial_step
        y += dy * radial_step
        travelled += radial_step
        px, py = int(round(x)), int(round(y))

        if px < 0 or px >= w or py < 0 or py >= h:
            return -1, -1, -1, -1, travelled, last_val, last_val

        cur_val = mat[py, px]
        if cur_val != last_val:
            return px, py, px_prev, py_prev, travelled, last_val, cur_val

        px_prev, py_prev = px, py
        last_val = cur_val

    return -1, -1, -1, -1, travelled, last_val, last_val

# ---------------------------------------------------------------------#
#  NORMALS USAGE                                                       #
# ---------------------------------------------------------------------#
@njit(inline='always')
def _estimate_normal(nx_img, ny_img, px, py):
    return nx_img[py, px], ny_img[py, px]

@njit(inline='always')
def _reflect_dir(dx, dy, nx, ny):
    dot = dx*nx + dy*ny
    rx = dx - 2.0*dot*nx;  ry = dy - 2.0*dot*ny
    mag = np.hypot(rx, ry)
    return (-dx, -dy) if mag==0 else (rx/mag, ry/mag)

@njit(cache=False)
def _trace_ray_recursive(
    refl_mat, trans_mat, nx_img, ny_img,
    out_img, counts,
    x0, y0, dx, dy,
    trans_ct, refl_ct,
    acc_loss,
    global_r,
    pixel_size, freq_MHz,
    radial_step, max_dist,
    max_trans, max_refl, max_loss,
    fspl_lut, use_lut
):
    if acc_loss >= max_loss:
        return

    px_hit, py_hit, px_prev, py_prev, travelled, last_val, cur_val = _step_until_wall(
        trans_mat, x0, y0, dx, dy, radial_step, max_dist
    )

    steps = int(travelled / radial_step) + 1
    for s in range(1, steps):
        xi = x0 + dx * radial_step * s
        yi = y0 + dy * radial_step * s
        ix = int(round(xi));  iy = int(round(yi))
        if ix < 0 or ix >= out_img.shape[1] or iy < 0 or iy >= out_img.shape[0]:
            break

        if use_lut:
            k = int(global_r + s)
            fspl = _fspl_from_lut(fspl_lut, k)
        else:
            fspl = _fspl((global_r + radial_step * s) * pixel_size, freq_MHz)
        tot  = acc_loss + fspl
        if tot > max_loss:
            tot = max_loss

        if tot < out_img[iy, ix]:
            out_img[iy, ix] = tot
        counts[iy, ix] += 1.0

    if steps <= 1:
        ix = px_prev; iy = py_prev
        if 0 <= ix < out_img.shape[1] and 0 <= iy < out_img.shape[0]:
            if use_lut:
                k = int(global_r + travelled)
                fspl = _fspl_from_lut(fspl_lut, k)
            else:
                fspl = _fspl((global_r + travelled) * pixel_size, freq_MHz)
            tot = acc_loss + fspl
            if tot > max_loss:
                tot = max_loss
            if tot < out_img[iy, ix]:
                out_img[iy, ix] = tot
            counts[iy, ix] += 1.0

    if px_hit < 0:
        return

    new_x = x0 + dx * travelled
    new_y = y0 + dy * travelled
    new_r = global_r + travelled

    is_transmit_exit = (last_val > 0. and cur_val == 0.)

    acc_loss_trans = acc_loss + (last_val if is_transmit_exit else 0.0)
    trans_ct_trans = trans_ct + (1 if is_transmit_exit else 0)
    if not (acc_loss_trans >= max_loss or trans_ct_trans > max_trans):
        _trace_ray_recursive(
            refl_mat, trans_mat, nx_img, ny_img,
            out_img, counts,
            new_x, new_y, dx, dy,
            trans_ct_trans, refl_ct,
            acc_loss_trans, new_r,
            pixel_size, freq_MHz,
            radial_step, max_dist,
            max_trans, max_refl, max_loss,
            fspl_lut, use_lut
        )

    if refl_ct < max_refl:
        refl_val = refl_mat[py_hit, px_hit]
        if refl_val > 0.0:
            nx, ny = _estimate_normal(nx_img, ny_img, px_hit, py_hit)
            if nx != 0.0 or ny != 0.0:
                rdx, rdy = _reflect_dir(dx, dy, nx, ny)
                if (last_val == 0.0 and cur_val > 0.0):
                    rx0 = float(px_prev)
                    ry0 = float(py_prev)
                else:
                    rx0 = new_x
                    ry0 = new_y
                _trace_ray_recursive(
                    refl_mat, trans_mat, nx_img, ny_img,
                    out_img, counts,
                    rx0, ry0, rdx, rdy,
                    trans_ct, refl_ct + 1,
                    acc_loss + refl_val,
                    new_r,
                    pixel_size, freq_MHz,
                    radial_step, max_dist,
                    max_trans, max_refl, max_loss,
                    fspl_lut, use_lut
                )


@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False, cache=False)
def calculate_combined_loss_with_normals(
    reflectance_mat, transmittance_mat, nx_img, ny_img,
    x_ant, y_ant, freq_MHz,
    n_angles,
    max_refl=MAX_REFL, max_trans=MAX_TRANS,
    pixel_size=0.25,
    radial_step=1.0,
    max_loss=32000.0,
    use_fspl_lut=True
):
    h, w = reflectance_mat.shape
    out  = np.full((h, w), max_loss, np.float32)
    cnt  = np.zeros((h, w), np.float32)

    dtheta   = 2.0 * np.pi / n_angles
    max_dist = 100.0 * np.hypot(w, h)
    cos_v    = np.cos(np.arange(n_angles) * dtheta)
    sin_v    = np.sin(np.arange(n_angles) * dtheta)

    use_lut = use_fspl_lut and (radial_step == 1.0)
    if use_lut:
        max_steps = int(max_dist) + 2
        fspl_lut = _build_fspl_lut(max_steps, pixel_size, freq_MHz)
    else:
        fspl_lut = np.zeros(1, np.float64)

    for i in prange(n_angles):
        _trace_ray_recursive(
            reflectance_mat, transmittance_mat,
            nx_img, ny_img,
            out, cnt,
            x_ant, y_ant,
            cos_v[i], sin_v[i],
            0, 0,
            0.0, 0.0,
            pixel_size, freq_MHz,
            radial_step, max_dist,
            max_trans, max_refl, max_loss,
            fspl_lut, use_lut
        )

    return out, cnt

def _warmup_numba_once():
    global _WARMED_UP
    if _WARMED_UP:
        return
    h, w = 8, 8
    zero = np.zeros((h, w), np.float64)
    trans = zero.copy(); trans[:, 4] = 1.0
    nx = zero.copy(); ny = zero.copy(); ny[:, 4] = 1.0
    calculate_combined_loss_with_normals(
        zero, trans, nx, ny,
        x_ant=4.0, y_ant=4.0, freq_MHz=1000.0,
        n_angles=N_ANGLES,
        max_refl=0, max_trans=1,
        radial_step=1.0,
        use_fspl_lut=True
    )
    _WARMED_UP = True

# ---------------------------------------------------------------------#
#  APPROX WRAPPER                                                      #
# ---------------------------------------------------------------------#
class Approx:
    def __init__(self, method='combined'):
        self.method = method
        _warmup_numba_once()

    def approximate(self, sample: RadarSample,
                    max_trans=MAX_TRANS, max_refl=MAX_REFL):
        ref, trans, _ = sample.input_img.cpu().numpy()
        x, y, f = sample.x_ant, sample.y_ant, sample.freq_MHz
        nx_img, ny_img = _normals_from_sample(sample, ref, trans)
        ref_c  = np.ascontiguousarray(ref, dtype=np.float64)
        trans_c = np.ascontiguousarray(trans, dtype=np.float64)

        if self.method == 'combined':
            feat, cnt = calculate_combined_loss_with_normals(
                ref_c, trans_c, nx_img, ny_img,
                x, y, f,
                n_angles=N_ANGLES,
                max_refl=max_refl,
                max_trans=max_trans,
                radial_step=1.0,
                use_fspl_lut=True
            )
            feat = apply_backfill(feat, cnt, x, y, 0.25, f, 32000.0, BACKFILL_METHOD, BACKFILL_PARAMS, trans_mat=trans_c)
        else:
            feat, cnt = calculate_transmission_loss_numpy(trans_c, x, y, f, n_angles=360*128, max_walls=max_trans)
            feat = feat.astype(np.float32)
            feat = apply_backfill(feat, cnt.astype(np.float32), x, y, 0.25, f, 32000.0, BACKFILL_METHOD, BACKFILL_PARAMS, trans_mat=trans_c)
        feat = np.minimum(feat, 32000.0)
        return torch.from_numpy(np.floor(feat))

    def predict(self, samples, max_trans=MAX_TRANS, max_refl=MAX_REFL, num_workers: int = 0, numba_threads: int = 0, backend: str = "threads"):
        if num_workers is None or num_workers <= 1:
            return [self.approximate(s, max_trans, max_refl) for s in tqdm(samples, "predicting")]
        max_workers = num_workers if isinstance(num_workers, int) and num_workers > 0 else max(1, (_mp.cpu_count() or 2) - 1)
        if backend == "threads":
            try:
                import numba as _nb
                if numba_threads and numba_threads > 0:
                    _nb.set_num_threads(numba_threads)
                else:
                    per = max(1, (_mp.cpu_count() or 2) // max_workers)
                    _nb.set_num_threads(per)
            except Exception:
                pass
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(self.approximate, s, max_trans, max_refl) for s in samples]
                return [f.result() for f in tqdm(futures, total=len(futures), desc="predicting")]
        else:
            raise ValueError(f"Only 'threads' backend supported in standalone mode, got '{backend}'")


@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False)
def calculate_transmission_loss_numpy(trans_mat, x_ant, y_ant, freq_MHz, n_angles=360*128, radial_step=1.0, max_walls=MAX_TRANS, max_loss=32000.0, pixel_size=0.25):

    h, w  = trans_mat.shape
    out   = np.full((h,w), max_loss, np.float64)
    cnt   = np.zeros((h,w), np.float64)

    dtheta = 2.0*np.pi / n_angles
    max_dist = np.hypot(w, h)
    cos_v = np.cos(np.arange(n_angles)*dtheta)
    sin_v = np.sin(np.arange(n_angles)*dtheta)

    for i in range(n_angles):
        ct, st = cos_v[i], sin_v[i]
        sum_loss = 0.0; last_val = None; wall_ct = 0; r=0.0

        while r<=max_dist:
            x = x_ant + r*ct; y = y_ant + r*st
            px = int(round(x)); py = int(round(y))

            if px<0 or px>=w or py<0 or py>=h:
                if last_val is not None and last_val>0:
                    sum_loss = min(sum_loss+last_val, max_loss)
                break

            val = trans_mat[py,px]
            if last_val is None:
                last_val = val
            if val!=last_val and last_val>0 and val==0:
                sum_loss += last_val
                wall_ct += 1
                if sum_loss>=max_loss or wall_ct>=max_walls:
                    sum_loss = min(sum_loss, max_loss)
                    fspl = _fspl(r * pixel_size, freq_MHz)
                    tot = sum_loss + fspl
                    tot = max_loss if tot > max_loss else tot
                    if tot < out[py,px]:
                        out[py,px] = tot
                    cnt[py,px] += 1.0
                    break
            last_val = val

            fspl = _fspl(r * pixel_size, freq_MHz)
            tot = sum_loss + fspl
            tot = max_loss if tot > max_loss else tot

            if tot < out[py,px]:
                out[py,px] = tot
            cnt[py,px] += 1.0
            r += radial_step

    return out, cnt
