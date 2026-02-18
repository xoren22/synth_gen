from numba import njit, prange
import numpy as np
import math


@njit(cache=True, inline='always')
def _pca_angle(xs, ys):
    """
    Return principal-axis angle (deg in [0,180)) of the 2-D point cloud.
    Closed-form 2x2 eigen decomposition:
        theta = 1/2 * atan2(2*cov_xy, cov_xx - cov_yy)
    """
    n = xs.size
    x_mean = xs.mean()
    y_mean = ys.mean()

    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for i in range(n):
        dx = xs[i] - x_mean
        dy = ys[i] - y_mean
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    sxx /= n
    syy /= n
    sxy /= n

    angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)   # radians
    angle_deg = (math.degrees(angle)) % 180.0
    return angle_deg


@njit(cache=True, inline='always')
def _pca_angle_trimmed(xs, ys):
    """
    Robust PCA angle using trimmed fitting.
    1. Initial PCA fit
    2. Remove worst 20% of points (furthest from line)
    3. Refit PCA on remaining 80%
    """
    n = xs.size
    if n < 3:
        return _pca_angle(xs, ys)  # Too few points for trimming

    # Step 1: Initial PCA fit
    initial_angle = _pca_angle(xs, ys)

    # Step 2: Compute distances to initial line
    angle_rad = math.radians(initial_angle)

    if abs(initial_angle - 90.0) < 1.0:
        line_x = xs.mean()
        distances = np.abs(xs - line_x)
    else:
        m = math.tan(angle_rad)
        b = ys.mean() - m * xs.mean()
        norm_factor = math.sqrt(m * m + 1.0)
        distances = np.abs(m * xs - ys + b) / norm_factor

    # Step 3: Keep best 80% (smallest distances)
    keep_count = max(3, int(0.8 * n))
    if keep_count >= n:
        return initial_angle

    indices = np.argsort(distances)[:keep_count]

    xs_trimmed = xs[indices]
    ys_trimmed = ys[indices]

    return _pca_angle(xs_trimmed, ys_trimmed)


# distance-weighted PCA centered at source pixel
@njit(cache=True, inline='always')
def _pca_angle_weighted_centered(xs, ys, px, py, distance_power=1.0):
    n = xs.size
    if n < 3:
        return _pca_angle(xs, ys)

    dx = xs - px
    dy = ys - py

    distances = np.sqrt(dx * dx + dy * dy)
    weights = np.empty(n, dtype=np.float32)
    for i in range(n):
        d = distances[i]
        if d < 1e-6:
            weights[i] = 10.0
        else:
            weights[i] = 1.0 / (d ** distance_power)

    wsum = weights.sum()
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for i in range(n):
        w = weights[i] / wsum
        sxx += w * dx[i] * dx[i]
        syy += w * dy[i] * dy[i]
        sxy += w * dx[i] * dy[i]

    ang = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return math.degrees(ang) % 180.0


@njit(cache=True, inline='always')
def _angle_vote_oriented_strip(xs, ys, px, py, win, band=1.25, along_margin=2):
    """
    Votes for the line orientation that passes through (px,py) using a thin
    oriented strip.
    Returns angle in [0,180) or -1 if insufficient support.
    """
    n = xs.size
    if n < 3:
        return -1.0

    best_score = -1.0
    best_angle = -1.0

    max_len = float(max(3, win - along_margin))

    for k in range(60):
        ang = 3.0 * k
        rad = ang * math.pi / 180.0
        c = math.cos(rad)
        s = math.sin(rad)

        score = 0.0
        count = 0
        for i in range(n):
            dx = xs[i] - px
            dy = ys[i] - py
            perp = abs(-s * dx + c * dy)
            if perp <= band:
                t = c * dx + s * dy
                if abs(t) <= max_len:
                    score += 1.0 / (1.0 + 0.5 * perp + 0.25 * abs(t))
                    count += 1
        if count >= 3 and score > best_score:
            best_score = score
            best_angle = ang

    return best_angle if best_score >= 0.0 else -1.0


@njit(cache=True)
def compute_wall_angle_pca(img, px, py, win=5):
    h, w = img.shape
    y0, y1 = max(py - win, 0), min(py + win + 1, h)
    x0, x1 = max(px - win, 0), min(px + win + 1, w)

    patch = img[y0:y1, x0:x1]
    ys, xs = np.nonzero(patch)
    n = xs.size
    if n < 3:
        return -1.0

    xs = xs.astype(np.float32) + x0
    ys = ys.astype(np.float32) + y0

    angle1 = _angle_vote_oriented_strip(xs, ys, px, py, win, band=1.25, along_margin=2)

    if angle1 < 0.0:
        angle1 = _pca_angle_weighted_centered(xs, ys, px, py, 1.0)

    rad1 = math.radians(angle1)
    c1 = math.cos(rad1)
    s1 = math.sin(rad1)

    dt = 1.0
    xs1 = np.empty(n, dtype=np.float32)
    ys1 = np.empty(n, dtype=np.float32)
    xs2 = np.empty(n, dtype=np.float32)
    ys2 = np.empty(n, dtype=np.float32)
    c1_idx = 0
    c2_idx = 0

    for i in range(n):
        dx = xs[i] - px
        dy = ys[i] - py
        dist = abs(-s1 * dx + c1 * dy)
        if dist <= dt:
            xs1[c1_idx] = xs[i]
            ys1[c1_idx] = ys[i]
            c1_idx += 1
        else:
            xs2[c2_idx] = xs[i]
            ys2[c2_idx] = ys[i]
            c2_idx += 1

    if c2_idx < 3:
        if c1_idx >= 3:
            xs1_f = xs1[:c1_idx]
            ys1_f = ys1[:c1_idx]
            refined = _pca_angle_trimmed(xs1_f, ys1_f)
            return refined
        return angle1

    xs2 = xs2[:c2_idx]
    ys2 = ys2[:c2_idx]
    angle2 = _pca_angle(xs2, ys2)

    diff = abs(((angle1 - angle2 + 90.0) % 180.0) - 90.0)

    merge_th = 20.0
    if diff < merge_th:
        return angle1

    if c1_idx < 3:
        return angle1

    xs1 = xs1[:c1_idx]
    ys1 = ys1[:c1_idx]
    final_angle = _pca_angle_weighted_centered(xs1, ys1, px, py, 1.0)
    return final_angle


@njit(cache=True)
def compute_wall_angle_multiscale_pca(img, px, py):
    """
    Multi-scale PCA-based wall angle estimator.
    Tries multiple window sizes and picks the most consistent result.
    Returns: wall angle in degrees (0-180)
    """
    h, w = img.shape

    window_sizes = np.array([9, 11, 13], dtype=np.int64)
    angles = np.zeros(len(window_sizes), dtype=np.float32)
    valid_count = 0

    for i, win_size in enumerate(window_sizes):
        angle = compute_wall_angle_pca(img, px, py, win_size)
        if angle >= 0:
            angles[valid_count] = angle
            valid_count += 1

    if valid_count == 0:
        for radius in range(1, min(h, w)):
            found = False
            best_dx, best_dy = 0, 0
            min_dist_sq = float('inf')

            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    ny_check, nx_check = py + dy, px + dx
                    if (0 <= ny_check < h and 0 <= nx_check < w and
                        img[ny_check, nx_check] == 0):
                        dist_sq = dx*dx + dy*dy
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            best_dx, best_dy = dx, dy
                            found = True

            if found:
                normal_angle = math.degrees(math.atan2(best_dy, best_dx))
                wall_angle = (normal_angle + 90) % 180
                return wall_angle

        return 90.0

    else:
        return angles[0]


def precompute_wall_angles_pca(building_mask: np.ndarray) -> np.ndarray:
    """
    Computes wall angles for ALL non-zero pixels using multi-scale PCA method.
    Returns wall angles (0-180) which are side-agnostic for reflection calculations.
    """
    h, w = building_mask.shape
    angles_img = np.zeros((h, w), dtype=np.float32)

    for py in range(h):
        for px in range(w):
            if building_mask[py, px] > 0:
                angle = compute_wall_angle_multiscale_pca(building_mask, px, py)
                angles_img[py, px] = angle

    return angles_img
