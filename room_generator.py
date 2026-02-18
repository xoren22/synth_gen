# Continuous per-region room-size distribution (no large/small modes)
# Mix of large & small rooms within the same floor via a single heavy‑tailed law
# and a lognormal scale factor per region.
#
# Keeps: minimum room width logic, doors, normals, metadata, "New floor" button.

import json, math, os
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Raster canvas with normals ----------------

@dataclass
class RasterCanvas:
    H: int
    W: int
    wall: np.ndarray = field(init=False)
    nx: np.ndarray = field(init=False)
    ny: np.ndarray = field(init=False)
    d:  np.ndarray = field(init=False)
    corridor_mask: np.ndarray = field(init=False)

    def __post_init__(self):
        self.wall = np.zeros((self.H, self.W), dtype=bool)
        self.nx   = np.zeros((self.H, self.W), dtype=np.float32)
        self.ny   = np.zeros((self.H, self.W), dtype=np.float32)
        self.d    = np.full((self.H, self.W), np.inf, dtype=np.float32)
        self.corridor_mask = np.zeros((self.H, self.W), dtype=bool)

    def _paint_segment(self, p0, p1, width_px, set_wall=True, mark_corridor=False):
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        w = max(1.0, float(width_px))
        r = w / 2.0

        xmin = int(np.floor(min(x0, x1) - r - 2))
        xmax = int(np.ceil (max(x0, x1) + r + 2))
        ymin = int(np.floor(min(y0, y1) - r - 2))
        ymax = int(np.ceil (max(y0, y1) + r + 2))

        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(self.W - 1, xmax); ymax = min(self.H - 1, ymax)
        if xmin > xmax or ymin > ymax:
            return

        xs = np.arange(xmin, xmax + 1)
        ys = np.arange(ymin, ymax + 1)
        gx, gy = np.meshgrid(xs, ys)

        dx = x1 - x0
        dy = y1 - y0
        L2 = dx*dx + dy*dy
        if L2 == 0:
            cx = np.full_like(gx, x0, dtype=np.float32)
            cy = np.full_like(gy, y0, dtype=np.float32)
        else:
            t = ((gx - x0) * dx + (gy - y0) * dy) / L2
            t = np.clip(t, 0.0, 1.0)
            cx = x0 + t * dx
            cy = y0 + t * dy

        vx = gx - cx; vy = gy - cy
        d2 = vx*vx + vy*vy
        inside = d2 <= (r*r)

        if set_wall:
            sqrt_d2 = np.sqrt(d2, dtype=np.float32)
            closer = inside & (sqrt_d2 < self.d[ymin:ymax+1, xmin:xmax+1])
            if np.any(closer):
                self.wall[ymin:ymax+1, xmin:xmax+1][inside] = True
                self.d[ymin:ymax+1, xmin:xmax+1][closer] = sqrt_d2[closer].astype(np.float32)
                nrm = sqrt_d2[closer]
                nz = nrm > 1e-6
                nx_new = np.zeros_like(nrm, dtype=np.float32)
                ny_new = np.zeros_like(nrm, dtype=np.float32)
                nx_new[nz] = (vx[closer][nz] / nrm[nz]).astype(np.float32)
                ny_new[nz] = (vy[closer][nz] / nrm[nz]).astype(np.float32)
                self.nx[ymin:ymax+1, xmin:xmax+1][closer] = nx_new
                self.ny[ymin:ymax+1, xmin:xmax+1][closer] = ny_new
        else:
            if np.any(inside):
                sub = self.wall[ymin:ymax+1, xmin:xmax+1]
                will_clear = inside & sub
                self.wall[ymin:ymax+1, xmin:xmax+1][will_clear] = False
                self.nx[ymin:ymax+1, xmin:xmax+1][will_clear] = 0.0
                self.ny[ymin:ymax+1, xmin:xmax+1][will_clear] = 0.0
                self.d[ymin:ymax+1, xmin:xmax+1][will_clear]  = np.inf
        if mark_corridor and np.any(inside):
            self.corridor_mask[ymin:ymax+1, xmin:xmax+1] |= inside

    def paint_polyline(self, pts, width_px, closed=False, mark_corridor=False):
        seq = pts + [pts[0]] if (closed and len(pts)>=2) else pts
        for i in range(len(seq)-1):
            self._paint_segment(seq[i], seq[i+1], width_px, set_wall=True, mark_corridor=mark_corridor)

    def carve_polyline(self, pts, width_px, closed=False, mark_corridor=False):
        seq = pts + [pts[0]] if (closed and len(pts)>=2) else pts
        for i in range(len(seq)-1):
            self._paint_segment(seq[i], seq[i+1], width_px, set_wall=False, mark_corridor=mark_corridor)

    def paint_bezier_quadratic(self, p0, p1, pc, width_px, samples=64, mark_corridor=False):
        t = np.linspace(0.0, 1.0, samples)
        x = (1 - t)**2 * p0[0] + 2*(1 - t)*t * pc[0] + t**2 * p1[0]
        y = (1 - t)**2 * p0[1] + 2*(1 - t)*t * pc[1] + t**2 * p1[1]
        pts = list(zip(x, y))
        self.paint_polyline(pts, width_px, closed=False, mark_corridor=mark_corridor)

    def paint_rect_border(self, x0, y0, x1, y1, width_px, rounded_r_px=0, samples_per_quadrant=24, mark_corridor=False):
        # Ensure proper ordering
        x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
        y0, y1 = (y0, y1) if y0 <= y1 else (y1, y0)

        r = int(max(0, rounded_r_px))
        # Clamp radius to half of side lengths
        r = int(min(r, max(0, (x1 - x0) // 2), max(0, (y1 - y0) // 2)))
        if r == 0:
            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            self.paint_polyline(pts, width_px, closed=True, mark_corridor=mark_corridor)
            return

        def arc(cx, cy, start_ang, end_ang, samples):
            tt = np.linspace(start_ang, end_ang, samples)
            xs = cx + r * np.cos(tt)
            ys = cy + r * np.sin(tt)
            return list(zip(xs, ys))

        pts = []
        # Top edge
        pts.append((x0 + r, y0))
        pts.append((x1 - r, y0))
        # Top-right corner
        pts += arc(x1 - r, y0 + r, -np.pi/2, 0.0, samples_per_quadrant)
        # Right edge
        pts.append((x1, y0 + r))
        pts.append((x1, y1 - r))
        # Bottom-right corner
        pts += arc(x1 - r, y1 - r, 0.0, np.pi/2, samples_per_quadrant)
        # Bottom edge
        pts.append((x1 - r, y1))
        pts.append((x0 + r, y1))
        # Bottom-left corner
        pts += arc(x0 + r, y1 - r, np.pi/2, np.pi, samples_per_quadrant)
        # Left edge
        pts.append((x0, y1 - r))
        pts.append((x0, y0 + r))
        # Top-left corner
        pts += arc(x0 + r, y0 + r, np.pi, 3*np.pi/2, samples_per_quadrant)

        self.paint_polyline(pts, width_px, closed=True, mark_corridor=mark_corridor)

# ---------------- Utilities ----------------

def closing(binmask, r=1):
    if r <= 0:
        return binmask.copy()
    H, W = binmask.shape
    # dilation
    out = np.zeros_like(binmask, dtype=bool)
    for dy in range(-r, r + 1):
        ys = slice(max(0, dy), H + min(0, dy))
        yd = slice(max(0, -dy), H + min(0, -dy))
        for dx in range(-r, r + 1):
            xs = slice(max(0, dx), W + min(0, dx))
            xd = slice(max(0, -dx), W + min(0, -dx))
            out[yd, xd] |= binmask[ys, xs]
    dil = out
    # erosion
    out2 = np.ones_like(binmask, dtype=bool)
    for dy in range(-r, r + 1):
        ys = slice(max(0, dy), H + min(0, dy))
        yd = slice(max(0, -dy), H + min(0, -dy))
        for dx in range(-r, r + 1):
            xs = slice(max(0, dx), W + min(0, dx))
            xd = slice(max(0, -dx), W + min(0, -dx))
            out2[yd, xd] &= dil[ys, xs]
    return out2

def connected_components(binmask):
    H, W = binmask.shape
    labels = np.full((H, W), -1, dtype=np.int32)
    comp_id = 0
    sizes = {}
    for y in range(H):
        for x in range(W):
            if binmask[y, x] and labels[y, x] == -1:
                qx = [x]; qy = [y]
                labels[y, x] = comp_id
                size = 1
                head = 0
                while head < len(qx):
                    cx, cy = qx[head], qy[head]
                    head += 1
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx = cx + dx; ny = cy + dy
                        if 0 <= nx < W and 0 <= ny < H:
                            if binmask[ny, nx] and labels[ny, nx] == -1:
                                labels[ny, nx] = comp_id
                                qx.append(nx); qy.append(ny)
                                size += 1
                sizes[comp_id] = size
                comp_id += 1
    return labels, sizes

def line_length(p0, p1):
    return float(math.hypot(p1[0]-p0[0], p1[1]-p0[1]))

def unit(vx, vy):
    n = math.hypot(vx, vy)
    if n < 1e-6: 
        return (0.0, 0.0)
    return (vx / n, vy / n)

def sample_frequency_mhz(seed: int | None = None, freq_min: int | None = None, freq_max: int | None = None) -> int:
    """
    Return a frequency in MHz.
    If freq_min and freq_max are provided, sample uniformly from [freq_min, freq_max].
    Otherwise, sample from {868, 1800, 3500}.
    Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed if seed is not None else int(np.random.SeedSequence().entropy))
    if freq_min is not None and freq_max is not None:
        return int(rng.integers(freq_min, freq_max + 1))
    choices = np.array([868, 1800, 3500], dtype=np.int64)
    return int(rng.choice(choices))

def sample_antenna_location(wall_mask: np.ndarray, seed: int | None = None) -> tuple:
    """
    Return (x, y) antenna location sampled uniformly over free space (non-wall) pixels.
    If no free-space pixel exists, fallback to image center.
    """
    rng = np.random.default_rng(seed if seed is not None else int(np.random.SeedSequence().entropy))
    free = ~wall_mask
    ys, xs = np.where(free)
    if xs.size == 0:
        raise ValueError("No free-space available for antenna placement")
    k = int(rng.integers(0, xs.size))
    return (int(xs[k]), int(ys[k]))

def euclidean_distance_map(H: int, W: int, x: int, y: int) -> np.ndarray:
    yy, xx = np.indices((H, W))
    return np.sqrt((xx - float(x))**2 + (yy - float(y))**2, dtype=np.float32)

# ---------------- Generator + metadata ----------------

def sample_canvas_size(seed: int) -> tuple:
    """
    Return (width_m, height_m) for the canvas, sampled from embedded ranges.

    - height_m: sampled from a reasonable range for floor-plate extents
      using a log-normal-ish shape and clamped.
    - aspect_ratio: sampled to prefer moderate rectangles (width >= height).
    - width_m = height_m * aspect_ratio

    This function is deterministic for a given seed.
    """
    # Embedded ranges (tweakable here only)
    MIN_AR, MAX_AR = 0.4, 2.5
    MIN_H_M = MAX_H_M = 160.0 / MAX_AR

    rng = np.random.default_rng(int(seed) + 911_531)

    # Height: log-normal-like around ~110 m, then clamp
    base = rng.normal(loc=math.log(110.0), scale=0.18)
    height_m = float(np.clip(math.exp(base), MIN_H_M, MAX_H_M))

    # Aspect ratio: mix to bias towards ~2 while allowing tails
    mix = rng.random()
    if mix < 0.55:
        ar = float(rng.normal(loc=2.0, scale=0.35))
    elif mix < 0.85:
        # Beta mapped to [MIN_AR, MAX_AR]
        u = float(rng.beta(2.2, 2.0))
        ar = MIN_AR + u * (MAX_AR - MIN_AR)
    else:
        ar = float(rng.uniform(MIN_AR, MAX_AR))
    aspect_ratio = float(np.clip(ar, MIN_AR, MAX_AR))

    width_m = float(height_m * aspect_ratio)
    return width_m, height_m

def generate_floor_scene(width_m=None, height_m=None, px_per_m=4, seed=None, freq_min=None, freq_max=None):
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)

    # If size unspecified, sample from modular distribution
    if width_m is None or height_m is None:
        w_m_s, h_m_s = sample_canvas_size(seed)
        width_m = float(w_m_s if width_m is None else width_m)
        height_m = float(h_m_s if height_m is None else height_m)

    W = int(round(width_m * px_per_m))
    H = int(round(height_m * px_per_m))

    canvas = RasterCanvas(H, W)
    ops: List[Dict[str, Any]] = []
    strokes: List[Dict[str, Any]] = []

    params = {'px_per_m': float(px_per_m)}

    ext_th      = float(rng.uniform(0.28, 0.45) * px_per_m)
    part_th     = float(rng.uniform(0.09, 0.14) * px_per_m)
    core_th     = float(rng.uniform(0.20, 0.30) * px_per_m)
    curve_prob  = float(0.20)
    diag_prob   = float(0.08)

    corridor_w        = float(rng.uniform(1.9, 2.7) * px_per_m)
    corridor_wall_th  = float(rng.uniform(0.10, 0.18) * px_per_m)
    belt_offset       = float(rng.uniform(7.0, 12.0) * px_per_m)

    module_m = float(rng.choice([4.2, 4.8, 5.4, 6.0, 6.6, 7.2]))
    module_px = module_m * px_per_m

    # Minimum room width (unchanged policy)
    if rng.random() < 0.08:
        min_room_w_m = rng.uniform(4.2, 4.8)  # rare small but not tiny
    else:
        min_room_w_m = rng.uniform(5.0, 6.5)
    min_room_w = float(min_room_w_m * px_per_m)

    # Single continuous heavy-tailed law per region: Pareto(alpha) * LogNormal
    alpha = 1.2             # heavier tail for more variability across regions
    base_area_m2 = rng.uniform(90.0, 180.0)
    ln_sigma = 0.55         # more spread for non-uniform sizes
    params.update(dict(ext_th=ext_th, part_th=part_th, core_th=core_th,
                       curve_prob=curve_prob, diag_prob=diag_prob,
                       corridor_w=corridor_w, corridor_wall_th=corridor_wall_th,
                       belt_offset=belt_offset, module_m=module_m,
                       min_room_w_m=min_room_w_m, alpha=alpha, base_area_m2=base_area_m2,
                       ln_sigma=ln_sigma))

    # ---- Footprint ----
    round_r = float(rng.uniform(0.0, 6.0) * px_per_m if rng.random() < 0.65 else 0.0)
    canvas.paint_rect_border(2, 2, W-3, H-3, ext_th, rounded_r_px=int(round_r))
    strokes.append(dict(kind="rect_border", layer="exterior", x0=2, y0=2, x1=W-3, y1=H-3, width_px=ext_th, rounded_r_px=int(round_r)))

    # removed unused variable 'inset'

    # ---- Cores ----
    n_cores = int(rng.integers(1, 4))
    core_rects = []
    for _ in range(n_cores):
        side_m = rng.uniform(6.5, 12.0)
        cw = int(side_m * px_per_m)
        ch = int((side_m * rng.uniform(0.85, 1.2)) * px_per_m)
        cx = int(rng.integers(W // 5, 4 * W // 5 - cw))
        cy = int(rng.integers(H // 5, 4 * H // 5 - ch))
        rect = (cx, cy, cx + cw, cy + ch)
        core_rects.append(rect)
        canvas.paint_rect_border(*rect, core_th, rounded_r_px=int(rng.uniform(0, 2.0) * px_per_m))
        strokes.append(dict(kind="rect_border", layer="core", x0=int(rect[0]), y0=int(rect[1]), x1=int(rect[2]), y1=int(rect[3]), width_px=float(core_th)))

    # ---- Corridor ring ----
    rx0 = int(2 + belt_offset); ry0 = int(2 + belt_offset)
    rx1 = int(W - 3 - belt_offset); ry1 = int(H - 3 - belt_offset)
    if rx1 - rx0 > 40 and ry1 - ry0 > 40:
        canvas.paint_rect_border(rx0, ry0, rx1, ry1, corridor_w + 2*corridor_wall_th, rounded_r_px=int(round_r * 0.6))
        strokes.append(dict(kind="rect_border", layer="corridor_wall", x0=rx0, y0=ry0, x1=rx1, y1=ry1,
                            width_px=float(corridor_w + 2*corridor_wall_th), rounded_r_px=int(round_r * 0.6)))
        canvas.carve_polyline([(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)], corridor_w, closed=True, mark_corridor=True)
        ops.append(dict(op="carve_corridor_ring", x0=rx0, y0=ry0, x1=rx1, y1=ry1, width_px=float(corridor_w), rounded_r_px=int(round_r * 0.6)))

    # ---- Corridor spines ----
    n_spines = int(rng.integers(1, 4))
    for _ in range(n_spines):
        if rng.random() < 0.6:
            x = int(rng.integers(int(W*0.25), int(W*0.75)))
            canvas._paint_segment((x, 3), (x, H-4), corridor_w + 2*corridor_wall_th, set_wall=True)
            strokes.append(dict(kind="segment", layer="corridor_wall", p0=(x, 3), p1=(x, H-4), width_px=float(corridor_w + 2*corridor_wall_th)))
            canvas._paint_segment((x, 3), (x, H-4), corridor_w, set_wall=False, mark_corridor=True)
            ops.append(dict(op="carve_corridor_spine", p0=(x, 3), p1=(x, H-4), width_px=float(corridor_w)))
        else:
            y = int(rng.integers(int(H*0.25), int(H*0.75)))
            canvas._paint_segment((3, y), (W-4, y), corridor_w + 2*corridor_wall_th, set_wall=True)
            strokes.append(dict(kind="segment", layer="corridor_wall", p0=(3, y), p1=(W-4, y), width_px=float(corridor_w + 2*corridor_wall_th)))
            canvas._paint_segment((3, y), (W-4, y), corridor_w, set_wall=False, mark_corridor=True)
            ops.append(dict(op="carve_corridor_spine", p0=(3, y), p1=(W-4, y), width_px=float(corridor_w)))

    # Connect cores
    def nearest_point_on_ring(cx, cy):
        cands = [(cx, ry0), (cx, ry1), (rx0, cy), (rx1, cy)]
        d2 = [(cx - px)**2 + (cy - py)**2 for (px, py) in cands]
        return cands[int(np.argmin(d2))]
    for (x0, y0, x1, y1) in core_rects:
        cx = (x0 + x1) // 2; cy = (y0 + y1) // 2
        tx, ty = nearest_point_on_ring(cx, cy)
        canvas._paint_segment((cx, cy), (tx, ty), corridor_w + 2*corridor_wall_th, set_wall=True)
        strokes.append(dict(kind="segment", layer="corridor_wall", p0=(int(cx), int(cy)), p1=(int(tx), int(ty)), width_px=float(corridor_w + 2*corridor_wall_th)))
        canvas._paint_segment((cx, cy), (tx, ty), corridor_w, set_wall=False, mark_corridor=True)
        ops.append(dict(op="carve_corridor_link", p0=(int(cx), int(cy)), p1=(int(tx), int(ty)), width_px=float(corridor_w)))

    # ---- Partitioning with spatially varying heavy-tailed threshold per region ----
    inner = (int(ext_th + px_per_m * 0.6), int(ext_th + px_per_m * 0.6),
             W - int(ext_th + px_per_m * 0.6) - 1, H - int(ext_th + px_per_m * 0.6) - 1)
    stack = [inner]
    splits_drawn = 0
    # Scale partition budget with canvas area to keep density roughly stable
    A_px = float(W * H)
    # Reduce wall density: fewer partition splits per unit area to lower walls/area ratio
    splits_density = 24.0e-5  # previously 1.5e-4 (~48 splits for 800x400); now ~half
    max_splits = int(np.clip(splits_density * A_px, 35, 150))

    # Spatial bias map: encourages larger rooms in some areas and smaller in others
    gh = max(3, int(round(H / max(1, int(40 * px_per_m)))))
    gw = max(3, int(round(W / max(1, int(40 * px_per_m)))))
    region_grid = np.exp(rng.normal(0.0, 0.60, size=(gh, gw))).astype(np.float32)
    up_h = (H + gh - 1) // gh
    up_w = (W + gw - 1) // gw
    scale_map = np.kron(region_grid, np.ones((up_h, up_w), dtype=np.float32))[:H, :W]
    for _ in range(2):
        scale_map = (np.roll(scale_map, 1, axis=1) + scale_map + np.roll(scale_map, -1, axis=1)) / 3.0
        scale_map = (np.roll(scale_map, 1, axis=0) + scale_map + np.roll(scale_map, -1, axis=0)) / 3.0
    scale_map = scale_map / max(1e-6, float(scale_map.mean()))
    scale_map = np.clip(scale_map, 0.7, 2.2).astype(np.float32)

    def area_stop_threshold_px2(cx, cy):
        # A_m2 ~ base * lognormal(0, ln_sigma) * (1 + Pareto(alpha))
        ln_scale = math.exp(rng.normal(0.0, ln_sigma))
        A_m2 = base_area_m2 * ln_scale * (1.0 + rng.pareto(alpha))
        cxi = int(min(max(int(cx), 0), W - 1))
        cyi = int(min(max(int(cy), 0), H - 1))
        A_m2 *= float(scale_map[cyi, cxi])
        return A_m2 * (px_per_m ** 2)

    def draw_split(rect, vertical, s, curve=False, diag=False):
        x0, y0, x1, y1 = rect
        if vertical:
            if curve:
                p0 = (s, y0); p1 = (s, y1)
                ctrl = (s + rng.normal(0, module_px * 0.6), (y0 + y1) / 2 + rng.normal(0, module_px * 0.6))
                canvas.paint_bezier_quadratic(p0, p1, ctrl, part_th, samples=48)
                strokes.append(dict(kind="quad_bezier", layer="partition", p0=(float(p0[0]), float(p0[1])),
                                    p1=(float(p1[0]), float(p1[1])), pc=(float(ctrl[0]), float(ctrl[1])), width_px=float(part_th)))
            elif diag:
                dy = rng.integers(-int((y1 - y0)*0.25), int((y1 - y0)*0.25))
                p0 = (s, y0); p1 = (s + dy, y1)
                canvas._paint_segment(p0, p1, part_th, set_wall=True)
                strokes.append(dict(kind="segment", layer="partition", p0=(int(p0[0]), int(p0[1])), p1=(int(p1[0]), int(p1[1])), width_px=float(part_th)))
            else:
                p0, p1 = (s, y0), (s, y1)
                canvas._paint_segment(p0, p1, part_th, set_wall=True)
                strokes.append(dict(kind="segment", layer="partition", p0=(int(p0[0]), int(p0[1])), p1=(int(p1[0]), int(p1[1])), width_px=float(part_th)))
        else:
            if curve:
                p0 = (x0, s); p1 = (x1, s)
                ctrl = ((x0 + x1) / 2 + rng.normal(0, module_px * 0.6), s + rng.normal(0, module_px * 0.6))
                canvas.paint_bezier_quadratic(p0, p1, ctrl, part_th, samples=48)
                strokes.append(dict(kind="quad_bezier", layer="partition", p0=(float(p0[0]), float(p0[1])),
                                    p1=(float(p1[0]), float(p1[1])), pc=(float(ctrl[0]), float(ctrl[1])), width_px=float(part_th)))
            elif diag:
                dx = rng.integers(-int((x1 - x0)*0.25), int((x1 - x0)*0.25))
                p0 = (x0, s); p1 = (x1, s + dx)
                canvas._paint_segment(p0, p1, part_th, set_wall=True)
                strokes.append(dict(kind="segment", layer="partition", p0=(int(p0[0]), int(p0[1])), p1=(int(p1[0]), int(p1[1])), width_px=float(part_th)))
            else:
                p0, p1 = (x0, s), (x1, s)
                canvas._paint_segment(p0, p1, part_th, set_wall=True)
                strokes.append(dict(kind="segment", layer="partition", p0=(int(p0[0]), int(p0[1])), p1=(int(p1[0]), int(p1[1])), width_px=float(part_th)))

    while stack and splits_drawn < max_splits:
        areas = [ (r[2]-r[0])*(r[3]-r[1]) for r in stack ]
        idx = int(np.argmax(areas))
        x0, y0, x1, y1 = stack.pop(idx)
        w = x1 - x0; h = y1 - y0
        if w < 2 * min_room_w or h < 2 * min_room_w:
            continue
        A = w * h
        if A < area_stop_threshold_px2((x0 + x1) * 0.5, (y0 + y1) * 0.5):
            continue

        # choose split orientation
        if (w > h * 1.1 and rng.random() < 0.8) or (w > h and rng.random() < 0.6):
            vertical = True
        elif (h > w * 1.1 and rng.random() < 0.8) or (h > w and rng.random() < 0.6):
            vertical = False
        else:
            vertical = rng.random() < 0.5

        # edge-biased split position (continuous)
        if vertical:
            smin = x0 + int(min_room_w); smax = x1 - int(min_room_w)
            if smax - smin < module_px * 1.3: 
                continue
            u = 0.5 * rng.random() + 0.5 * rng.beta(0.65, 2.8)  # mix for continuity
            s = int(np.clip(int(smin + u * (smax - smin)), smin, smax))
            s = int(np.clip(int(rng.normal(s, module_px * 0.12)), smin, smax))
            draw_split((x0, y0, x1, y1), True, s, curve=(rng.random()<curve_prob), diag=(rng.random()<diag_prob))
            stack.append((x0, y0, s, y1)); stack.append((s, y0, x1, y1))
        else:
            smin = y0 + int(min_room_w); smax = y1 - int(min_room_w)
            if smax - smin < module_px * 1.3: 
                continue
            u = 0.5 * rng.random() + 0.5 * rng.beta(0.65, 2.8)
            s = int(np.clip(int(smin + u * (smax - smin)), smin, smax))
            s = int(np.clip(int(rng.normal(s, module_px * 0.12)), smin, smax))
            draw_split((x0, y0, x1, y1), False, s, curve=(rng.random()<curve_prob), diag=(rng.random()<diag_prob))
            stack.append((x0, y0, x1, s)); stack.append((x0, s, x1, y1))
        splits_drawn += 1

    # Clean
    canvas.wall = closing(canvas.wall, r=1)

    # ---- Doors (same as before) ----
    def sample_door_width_m(local_wall_len_m):
        w = float(np.exp(rng.normal(np.log(0.95), 0.18)))
        w = max(0.70, min(w, 2.00, 0.33 * local_wall_len_m))
        return w

    def carve_capsule(center_xy, t_hat, L_px, R_px):
        tx, ty = unit(t_hat[0], t_hat[1])
        if tx == 0 and ty == 0: return
        half = max(1.0, L_px / 2.0)
        p0 = (center_xy[0] - tx * half, center_xy[1] - ty * half)
        p1 = (center_xy[0] + tx * half, center_xy[1] + ty * half)
        canvas._paint_segment(p0, p1, 2*R_px, set_wall=False)
        ops.append(dict(op="carve_door_capsule", p0=(float(p0[0]), float(p0[1])), p1=(float(p1[0]), float(p1[1])), radius_px=float(R_px)))

    # Seed doors: along corridor spines and ring
    def add_corridor_seed_doors():
        center_prims = []
        for o in ops:
            if o.get('op') in ('carve_corridor_spine', 'carve_corridor_link'):
                center_prims.append(('segment', o['p0'], o['p1']))
        for s in strokes:
            if s.get('kind') == 'rect_border' and s.get('layer') == 'corridor_wall':
                x0, y0, x1, y1 = s['x0'], s['y0'], s['x1'], s['y1']
                center_prims += [
                    ('segment', (x0, y0), (x1, y0)),
                    ('segment', (x1, y0), (x1, y1)),
                    ('segment', (x1, y1), (x0, y1)),
                    ('segment', (x0, y1), (x0, y0)),
                ]
                break
        for kind, p0, p1 in center_prims:
            L = line_length(p0, p1)
            exp_spacing_px = (np.random.uniform(8.0, 15.0) * px_per_m)
            n_doors = max(1, int(L / exp_spacing_px))
            for _ in range(n_doors):
                t = np.random.random()
                cx = p0[0] + t * (p1[0] - p0[0])
                cy = p0[1] + t * (p1[1] - p0[1])
                tx = (p1[0] - p0[0]); ty = (p1[1] - p0[1])
                txu, tyu = unit(tx, ty)
                nx, ny = -tyu, txu
                side = 1.0 if np.random.random() < 0.5 else -1.0
                door_center = (cx + side * nx * (corridor_w/2 + corridor_wall_th*0.75),
                               cy + side * ny * (corridor_w/2 + corridor_wall_th*0.75))
                local_len_m = (L / px_per_m)
                clear_m = sample_door_width_m(local_len_m)
                carve_capsule(door_center, (txu, tyu), L_px=clear_m*px_per_m, R_px=corridor_wall_th*0.7 + 1.5)

    add_corridor_seed_doors()

    # Ensure most rooms have at least one door
    def ensure_room_doors():
        space = ~canvas.wall
        labels, sizes = connected_components(space)
        corridor_ids = set(np.unique(labels[canvas.corridor_mask])); corridor_ids.discard(-1)
        if not corridor_ids: return
        H, W = space.shape
        all_ids = set(np.unique(labels)); all_ids.discard(-1)
        room_ids = [cid for cid in all_ids if cid not in corridor_ids]
        np.random.shuffle(room_ids)
        for cid in room_ids:
            if np.random.random() < 0.03:  # rare doorless
                continue
            mask_room = labels == cid
            bd = np.zeros_like(mask_room, dtype=bool)
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dx==0 and dy==0: continue
                    ys = slice(max(0, dy), H + min(0, dy))
                    yd = slice(max(0, -dy), H + min(0, -dy))
                    xs = slice(max(0, dx), W + min(0, dx))
                    xd = slice(max(0, -dx), W + min(0, -dx))
                    bd[yd, xd] |= mask_room[ys, xs]
            bd &= ~mask_room
            near_room = bd
            near_corr = np.zeros_like(mask_room, dtype=bool)
            for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                ys = slice(max(0, dy), H + min(0, dy))
                yd = slice(max(0, -dy), H + min(0, -dy))
                xs = slice(max(0, dx), W + min(0, dx))
                xd = slice(max(0, -dx), W + min(0, -dx))
                near_corr[yd, xd] |= canvas.corridor_mask[ys, xs]
            candidate = canvas.wall & near_room & near_corr
            ys, xs = np.where(candidate)
            if len(xs) == 0:
                candidate = canvas.wall & near_room
                ys, xs = np.where(candidate)
                if len(xs) == 0: continue
            k = int(np.random.randint(0, len(xs)))
            x = xs[k]; y = ys[k]
            nx = float(canvas.nx[y, x]); ny = float(canvas.ny[y, x])
            tx, ty = -ny, nx
            L_count = 0
            max_walk = int(12 * px_per_m)
            for s in range(1, max_walk):
                xi = int(round(x + tx*s)); yi = int(round(y + ty*s))
                if not (0 <= xi < W and 0 <= yi < H): break
                if not canvas.wall[yi, xi]: break
                nxd = canvas.nx[yi, xi]; nyd = canvas.ny[yi, xi]
                if nxd*nx + nyd*ny < 0.94: break
                L_count += 1
            for s in range(1, max_walk):
                xi = int(round(x - tx*s)); yi = int(round(y - ty*s))
                if not (0 <= xi < W and 0 <= yi < H): break
                if not canvas.wall[yi, xi]: break
                nxd = canvas.nx[yi, xi]; nyd = canvas.ny[yi, xi]
                if nxd*nx + nyd*ny < 0.94: break
                L_count += 1
            local_len_px = max(4, L_count)
            clear_m = sample_door_width_m(local_len_px / px_per_m)
            half = max(1.0, (clear_m*px_per_m)/2.0)
            p0 = (x - tx*half, y - ty*half)
            p1 = (x + tx*half, y + ty*half)
            canvas._paint_segment(p0, p1, 2*(corridor_wall_th*0.7 + 1.5), set_wall=False)

    ensure_room_doors()

    canvas.wall = closing(canvas.wall, r=1)
    canvas.nx[~canvas.wall] = 0.0
    canvas.ny[~canvas.wall] = 0.0

    # Antenna location and distance map
    # Antenna location; if none available, regenerate with a different seed
    try:
        ant_x, ant_y = sample_antenna_location(canvas.wall, seed + 12345)
    except ValueError:
        # Try a few alternative seeds deterministically derived from the base
        for bump in (101, 202, 303, 404, 505):
            try:
                ant_x, ant_y = sample_antenna_location(canvas.wall, seed + 12345 + bump)
                break
            except ValueError:
                continue
        else:
            # As last resort, restart the whole generation with a new seed
            return generate_floor_scene(width_m, height_m, px_per_m, seed + 9999)
    dist_map = euclidean_distance_map(H, W, ant_x, ant_y).astype(np.float32)

    # Frequency metadata (for simulators)
    freq_MHz = sample_frequency_mhz(seed + 22222, freq_min=freq_min, freq_max=freq_max)

    scene = dict(
        version=4,
        seed=int(seed),
        canvas=dict(width_m=float(width_m), height_m=float(height_m), px_per_m=float(px_per_m),
                    W=int(W), H=int(H)),
        params=params,
        strokes=strokes,
        ops=ops,
        antenna=dict(x=int(ant_x), y=int(ant_y)),
        frequency_mhz=int(freq_MHz)
    )
    normals = np.stack([canvas.nx, canvas.ny], axis=-1).astype(np.float32)

    # ---- Reflectance and transmittance maps (for wall pixels) ----
    wall_mask = canvas.wall
    reflectance = np.zeros_like(wall_mask, dtype=np.float32)
    transmittance = np.zeros_like(wall_mask, dtype=np.float32)
    if np.any(wall_mask):
        # Default: reflectance smaller than transmittance; 20% of cases inverted
        invert_rare = rng.random() < 0.20
        extreme_mix = rng.random() < 0.15

        if not invert_rare:
            # Typical case: smaller reflectance, larger transmittance
            if extreme_mix:
                # Make it more extreme: very small reflectance with larger transmittance
                base_reflect = float(rng.uniform(0.1, 1.0))
                base_trans   = float(rng.uniform(15.0, 30.0))
            else:
                base_reflect = float(rng.uniform(0.5, 8.0))
                base_trans   = float(rng.uniform(8.0, 25.0))
        else:
            # Inverted case (rare ~20%): transmittance smaller than reflectance
            if extreme_mix:
                # Very small transmittance with higher reflectance (but respect refl cap ~15)
                base_trans   = float(rng.uniform(0.1, 1.0))
                base_reflect = float(rng.uniform(10.0, 15.0))
            else:
                base_trans   = float(rng.uniform(0.5, 8.0))
                base_reflect = float(rng.uniform(8.0, 15.0))

        reflectance[wall_mask] = float(np.clip(base_reflect, 0.0, 15.0))
        # Allow transmittance to exceed 15 in non-window regions for variety (as before)
        transmittance[wall_mask] = float(np.clip(base_trans, 0.0, 30.0))

        # Create low-frequency patches to simulate windows/thin parts with lower transmittance loss
        Hm, Wm = wall_mask.shape
        gh = max(1, Hm // 32)
        gw = max(1, Wm // 32)
        noise_grid = rng.random((gh, gw)).astype(np.float32)
        up_h = (Hm + gh - 1) // gh
        up_w = (Wm + gw - 1) // gw
        noise = np.kron(noise_grid, np.ones((up_h, up_w), dtype=np.float32))[:Hm, :Wm]
        for _ in range(2):
            noise = (np.roll(noise, 1, axis=1) + noise + np.roll(noise, -1, axis=1)) / 3.0
            noise = (np.roll(noise, 1, axis=0) + noise + np.roll(noise, -1, axis=0)) / 3.0
        windows_mask = (noise > 0.85) & wall_mask
        if np.any(windows_mask):
            # Slightly expand patches
            wm = windows_mask.copy()
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                ys = slice(max(0, dy), Hm + min(0, dy))
                yd = slice(max(0, -dy), Hm + min(0, -dy))
                xs = slice(max(0, dx), Wm + min(0, dx))
                xd = slice(max(0, -dx), Wm + min(0, -dx))
                wm[yd, xd] |= windows_mask[ys, xs]
            # In window/patch regions, modulate the smaller quantity further for contrast
            if not invert_rare:
                # Make some areas thinner: reduce transmittance (upper bound ~15 inside patches)
                red_factor = float(rng.uniform(0.3, 0.7))
                trans_val = float(np.clip(base_trans * red_factor, 0.5, 15.0))
                transmittance[wm] = trans_val
                # Slight tweaks to reflectance
                rf_factor = float(rng.uniform(0.85, 1.05))
                refl_val = float(np.clip(base_reflect * rf_factor, 0.1, 15.0))
                reflectance[wm] = refl_val
            else:
                # In inverted cases, reduce reflectance in patches for variety
                rf_factor = float(rng.uniform(0.5, 0.9))
                refl_val = float(np.clip(base_reflect * rf_factor, 0.1, 15.0))
                reflectance[wm] = refl_val
                # And keep transmittance a bit higher within reasonable bounds
                tf_factor = float(rng.uniform(0.9, 1.1))
                trans_val = float(np.clip(base_trans * tf_factor, 0.1, 20.0))
                transmittance[wm] = trans_val

    return canvas.wall, normals, scene, reflectance, transmittance, dist_map

# ---------------- UI + save ----------------

def _gen_id(scene: Dict[str, Any]) -> str:
    seed = int(scene.get("seed", 0))
    c = scene.get("canvas", {})
    w_m = float(c.get("width_m", 0.0))
    h_m = float(c.get("height_m", 0.0))
    return f"seed{seed}_w{int(round(w_m))}m_h{int(round(h_m))}m"

def save_artifacts(mask, normals, scene, reflectance, transmittance, dist_map, prefix_dir="/Users/xoren/icassp2025/generated_rooms"):
    os.makedirs(prefix_dir, exist_ok=True)
    gen_id = _gen_id(scene)
    prefix = os.path.join(prefix_dir, gen_id)
    np.save(prefix + "_mask.npy", mask.astype(np.uint8))
    np.save(prefix + "_normals.npy", normals.astype(np.float32))
    np.save(prefix + "_reflectance.npy", reflectance.astype(np.float32))
    np.save(prefix + "_transmittance.npy", transmittance.astype(np.float32))
    # Save composite RGB image (R=refl, G=trans, B=distance)
    # Normalize distance to [0, 1] per image for visualization; keep refl/trans in [1,15] mapped to [0,1]
    dist_norm = dist_map / max(1e-6, float(np.max(dist_map)))
    refl_norm = (reflectance - 1.0) / 14.0
    trans_norm = (transmittance - 1.0) / 14.0
    rgb = np.stack([
        np.clip(refl_norm, 0.0, 1.0),
        np.clip(trans_norm, 0.0, 1.0),
        np.clip(dist_norm, 0.0, 1.0)
    ], axis=-1).astype(np.float32)
    np.save(prefix + "_rt_antenna_rgb.npy", rgb)
    with open(prefix + ".json", "w") as f:
        json.dump(scene, f, indent=2)

def draw_mask(ax, mask):
    ax.clear()
    ax.imshow(mask, interpolation="nearest")  # default colormap
    ax.set_title("Wall mask (1=wall)")
    ax.set_axis_off()

def draw_overview(fig, axes, mask, reflectance, transmittance):
    ax_mask, ax_refl, ax_trans = axes
    # Mask panel
    ax_mask.clear(); ax_mask.set_axis_off()
    ax_mask.imshow(mask, cmap="gray", interpolation="nearest")
    ax_mask.set_title("Wall mask")

    # Reflectance panel (mask non-walls)
    ax_refl.clear(); ax_refl.set_axis_off()
    refl_vis = np.ma.masked_where(~mask, reflectance)
    im_refl = ax_refl.imshow(refl_vis, cmap="magma", vmin=1.0, vmax=15.0, interpolation="nearest")
    ax_refl.set_title("Reflectance (1–15)")
    fig.colorbar(im_refl, ax=ax_refl, fraction=0.046, pad=0.04)

    # Transmittance panel (mask non-walls)
    ax_trans.clear(); ax_trans.set_axis_off()
    trans_vis = np.ma.masked_where(~mask, transmittance)
    im_trans = ax_trans.imshow(trans_vis, cmap="viridis", vmin=1.0, vmax=15.0, interpolation="nearest")
    ax_trans.set_title("Transmittance (1–15)")
    fig.colorbar(im_trans, ax=ax_trans, fraction=0.046, pad=0.04)

def show_generator():
    # Initial generation and save
    mask, normals, scene, refl, trans, dist = generate_floor_scene()
    save_artifacts(mask, normals, scene, refl, trans, dist)

    ant = scene.get("antenna", {})
    ax0, ay0 = int(ant.get("x", 0)), int(ant.get("y", 0))

    # Create figure with four panels once
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for ax in axes:
        ax.set_axis_off()

    # Mask panel
    axes[0].set_title("Wall mask")
    im_mask = axes[0].imshow(mask, cmap="gray", interpolation="nearest", origin="upper")
    star0 = axes[0].scatter([ax0], [ay0], c="red", marker="*", s=120, edgecolors="black", linewidths=0.5)

    # Reflectance panel
    axes[1].set_title("Reflectance (1–15)")
    im_refl = axes[1].imshow(np.ma.masked_where(~mask, refl), cmap="magma", vmin=1.0, vmax=15.0, interpolation="nearest", origin="upper")
    fig.colorbar(im_refl, ax=axes[1], fraction=0.046, pad=0.04)
    star1 = axes[1].scatter([ax0], [ay0], c="red", marker="*", s=120, edgecolors="black", linewidths=0.5)

    # Transmittance panel
    axes[2].set_title("Transmittance (1–15)")
    im_trans = axes[2].imshow(np.ma.masked_where(~mask, trans), cmap="viridis", vmin=1.0, vmax=15.0, interpolation="nearest", origin="upper")
    fig.colorbar(im_trans, ax=axes[2], fraction=0.046, pad=0.04)
    star2 = axes[2].scatter([ax0], [ay0], c="red", marker="*", s=120, edgecolors="black", linewidths=0.5)

    # Distance panel (normalized: antenna location should be near 0, thus whitish in 'Blues')
    axes[3].set_title("Distance from antenna")
    dist_norm = dist / max(1e-6, float(np.max(dist)))
    im_dist = axes[3].imshow(dist_norm, cmap="plasma", vmin=0.0, vmax=1.0, interpolation="nearest", origin="upper")
    fig.colorbar(im_dist, ax=axes[3], fraction=0.046, pad=0.04)
    star3 = axes[3].scatter([ax0], [ay0], c="red", marker="*", s=120, edgecolors="black", linewidths=0.5, zorder=3)

    # Optional debug: verify distance minimum equals antenna location
    debug = os.environ.get("RG_DEBUG", "0") == "1"
    min_marker = None
    if debug:
        mn_idx = int(np.argmin(dist)); my, mx = np.unravel_index(mn_idx, dist.shape)
        d_at = float(dist[int(ay0), int(ax0)]) if 0 <= ay0 < dist.shape[0] and 0 <= ax0 < dist.shape[1] else float('nan')
        print(f"[RG_DEBUG] Ant=({ax0},{ay0}) d_at={d_at:.6f} min_d={float(dist.min()):.6f} at=({mx},{my})")
        min_marker = axes[3].scatter([mx], [my], c="lime", marker="x", s=60, linewidths=1.5, zorder=3)

    # Button (Matplotlib)
    from matplotlib.widgets import Button
    plt.subplots_adjust(bottom=0.16)
    bax = plt.axes([0.42, 0.04, 0.16, 0.08])
    btn = Button(bax, "New floor")

    def on_click(event):
        # Generate a new layout and update images without creating new artists
        m, n, s, r, t, d = generate_floor_scene()
        save_artifacts(m, n, s, r, t, d)
        antn = s.get("antenna", {})
        axn, ayn = int(antn.get("x", 0)), int(antn.get("y", 0))
        im_mask.set_data(m)
        im_refl.set_data(np.ma.masked_where(~m, r))
        im_trans.set_data(np.ma.masked_where(~m, t))
        im_dist.set_data(d / max(1e-6, float(np.max(d))))
        # Ensure axes limits match new dimensions so stars are always visible
        Hn, Wn = m.shape
        for ax in axes:
            ax.set_xlim(-0.5, Wn - 0.5)
            ax.set_ylim(Hn - 0.5, -0.5)
        # Update star positions
        star0.set_offsets(np.array([[axn, ayn]]))
        star1.set_offsets(np.array([[axn, ayn]]))
        star2.set_offsets(np.array([[axn, ayn]]))
        star3.set_offsets(np.array([[axn, ayn]]))
        if os.environ.get("RG_DEBUG", "0") == "1":
            mn_idx2 = int(np.argmin(d)); my2, mx2 = np.unravel_index(mn_idx2, d.shape)
            d_at2 = float(d[int(ayn), int(axn)]) if 0 <= ayn < d.shape[0] and 0 <= axn < d.shape[1] else float('nan')
            print(f"[RG_DEBUG] (New) Ant=({axn},{ayn}) d_at={d_at2:.6f} min_d={float(d.min()):.6f} at=({mx2},{my2})")
            if min_marker is not None:
                min_marker.set_offsets(np.array([[mx2, my2]]))
        fig.canvas.draw_idle()

    btn.on_clicked(on_click)
    plt.show()

if __name__ == "__main__":
    show_generator()

# ---------------- Self-tests (gated by RG_TESTS=1) ----------------
if os.environ.get("RG_TESTS", "0") == "1":
    # Deterministic seed for repeatability
    mask, normals, scene, refl, trans, dist = generate_floor_scene(seed=1234)

    # Shapes and dtypes
    assert mask.dtype == np.bool_, "mask must be boolean"
    assert normals.dtype == np.float32 and normals.shape[-1] == 2, "normals must be float32 HxWx2"
    H, W = mask.shape
    assert normals.shape[:2] == (H, W), "normals must match mask shape"

    # Non-walls must have zero normals
    nz = np.any(normals != 0.0, axis=-1)
    assert not np.any(nz & (~mask)), "non-wall pixels must have zero normals"

    # Some wall pixels should have non-zero normals
    wall = mask
    wall_nz = wall & nz
    assert np.any(wall_nz), "expected some wall pixels to have normals"

    # Normals should be approximately unit length where present
    mag = np.linalg.norm(normals, axis=-1)
    mag_wall_nz = mag[wall_nz]
    assert np.isfinite(mag_wall_nz).all(), "wall normal magnitudes must be finite"
    avg_mag_wall_nz = float(mag_wall_nz.mean()) if mag_wall_nz.size > 0 else float("nan")
    q05 = q95 = None
    if mag_wall_nz.size > 0:
        q05 = float(np.quantile(mag_wall_nz, 0.05))
        q95 = float(np.quantile(mag_wall_nz, 0.95))
        assert q05 >= 0.75, f"5th percentile of wall normal magnitude too low: {q05:.3f}"
        assert q95 <= 1.05, f"95th percentile of wall normal magnitude too high: {q95:.3f}"

    # Coverage: a reasonable fraction of wall pixels should carry normals (closing may add walls without normals)
    wall_count = int(wall.sum())
    coverage = (int(wall_nz.sum()) / max(1, wall_count)) if wall_count > 0 else 0.0
    assert coverage >= 0.10, f"wall normal coverage too low: {coverage:.3f} (< 0.10)"

    # Neighbor alignment: many adjacent wall-normal pairs should be roughly aligned (dot > 0.2)
    aligned = 0
    total_pairs = 0
    for dy, dx in ((1,0),(0,1)):
        ys = slice(max(0, dy), H)
        xs = slice(max(0, dx), W)
        yd = slice(0, H - dy)
        xd = slice(0, W - dx)
        pair_mask = wall_nz[ys, xs] & wall_nz[yd, xd]
        if np.any(pair_mask):
            v1 = normals[ys, xs][pair_mask]
            v2 = normals[yd, xd][pair_mask]
            dots = np.sum(v1 * v2, axis=-1)
            aligned += int(np.sum(dots > 0.2))
            total_pairs += int(dots.size)
    frac_aligned = float("nan")
    if total_pairs > 0:
        frac_aligned = aligned / total_pairs
        assert frac_aligned >= 0.25, f"insufficient neighbor alignment: {frac_aligned:.3f} (< 0.25)"

    # Scene metadata sanity
    assert scene["canvas"]["W"] == W and scene["canvas"]["H"] == H
    assert isinstance(scene.get("strokes"), list) and isinstance(scene.get("ops"), list)
    assert scene.get("version") == 4

    # Sampler determinism and ranges
    w1, h1 = sample_canvas_size(1234)
    w2, h2 = sample_canvas_size(1234)
    assert (w1, h1) == (w2, h2), "sample_canvas_size must be deterministic for a given seed"
    assert 70.0 <= h1 <= 160.0 and 70.0 <= h2 <= 160.0, "height out of embedded range"
    ar = w1 / max(1e-6, h1)
    assert 0.4 <= ar <= 2.5, "aspect ratio out of embedded range"

    # Save ID format check
    gen_id = _gen_id(scene)
    assert str(scene["seed"]) in gen_id and "w" in gen_id and "h" in gen_id, "gen id format unexpected"

    # Reflectance/transmittance sanity
    assert refl.shape == mask.shape and trans.shape == mask.shape
    assert refl.dtype == np.float32 and trans.dtype == np.float32
    # Non-walls must be zero in refl/trans
    assert np.all(refl[~mask] == 0.0) and np.all(trans[~mask] == 0.0)
    if np.any(mask):
        # Walls within specified ranges
        rmin, rmax = float(np.min(refl[mask])), float(np.max(refl[mask]))
        tmin, tmax = float(np.min(trans[mask])), float(np.max(trans[mask]))
        assert 1.0 <= rmin <= 15.0 and 1.0 <= rmax <= 15.0
        assert 1.0 <= tmin <= 15.0 and 1.0 <= tmax <= 15.0

    # Antenna location and distance map sanity
    ant = scene.get("antenna", {})
    assert "x" in ant and "y" in ant
    ax, ay = int(ant["x"]), int(ant["y"])
    assert 0 <= ax < W and 0 <= ay < H
    assert dist.shape == mask.shape and dist.dtype == np.float32
    # Distance at antenna should be ~0
    assert dist[ay, ax] <= 1e-3

    # If all assertions passed, print a concise success summary with key values
    print("[RG_TESTS] All tests passed.")
    if q05 is not None and q95 is not None:
        print(f"[RG_TESTS] Wall normal magnitudes: mean={avg_mag_wall_nz:.3f}, 5th={q05:.3f}, 95th={q95:.3f} (expected ~[0.75, 1.05]).")
    print(f"[RG_TESTS] Wall normal coverage: {coverage:.2%} (>= 10%).")
    if not math.isnan(frac_aligned):
        print(f"[RG_TESTS] Neighbor alignment: {frac_aligned:.2%} of adjacent pairs aligned (>= 25%).")
    print(f"[RG_TESTS] Sampler: height={h1:.1f} m within [70, 160], aspect_ratio={ar:.2f} within [0.4, 2.5].")
    # Reflectance/transmittance summary
    if np.any(mask):
        rmin, rmax = float(np.min(refl[mask])), float(np.max(refl[mask]))
        tmin, tmax = float(np.min(trans[mask])), float(np.max(trans[mask]))
        print(f"[RG_TESTS] Reflectance range on walls: [{rmin:.2f}, {rmax:.2f}] within [1, 15].")
        print(f"[RG_TESTS] Transmittance range on walls: [{tmin:.2f}, {tmax:.2f}] within [1, 15].")
    print(f"[RG_TESTS] Save ID: {gen_id}")
