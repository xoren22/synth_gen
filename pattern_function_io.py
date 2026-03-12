"""NPZ loader that reconstructs RadarSample from function metadata.

No try/except in this module.
"""
from __future__ import annotations

import json
import numpy as np
import torch

from models import RadarSample
from antenna_pattern import validate_pattern_function_info


def load_metadata_from_npz(npz_path: str) -> dict:
    data = np.load(npz_path, allow_pickle=False)
    meta_str = str(data["meta_json"])
    return json.loads(meta_str)


def radar_sample_from_npz(npz_path: str) -> RadarSample:
    data = np.load(npz_path, allow_pickle=False)
    meta_str = str(data["meta_json"])
    meta = json.loads(meta_str)

    antenna = meta["antenna"]
    pattern_function = antenna.get("pattern_function")
    if pattern_function is None:
        raise ValueError("NPZ missing antenna.pattern_function metadata")
    validate_pattern_function_info(pattern_function)

    shape_hw = meta["shape_hw"]
    H, W = int(shape_hw[0]), int(shape_hw[1])

    reflectance = torch.from_numpy(np.asarray(data["reflectance"], dtype=np.float32))
    transmittance = torch.from_numpy(np.asarray(data["transmittance"], dtype=np.float32))
    pathloss = torch.from_numpy(np.asarray(data["pathloss"], dtype=np.float32))
    radiation_pattern_db = torch.from_numpy(
        np.asarray(data["radiation_pattern_db"], dtype=np.float32)
    )

    # dist_map may or may not be present
    if "dist_map" in data:
        dist_map = torch.from_numpy(np.asarray(data["dist_map"], dtype=np.float32))
    else:
        dist_map = torch.zeros(H, W, dtype=torch.float32)

    mask = None
    if "mask" in data:
        mask = torch.from_numpy(np.asarray(data["mask"], dtype=np.float32))

    normals = None
    if "normals" in data:
        normals = np.asarray(data["normals"], dtype=np.float32)

    return RadarSample(
        H=H,
        W=W,
        x_ant=float(antenna["x_px"]),
        y_ant=float(antenna["y_px"]),
        azimuth=float(antenna["azimuth_deg"]),
        freq_MHz=float(meta.get("frequency_mhz", 1800)),
        reflectance=reflectance,
        transmittance=transmittance,
        dist_map=dist_map,
        pathloss=pathloss,
        radiation_pattern=radiation_pattern_db,
        pixel_size=float(meta.get("pixel_size_m", 0.25)),
        mask=mask,
        normals=normals,
        radiation_pattern_fn_info=pattern_function,
    )
