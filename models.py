import torch
import numpy as np
from typing import Union, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class RadarSample:
    H: int
    W: int
    x_ant: float
    y_ant: float
    azimuth: float
    freq_MHz: float
    reflectance: torch.Tensor  # (H, W)
    transmittance: torch.Tensor  # (H, W)
    dist_map: torch.Tensor  # (H, W)
    pathloss: torch.Tensor  # (H, W) or (1, H, W)
    radiation_pattern: torch.Tensor
    pixel_size: float = 0.25
    mask: Union[torch.Tensor, None] = None
    ids: Optional[List[Tuple[int, int, int, int]]] = None
    normals: Optional[np.ndarray] = None  # (H, W, 2) float array: [...,0]=nx, [...,1]=ny

    def copy(self):
        return RadarSample(
            H=self.H,
            W=self.W,
            x_ant=self.x_ant,
            y_ant=self.y_ant,
            azimuth=self.azimuth,
            freq_MHz=self.freq_MHz,
            reflectance=self.reflectance,
            transmittance=self.transmittance,
            dist_map=self.dist_map,
            pathloss=self.pathloss,
            radiation_pattern=self.radiation_pattern,
            pixel_size=self.pixel_size,
            mask=self.mask,
            ids=self.ids,
            normals=self.normals,
        )
