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
    input_img: torch.Tensor  # In format (C, H, W)
    output_img: torch.Tensor  # In format (H, W) or (1, H, W)
    radiation_pattern: torch.Tensor
    pixel_size: float = 0.25
    mask: Union[torch.Tensor, None] = None
    ids: Optional[List[Tuple[int, int, int, int]]] = None
    normals: Optional[np.ndarray] = None  # (H, W, 2) float array: [...,0]=nx, [...,1]=ny

    def copy(self):
        return RadarSample(
            self.H,
            self.W,
            self.x_ant,
            self.y_ant,
            self.azimuth,
            self.freq_MHz,
            self.input_img,
            self.output_img,
            self.radiation_pattern,
            self.pixel_size,
            self.mask,
            self.ids,
            self.normals,
        )
