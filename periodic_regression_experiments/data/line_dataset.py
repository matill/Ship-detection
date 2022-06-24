from __future__ import annotations
from torch import nn as nn, optim
import torch
from torch.utils.data import IterableDataset
from typing import Callable, Dict, Iterator, List, Optional, Tuple
import random
import numpy as np

USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"

LINE_LEN = 200.0
LINE_WIDTH = 10.0

def get_line_side_bitmap(point_on_line: torch.Tensor, norm: torch.Tensor):
    coordinates = torch.cat([
        torch.arange(0, 512, 1, dtype=torch.float32, device=DEVICE)[:, None, None].expand(-1, 512, -1), # y
        torch.arange(0, 512, 1, dtype=torch.float32, device=DEVICE)[None, :, None].expand(512, -1, -1), # x
    ], dim=2)
    assert coordinates.shape == (512, 512, 2)
    dists = coordinates - point_on_line[None, None, :]
    assert dists.shape == (512, 512, 2)
    bitmap = (dists * norm[None, None, :]).sum(dim=2) > 0
    assert bitmap.shape == (512, 512)
    return bitmap

def generate_image(angle_degrees: float) -> torch.Tensor:
    center = torch.tensor([250, 250], dtype=torch.float32, device=DEVICE)
    angle_radians_torch = torch.tensor(angle_degrees * 2 * 3.14159 / 360)
    sin = torch.sin(angle_radians_torch)
    cos = torch.cos(angle_radians_torch)
    front_dir = torch.tensor([-cos, sin], device=DEVICE)
    right_dir = torch.tensor([sin, cos], device=DEVICE)

    front_bitmap = get_line_side_bitmap(center + front_dir * LINE_LEN, -front_dir)
    rear_bitmap =  get_line_side_bitmap(center - front_dir * LINE_LEN, front_dir)
    right_bitmap = get_line_side_bitmap(center + right_dir * LINE_WIDTH, -right_dir)
    left_bitmap =  get_line_side_bitmap(center - right_dir * LINE_WIDTH, right_dir)
    complete_bitmap = (
        front_bitmap
        & rear_bitmap
        & right_bitmap
        & left_bitmap
    ) 
    assert complete_bitmap.shape == (512, 512)
    img = complete_bitmap.float()[None, :, :] * 160 + 10
    img = img.expand(3, -1, -1)
    assert img.shape == (3, 512, 512)
    return img


class LineDataset(IterableDataset):
    def __init__(self, epoch_size: Optional[int]=None, angles_01: Optional[List[int]]=None) -> None:
        super().__init__()
        assert (epoch_size is None) != (angles_01 is None), "Specify EITHER epoch_size xor angles_01"
        self.epoch_size = epoch_size
        self.angles_01 = angles_01

    @staticmethod
    def get_train_ds(epoch_size: int) -> LineDataset:
        return LineDataset(epoch_size=epoch_size)

    @staticmethod
    def get_test_ds(granularity_degrees: float) -> LineDataset:
        angles_degrees = []
        crt_angle = 0.0
        while crt_angle < 180.0:
            angles_degrees.append(crt_angle)
            crt_angle += granularity_degrees

        print("test_ds angles", angles_degrees)
        angles_01 = [x / 180 for x in angles_degrees]
        return LineDataset(angles_01=angles_01)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, float]]:
        if self.angles_01 is not None:
            print("Returning existing angle list")
            angles_01 = self.angles_01
        else:
            print("Creating new angle list")
            angles_01 = [np.random.uniform(0, 1) for _ in range(self.epoch_size)]

        return (
            (generate_image(angle_01 * 180), angle_01)
            for angle_01 in angles_01
        )

    def __len__(self) -> int:
        return self.epoch_size if self.epoch_size else len(self.angles_01)
