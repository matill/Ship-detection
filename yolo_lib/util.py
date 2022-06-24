from typing import Tuple
import torch

from yolo_lib.cfg import SAFE_MODE


def check_tensor(tensor: torch.Tensor, expected_shape: Tuple, expected_dtype: torch.dtype=None):
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    if expected_dtype:
        assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}"

