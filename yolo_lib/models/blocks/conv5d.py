import torch
from torch import nn

from yolo_lib.cfg import SAFE_MODE
from yolo_lib.models.activation_functions import ActivationFn
from yolo_lib.models.blocks.bn_conv import BNConv


class Conv5D(nn.Sequential):
    """
    Accepts a [batch, in_channels, h, w] 4d-tensor.
    Outputs a [batch, size_a, size_b, h, w] 5d-tensor
    """
    def __init__(
        self,
        in_channels: int,
        size_a: int,
        size_b: int,
        activation_fn: ActivationFn=None,
        kernel_size: int=3,
    ):
        super().__init__()
        out_channels = size_a * size_b
        self.conv = BNConv(in_channels, out_channels, activation_fn, kernel_size)
        self.size_a = size_a
        self.size_b = size_b

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        (batch_size, in_channels, h, w) = input.shape
        conv: torch.Tensor = self.conv(input)
        reshaped = conv.reshape(batch_size, self.size_a, self.size_b, h, w)
        return reshaped
