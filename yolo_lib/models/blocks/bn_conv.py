from typing import Optional
import torch
from torch import nn

from yolo_lib.models.activation_functions import ActivationFn


class BNConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: ActivationFn = None,
        kernel_size: int = 3,
        stride: int=1,
        padding="same",
        bottleneck_channels: Optional[int]=None,
    ):
        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)
        assert activation is None or isinstance(activation, ActivationFn)
        assert isinstance(kernel_size, int)
        assert isinstance(stride, int)
        super().__init__()
        if bottleneck_channels is None:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        else:
            assert isinstance(bottleneck_channels, int) and 0 < bottleneck_channels < min(in_channels, out_channels)
            mid_channels = bottleneck_channels
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, bias=False),
                nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x
