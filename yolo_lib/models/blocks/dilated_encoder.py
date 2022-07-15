from __future__ import annotations
from torch import nn
from dataclasses import dataclass
from typing import List
from yolo_lib.cfg import SAFE_MODE


@dataclass
class EncoderConfig:
    out_channels: int
    mid_channels: int # 64
    dilations: List[int] # [2, 4, 6, 8]

    def build(self, in_channels: int) -> DilatedEncoder:
        assert isinstance(in_channels, int)
        return DilatedEncoder(
            in_channels,
            self.out_channels,
            self.mid_channels,
            self.dilations,
        )

    @staticmethod
    def default() -> EncoderConfig:
        return EncoderConfig(512, 128, [2, 4, 6, 8])


class DilatedEncoder(nn.Sequential):
    def __init__(
        self,
        in_channels,
        encoder_n_channels,
        block_mid_channels,
        dilations
    ):
        super().__init__(
            # Projector
            Projector(in_channels, encoder_n_channels),

            # Dilated residual blocks
            DilatedResBlockSequence(encoder_n_channels, block_mid_channels, dilations),
        )


class BaseResidualBlock(nn.Module):
    def __init__(self, in_channels, wrapped_block):
        super().__init__()
        self.in_channels = in_channels
        self.wrapped_block = wrapped_block

    def clsname(self):
        return self.__class__.__name__

    def forward(self, x0):
        # Check input shape
        _batch_size, in_channels, _h, _w = x0.shape
        if SAFE_MODE:
            assert in_channels == self.in_channels, \
                    f"{self.clsname()} got input shape {x0.shape} with {in_channels} channels. " + \
                    f"Expected {self.in_channels} channels."

        # Get output of wrapped block and check output shape
        wrapped_block_output = self.wrapped_block(x0)
        if SAFE_MODE:
            assert wrapped_block_output.shape == x0.shape, \
                    f"{self.clsname()}'s wrapped block's output had shape {wrapped_block_output.shape} " + \
                    f"which is different from it's input shape {x0.shape}."

        return x0 + wrapped_block_output


class DilatedResBlock(BaseResidualBlock):
    NUM_PARAMETERS = sum([
        2, 2, 0, 2, 2, 0, 2, 2, 0
    ])

    def __init__(self, encoder_n_channels, mid_channels, dilation):
        super().__init__(encoder_n_channels, nn.Sequential(
            # First (1x1) convolution into smaller number of channels
            nn.Conv2d(encoder_n_channels, mid_channels, kernel_size=1, bias=True), # Bias=False makes more sense?
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),

            # Second (dilated 3x3) convoltuion
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, dilation=dilation, padding="same", bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),

            # Second (1x1) convolution to increase channel number
            nn.Conv2d(mid_channels, encoder_n_channels, kernel_size=1, bias=True), # Bias=False makes more sense?
            nn.BatchNorm2d(encoder_n_channels),
            nn.ReLU(),
        ))


class DilatedResBlockSequence(nn.Sequential):
    def __init__(self, encoder_n_channels, mid_channels, dilations):
        super().__init__(*[
            DilatedResBlock(encoder_n_channels, mid_channels, dilation)
            for dilation in dilations
        ])


class Projector(nn.Sequential):
    NUM_PARAMETERS = sum([
        2, # Conv-1
        2, # BatchNorm-1
        2, # Conv-2
        2  # BatchNorm-2 
    ])

    def __init__(self, in_channels, encoder_n_channels):
        bias=True
        super().__init__(
            nn.Conv2d(in_channels, encoder_n_channels, 1, padding=0, bias=bias),
            nn.BatchNorm2d(encoder_n_channels),
            nn.Conv2d(encoder_n_channels, encoder_n_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(encoder_n_channels),
        )
