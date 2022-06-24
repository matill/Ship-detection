from __future__ import annotations
from dataclasses import dataclass
from typing import List
from yolo_lib.models.blocks.dilated_encoder import DilatedEncoder


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
