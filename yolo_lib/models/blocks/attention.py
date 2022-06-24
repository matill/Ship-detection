from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from yolo_lib.models.activation_functions import Sigmoid
from yolo_lib.models.blocks.bn_conv import BNConv


# AttentionCfg classes
class AttentionCfg:
    def build(self, channels: int) -> AttentionModule:
        raise NotImplementedError


class NoAttentionCfg(AttentionCfg):
    def build(self, channels: int) -> AttentionModule:
        return NoAttentionModule()


class MaskAttentionCfg(AttentionCfg):
    def build(self, channels: int) -> AttentionModule:
        mask = self.__get_mask__(channels)
        assert isinstance(mask, AttentionMask)
        return MaskAttentionModule(mask)

    def __get_mask__(self, channels: int) -> AttentionMask:
        pass


@dataclass
class MultilayerAttentionCfg(MaskAttentionCfg):
    mid_channels: int
    num_hidden: int

    def __get_mask__(self, channels: int) -> MultiLayerAttentionMask:
        return MultiLayerAttentionMask(channels, self.mid_channels, self.num_hidden)


class YOLOv4AttentionCfg(MaskAttentionCfg):
    def __get_mask__(self, channels: int) -> MultiLayerAttentionMask:
        return YOLOv4SAMMask(channels)


# AttentionModule classes
class AttentionModule(nn.Module):
    def __base_attention_impl__(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        out = self.__base_attention_impl__(x)
        assert out.shape == x.shape
        return out


class NoAttentionModule(AttentionModule):
    def __base_attention_impl__(self, x: Tensor) -> Tensor:
        return x


class MaskAttentionModule(AttentionModule):
    def __init__(self, mask: AttentionMask) -> None:
        assert isinstance(mask, AttentionMask)
        super().__init__()
        self.mask = mask

    def __base_attention_impl__(self, x: Tensor) -> Tensor:
        mask = self.mask(x)
        return mask * x


class AttentionMask(nn.Sequential):
    def get_mask(self, x: Tensor) -> Tensor:
        mask = self(x)
        assert mask.shape == x.shape
        return mask


class MultiLayerAttentionMask(AttentionMask):
    def __init__(self, in_channels: int, mid_channels: int, n_hidden: int) -> None:

        # Create layers:
        # First: 1x1 convolution to reduce channel dimensions
        # N layers: 3x3 convolution in small channel dimensions
        # Last: 1x1 convolution to increase channel dimensions
        layers = []
        layers.append(BNConv(in_channels, mid_channels, Sigmoid(), kernel_size=1))
        for _ in range(n_hidden):
            layers.append(BNConv(mid_channels, mid_channels, Sigmoid(), kernel_size=3))

        layers.append(BNConv(mid_channels, in_channels, Sigmoid(), kernel_size=1))
        super().__init__(*layers)


class YOLOv4SAMMask(AttentionMask):
    def __init__(self, channels) -> None:
        super().__init__(BNConv(channels, channels, Sigmoid()))

