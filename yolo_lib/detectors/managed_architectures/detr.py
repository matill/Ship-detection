
from dataclasses import dataclass
from typing import Dict, List, Tuple
from torch import Tensor, nn
import torch
from yolo_lib.data.yolo_tile import YOLOTileStack
from yolo_lib.detectors.managed_architectures.base_detector import BaseDetector
from yolo_lib.data.detection import DetectionGrid
from yolo_lib.models.backbones import BackboneCfg
from yolo_lib.util.check_tensor import check_tensor


@dataclass
class TransformerCfg:
    feature_map_h: int
    feature_map_w: int

    encoder_len: int
    encoder_num_self_attention: int
    # encoder_




class DETR(BaseDetector):
    def __init__(
        self,
        backbone_cfg: BackboneCfg,
        transformer_cfg: TransformerCfg,
    ) -> None:
        pass

    def detect_objects(
        self,
        images: Tensor,
    ) -> DetectionGrid:
        """
        Must be implemented by sub classes.
        Accepts an image as input. Returns a list of Detection objects
        tuples. size, class_probability and mask elements can be None.
        """
        print(f"ERROR: {self.__class__.__name__} does not implement self.detect_objects")

    def compute_loss(self, tiles: YOLOTileStack) -> Tuple[Tensor, Dict[str, float]]:
        print(f"ERROR: {self.__class__.__name__} does not implement self.compute_loss")



class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_self_attention: int,
        dim: int,
    ) -> None:
        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)
        assert isinstance(n_self_attention, int)
        assert isinstance(dim, int)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_self_attention = n_self_attention
        self.dim = dim

        # Self attention weights
        self.weight_q = self._get_weight_xavier((in_channels, n_self_attention, dim), in_channels, n_self_attention * dim)
        self.weight_k = self._get_weight_xavier((in_channels, n_self_attention, dim), in_channels, n_self_attention * dim)
        self.weight_v = self._get_weight_xavier((in_channels, n_self_attention, dim), in_channels, n_self_attention * dim)

        # Concatenated projection weights
        self.weight_out = self._get_weight_xavier((out_channels, n_self_attention, dim), n_self_attention * dim, out_channels)

    def _get_weight_xavier(shape, in_channels, out_channels):
        upper = (torch.sqrt(6.0) / torch.sqrt(in_channels + out_channels))
        rand_unit = torch.rand(*shape)
        rand_xavier = rand_unit * upper * 2 - upper
        return nn.Parameter(rand_xavier)

    def _qkv_multiply(self, weight: Tensor, features: Tensor, batch_size: int, n_vectors: int) -> Tensor:
        # Expand weight and features to common shape, and perform "matmul" operation
        check_tensor(weight, (self.in_channels, self.n_self_attention, self.dim))
        check_tensor(features, (batch_size, n_vectors, self.in_channels))
        weight_r = weight[None, None, :, :, :]
        features_r = features[:, :, :, None, None]
        out_r = weight_r * features_r
        check_tensor(out_r, (batch_size, n_vectors, self.in_channels, self.n_self_attention, self.dim))
        out = out_r.sum(dim=2)
        check_tensor(out, (batch_size, n_vectors, self.n_self_attention, self.dim))
        return out

    def forward(self, features: Tensor) -> Tensor:
        (batch_size, n_vectors, in_channels) = features.shape
        assert in_channels == self.in_channels

        # Compute queries, keys, and values
        q = self._qkv_multiply(self.weight_q, features, batch_size, n_vectors)
        k = self._qkv_multiply(self.weight_k, features, batch_size, n_vectors)
        v = self._qkv_multiply(self.weight_v, features, batch_size, n_vectors)

        # Scaled query-key attention maps
        # First n_vectors dim is q, the second is k. Softmax is applied across the "q" axis
        q_r = q[:, :, None, :, :]
        k_r = k[:, None, :, :, :]
        qk = (q_r * k_r).sum(dim=4)
        qk_scaled = qk / (torch.sqrt(self.dim))
        qk_softmax = qk_scaled.softmax(dim=1)
        check_tensor(qk, (batch_size, n_vectors, n_vectors, self.n_self_attention))
        check_tensor(qk_scaled, (batch_size, n_vectors, n_vectors, self.n_self_attention))
        check_tensor(qk_softmax, (batch_size, n_vectors, n_vectors, self.n_self_attention))

        # V * attention maps
        # v                   (batch_size, n_vectors,            n_self_attention, dim)
        # qk_softmax          (batch_size, n_vectors, n_vectors, n_self_attention)
        # applied_attention   (batch_size, n_vectors,            n_self_attention, dim)
        v_r = v[:, :, None, :, :]
        qk_softmax_r = qk_softmax[:, :, :, :, None]
        applied_attention = (v_r * qk_softmax_r).sum(dim=2)
        check_tensor(applied_attention, (batch_size, n_vectors, self.n_self_attention, self.dim)) 

        # "Project" applied attention into the number of output channels
        # applied_attention  (batch_size,  n_vectors,               n_self_attention, dim)
        # weight_out         (                        out_channels, n_self_attention, dim)
        # out                (batch_size,  n_vectors, out_channels                       )
        applied_attention_r = applied_attention[:, :, None, :, :]
        weight_out_r = self.weight_out[None, None, :, :, :]
        out = (applied_attention_r * weight_out_r).sum(dim=(3, 4))
        check_tensor(out, (batch_size, n_vectors, self.out_channels))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        n_self_attention: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        


