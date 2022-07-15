from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from torch import Tensor
import torch
from yolo_lib.data.dataclasses import DetectionBlock, YOLOTileStack
from yolo_lib.data.detection import DetectionGrid
from yolo_lib.detectors.yolo_heads.yolo_head import YOLOHead, YOLOHeadCfg
from yolo_lib.models.blocks.dilated_encoder import EncoderConfig
from yolo_lib.models.blocks.attention import AttentionCfg
from yolo_lib.detectors.managed_architectures.base_detector import BaseDetector, DetectorCfg
from yolo_lib.models.backbones import BackBone, BackboneCfg


@dataclass
class AttentionYOLOFCfg(DetectorCfg):
    head_cfg: YOLOHeadCfg
    backbone_cfg: BackboneCfg
    dileated_encoder_cfg: EncoderConfig
    attention_cfg: AttentionCfg

    def build(self) -> AttentionYOLOF:
        return AttentionYOLOF(
            self.backbone_cfg.build(),
            self.dileated_encoder_cfg,
            self.attention_cfg,
            self.head_cfg.build(self.dileated_encoder_cfg.out_channels)
        )


class AttentionYOLOF(BaseDetector):
    def __init__(self,
        backbone: BackBone,
        dilated_encoder_cfg: EncoderConfig,
        attention_cfg: AttentionCfg,
        yolo_head: YOLOHead,
    ):
        assert isinstance(backbone, BackBone)
        assert isinstance(dilated_encoder_cfg, EncoderConfig)
        assert isinstance(attention_cfg, AttentionCfg)
        assert isinstance(yolo_head, YOLOHead)
        super().__init__()
        self.downsample_factor = torch.tensor(backbone.downsample_factor, requires_grad=False)
        self.backbone = backbone
        self.dilated_encoder = dilated_encoder_cfg.build(backbone.out_channels)
        self.attention = attention_cfg.build(dilated_encoder_cfg.out_channels)
        self.yolo_head = yolo_head

    def detect_objects(
        self,
        images: torch.Tensor,
    ) -> DetectionGrid:
        features = self.backbone(images)
        dilated_encoding = self.dilated_encoder(features)
        attention = self.attention(dilated_encoding)
        pre_activation = self.yolo_head.get_pre_activation(attention)
        post_activation = self.yolo_head.get_post_activation(pre_activation)
        return self.yolo_head.detect_objects(
            post_activation,
            self.downsample_factor,
        )

    def compute_loss(self, tiles: YOLOTileStack) -> Tuple[torch.Tensor, Dict[str, float]]:
        features = self.backbone(tiles.images)
        dilated_encoding = self.dilated_encoder(features)
        attention = self.attention(dilated_encoding)
        pre_activation = self.yolo_head.get_pre_activation(attention)
        post_activation = self.yolo_head.get_post_activation(pre_activation)
        return self.yolo_head.compute_loss(
            pre_activation,
            post_activation,
            tiles,
            self.downsample_factor
        )
