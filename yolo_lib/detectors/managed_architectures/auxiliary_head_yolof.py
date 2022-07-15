from typing import Dict, List, Optional, Tuple
from torch import Tensor
import torch
from yolo_lib.data.dataclasses import DetectionBlock, YOLOTileStack
from yolo_lib.data.detection import DetectionGrid
from yolo_lib.models.blocks.attention import AttentionCfg, AttentionModule
from yolo_lib.detectors.yolo_heads.yolo_head import YOLOHead
from yolo_lib.models.blocks.dilated_encoder import EncoderConfig
from yolo_lib.detectors.managed_architectures.base_detector import BaseDetector
from yolo_lib.models.backbones import BackBone

# TODO: Check if we should use intermediate conv here

class AuxiliaryHeadYOLOF(BaseDetector):
    def __init__(self,
        backbone: BackBone,
        dilated_encoder_cfg: EncoderConfig,
        attention_cfg: AttentionCfg,
        main_head: YOLOHead,
        auxiliary_head: YOLOHead,
        auxiliary_loss_weight: float,
    ):
        assert isinstance(backbone, BackBone)
        assert isinstance(dilated_encoder_cfg, EncoderConfig)
        assert isinstance(attention_cfg, AttentionCfg)
        assert isinstance(main_head, YOLOHead)
        assert isinstance(auxiliary_head, YOLOHead)
        assert isinstance(auxiliary_loss_weight, float) and auxiliary_loss_weight > 0
        super().__init__()
        self.downsample_factor = torch.tensor(backbone.downsample_factor, requires_grad=False)
        self.backbone = backbone
        self.dilated_encoder = dilated_encoder_cfg.build(backbone.out_channels)
        self.main_head_attention = attention_cfg.build(dilated_encoder_cfg.out_channels)
        self.auxiliary_head_attention = attention_cfg.build(dilated_encoder_cfg.out_channels)
        self.main_head = main_head
        self.auxiliary_head = auxiliary_head
        self.auxiliary_loss_weight = auxiliary_loss_weight

    def detect_objects(
        self,
        images: torch.Tensor,
    ) -> DetectionGrid:
        features = self.backbone(images)
        dilated_encoding = self.dilated_encoder(features)
        attention = self.main_head_attention(dilated_encoding)
        pre_activation = self.main_head.get_pre_activation(attention)
        post_activation = self.main_head.get_post_activation(pre_activation)
        return self.main_head.detect_objects(
            post_activation,
            self.downsample_factor,
        )

    def compute_loss(self, tiles: YOLOTileStack) -> Tuple[torch.Tensor, Dict[str, float]]:
        features = self.backbone(tiles.images)
        dilated_encoding = self.dilated_encoder(features)

        # Main head loss
        main_head_loss, main_head_loss_subterms = self.loss_helper(
            tiles,
            dilated_encoding,
            self.main_head_attention,
            self.main_head,
        )

        # Auxiliary head loss
        auxiliary_head_loss, auxiliary_head_loss_subterms = self.loss_helper(
            tiles,
            dilated_encoding,
            self.auxiliary_head_attention,
            self.auxiliary_head,
        )

        # Weighted sum of losses
        total_loss = main_head_loss + self.auxiliary_loss_weight * auxiliary_head_loss

        # Stack subterm dictionaries
        total_subterms = {}
        for key, value in main_head_loss_subterms.items():
            total_subterms[f"main-head-{key}"] = float(value)

        for key, value in auxiliary_head_loss_subterms.items():
            total_subterms[f"auxiliary-head-{key}"] = float(value)

        # Return result
        return (total_loss, total_subterms)

    def loss_helper(
        self,
        tiles: YOLOTileStack,
        dilated_encoding: torch.Tensor,
        attention_module: AttentionModule,
        head: YOLOHead,
    ) -> torch.Tensor:
        attention = attention_module(dilated_encoding)
        pre_activation = head.get_pre_activation(attention)
        post_activation = head.get_post_activation(pre_activation)
        loss_tuple = head.compute_loss(
            pre_activation,
            post_activation,
            tiles,
            self.downsample_factor, 
        )

        loss, loss_subterms = loss_tuple
        assert isinstance(loss, torch.Tensor) and loss.shape == ()
        assert isinstance(loss_subterms, dict)
        return loss_tuple


