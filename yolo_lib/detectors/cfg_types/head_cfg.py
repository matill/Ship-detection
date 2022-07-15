from dataclasses import dataclass
import torch
from yolo_lib.detectors.cfg_types.loss_cfg import FocalLossCfg
from yolo_lib.detectors.yolo_heads.label_assigner.hungarian_matching import HungarianMatching
from yolo_lib.detectors.yolo_heads.heads.managed_yolo_head import ManagedYOLOHead
from yolo_lib.detectors.yolo_heads.overlapping_cell_yolo_head import OverlappingCellYOLOHead, PointPotoMatchloss, PotoMatchloss, PotoMatchlossCfg
from yolo_lib.detectors.yolo_heads.heads.sincos_yolo_head import SinCosYOLOHead
from yolo_lib.detectors.yolo_heads.losses.center_yx_losses import CenterYXSmoothL1
from yolo_lib.detectors.yolo_heads.losses.siou_box_loss import SIoUBoxLoss, CompleteCenteredIoUMatchLoss
from yolo_lib.detectors.yolo_heads.losses.sincos_losses import SinCosLoss



focal_loss_standard = FocalLossCfg(neg_weight=0.3, pos_weight=1.0, gamma=2)


class YOLOHeadCfg:
    def build(self) -> ManagedYOLOHead:
        pass


@dataclass
class LocalMatchingYOLOHeadCfg(YOLOHeadCfg):
    in_channels: int
    num_anchors: int
    anchor_priors: torch.Tensor

    matchloss_objectness_weight: float
    matchloss_yx_weight: float
    matchloss_hw_weight: float
    matchloss_sincos_weight: float

    loss_objectness_weight: float
    loss_yx_weight: float
    loss_hw_weight: float
    loss_sincos_weight: float

    focal_loss: FocalLossCfg = focal_loss_standard
    allow_180: bool = True

    def get_matching_strategy(self):
        if self.matching_strategy is None:
            return HungarianMatching(self.num_anchors)
        else:
            return self.matching_strategy

    def build(self) -> SinCosYOLOHead:
        # Matchloss box (yx + hw) weighting
        matchloss_box_weight = self.matchloss_yx_weight + self.matchloss_hw_weight
        matchloss_yx_weight = self.matchloss_yx_weight / matchloss_box_weight
        matchloss_hw_weight = self.matchloss_hw_weight / matchloss_box_weight

        # Loss box (yx + hw) weighting
        loss_box_weight = self.loss_yx_weight + self.loss_hw_weight
        loss_yx_weight = self.loss_yx_weight / loss_box_weight
        loss_hw_weight = self.loss_hw_weight / loss_box_weight
        return SinCosYOLOHead(
            self.in_channels,
            self.num_anchors,
            self.anchor_priors,
            SinCosMatchLoss(True, False, self.num_anchors),
            SinCosLoss(True, False),
            CompleteCenteredIoUMatchLoss(self.num_anchors, matchloss_yx_weight, matchloss_hw_weight, CenterYXSmoothL1()),
            SIoUBoxLoss(loss_yx_weight, loss_hw_weight, CenterYXSmoothL1()),
            self.focal_loss.build(),
            self.get_matching_strategy(),
            self.matchloss_objectness_weight,
            matchloss_box_weight,
            self.matchloss_sincos_weight,
            self.loss_objectness_weight,
            loss_box_weight,
            self.loss_sincos_weight
        )


@dataclass
class OverlappingCellYOLOHeadCfg(YOLOHeadCfg):
    in_channels: int
    num_anchors: int
    anchor_priors: torch.Tensor

    yx_multiplier: float
    yx_match_threshold: float

    matchloss_cfg: PotoMatchlossCfg

    loss_objectness_weight: float
    loss_yx_weight: float
    loss_hw_weight: float
    loss_sincos_weight: float

    focal_loss: FocalLossCfg = focal_loss_standard
    flat_prior: bool = False

    def build(self) -> OverlappingCellYOLOHead:
        # Loss box (yx + hw) weighting
        loss_box_weight = self.loss_yx_weight + self.loss_hw_weight
        loss_yx_weight = self.loss_yx_weight / loss_box_weight
        loss_hw_weight = self.loss_hw_weight / loss_box_weight
        return OverlappingCellYOLOHead(
            self.in_channels,
            self.num_anchors,
            self.anchor_priors,
            self.yx_multiplier,
            self.yx_match_threshold,
            self.matchloss_cfg,
            SinCosLoss(True, False),
            SIoUBoxLoss(loss_yx_weight, loss_hw_weight, CenterYXSmoothL1()),
            self.focal_loss.build(),
            self.loss_objectness_weight,
            loss_box_weight,
            self.loss_sincos_weight,
            self.flat_prior,
        )

