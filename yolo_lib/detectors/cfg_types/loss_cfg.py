

from dataclasses import dataclass

from yolo_lib.detectors.yolo_heads.losses.objectness_loss import ConfidenceUnawareObjectnessLoss
from yolo_lib.models.binary_focal_loss import BinaryFocalLoss, SoftBinaryFocalLoss


@dataclass
class FocalLossCfg:
    neg_weight: float
    pos_weight: float
    gamma: int

    def build(self) -> ConfidenceUnawareObjectnessLoss:
        return ConfidenceUnawareObjectnessLoss(
            BinaryFocalLoss(self.gamma, self.pos_weight, self.neg_weight)
        )

    def build_soft(self) -> SoftBinaryFocalLoss:
        return SoftBinaryFocalLoss(
            self.gamma,
            self.pos_weight,
            self.neg_weight,
        )

