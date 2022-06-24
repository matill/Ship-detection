from __future__ import annotations
import torch
from yolo_lib.detectors.yolo_heads.annotation_encoding import PointAnnotationEncoding
from yolo_lib.detectors.yolo_heads.label_assignment.label_assignment import LabelAssignment


class LabelAssigner:
    def assign(
        self,
        match_loss: torch.Tensor,
        yx_annotation_encoding: PointAnnotationEncoding,
    ) -> LabelAssignment:
        raise NotImplementedError

