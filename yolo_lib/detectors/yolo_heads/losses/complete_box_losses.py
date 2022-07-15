import torch
from yolo_lib.detectors.yolo_heads.annotation_encoding import PointAnnotationEncoding, SizeAnnotationEncoding
from yolo_lib.detectors.yolo_heads.label_assignment.label_assignment import LabelAssignment


class BoxLoss(torch.nn.Module):
    """Base class for bounding box losses"""

    def forward(
        self,
        post_activation_yx: torch.Tensor,
        post_activation_hw: torch.Tensor,
        assignment: LabelAssignment,
        yx_annotation_encoding: PointAnnotationEncoding,
        hw_annotation_encoding: SizeAnnotationEncoding,
    ) -> torch.Tensor:
        print(f"ERROR: {self.__class__.__name__}.forward() not implemented")

