import torch
import torch.nn as nn
from yolo_lib.detectors.yolo_heads.label_assignment.label_assignment import LabelAssignment
from yolo_lib.detectors.yolo_heads.annotation_encoding import SinCosAnnotationEncoding
from yolo_lib.util import check_tensor


class SinCosLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        post_activation_sincos: torch.Tensor,
        sincos_annotation_encoding: SinCosAnnotationEncoding,
        assignment: LabelAssignment,
    ) -> torch.Tensor:
        assert isinstance(post_activation_sincos, torch.Tensor)
        assert isinstance(sincos_annotation_encoding, SinCosAnnotationEncoding)
        assert isinstance(assignment, LabelAssignment)

        # Get assignment/matching-subset where all targets have valid orientation
        valid_target_sincos_bitmap = sincos_annotation_encoding.has_rotation[assignment.object_idxs]
        valid_target_sincos_bitmap_bool = valid_target_sincos_bitmap.type(torch.bool)
        reduced_assignment = assignment.extract_bitmap(valid_target_sincos_bitmap_bool)

        # Index into target sincos
        true_sincos = sincos_annotation_encoding.sincos[reduced_assignment.object_idxs]
        check_tensor(true_sincos, (reduced_assignment.num_assignments, 2))

        # Index into prediced sincos
        (img_idxs, head_idxs, grid_y_idxs, grid_x_idxs) = reduced_assignment.get_grid_idx_vectors()
        predicted_sincos = post_activation_sincos[img_idxs, head_idxs, :, grid_y_idxs, grid_x_idxs]
        check_tensor(predicted_sincos, (reduced_assignment.num_assignments, 2))

        # Compute distances
        distances_sqrd = (true_sincos - predicted_sincos).square()
        check_tensor(distances_sqrd, (reduced_assignment.num_assignments, 2))
        return distances_sqrd.sum()

