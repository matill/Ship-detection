import torch
from torch import nn, Tensor
from yolo_lib.data.annotation import AnnotationBlock
from yolo_lib.detectors.yolo_heads.label_assignment.label_assignment import LabelAssignment
from yolo_lib.detectors.yolo_heads.annotation_encoding import SinCosAnnotationEncoding
from yolo_lib.util import check_tensor


class ADVLoss(nn.Module):
    def __init__(self, eccentricity: float=3.0) -> None:
        # NOTE: Might be using the word "eccentricity" a bit wrong here.
        assert 1.0 <= eccentricity, f"Eccentricity of elliptic loss function must be greater than one"
        super().__init__()
        self.eccentricity = eccentricity
        self.lambda_1 = 2 / (1 + eccentricity)
        self.lambda_2 = 2 - self.lambda_1

    def forward(
        self,
        post_activation_sincos: Tensor,
        # sincos_annotation_encoding: SinCosAnnotationEncoding,
        annotation_block: AnnotationBlock,
        assignment: LabelAssignment,
    ) -> Tensor:
        assert isinstance(post_activation_sincos, Tensor)
        assert isinstance(annotation_block, AnnotationBlock)
        assert isinstance(assignment, LabelAssignment)

        # Get assignment/matching-subset where all targets have valid orientation
        valid_target_rotation_bitmap = annotation_block.has_rotation[assignment.object_idxs]
        check_tensor(valid_target_rotation_bitmap, (assignment.num_assignments, ), torch.bool)
        reduced_assignment = assignment.extract_bitmap(valid_target_rotation_bitmap)
        num_assignments = reduced_assignment.num_assignments

        # sin and cos of true rotations
        true_rotation_01 = annotation_block.rotation[reduced_assignment.object_idxs]
        true_rotation_rad = true_rotation_01 * (2 * 3.14159)
        true_sin = true_rotation_rad.sin()
        true_cos = true_rotation_rad.cos()

        # Predicted direction vectors.
        (img_idxs, head_idxs, grid_y_idxs, grid_x_idxs) = reduced_assignment.get_grid_idx_vectors()
        predicted_adv = post_activation_sincos[img_idxs, head_idxs, :, grid_y_idxs, grid_x_idxs]
        predicted_adv_1 = predicted_adv[:, 0]
        predicted_adv_2 = predicted_adv[:, 1]
        check_tensor(predicted_adv, (num_assignments, 2))
        check_tensor(predicted_adv_1, (num_assignments, ))
        check_tensor(predicted_adv_2, (num_assignments, ))

        # Projected predictions
        projection_1 = predicted_adv_1 * true_sin + predicted_adv_2 * true_cos
        projection_2 = predicted_adv_1 * true_cos - predicted_adv_2 * true_sin

        # Loss:
        # projection_1 should be 1
        # projection_2 should be 0
        loss_1 = (projection_1 - 1).square()
        loss_2 = (projection_2 - 0).square()
        loss = loss_1.sum() * self.lambda_1 + loss_2.sum() * self.lambda_2
        assert loss_1.shape == (num_assignments, )
        assert loss_2.shape == (num_assignments, )
        return loss

