

import torch
from yolo_lib.detectors.yolo_heads.label_assigner.label_assigner import LabelAssignment
from yolo_lib.detectors.yolo_heads.annotation_encoding import PointAnnotationEncoding, SizeAnnotationEncoding
from yolo_lib.util import check_tensor
from .complete_box_losses import BoxLoss
from yolo_lib.iou import get_ciou_grid_loss, get_diou_grid_loss


class CIoUBoxLoss(BoxLoss):
    """
    Computes CIoU as a "final" output loss.
    For annotations that are missing valid height and width, L2 loss on yx positions is used.
    """
    def __init__(self, stable: bool = True) -> None:
        super().__init__()
        self.stable = stable

    def forward(
        self,
        post_activation_yx: torch.Tensor,
        post_activation_hw: torch.Tensor,
        assignment: LabelAssignment,
        yx_annotation_encoding: PointAnnotationEncoding,
        hw_annotation_encoding: SizeAnnotationEncoding,
    ) -> torch.Tensor:
        assert isinstance(post_activation_yx, torch.Tensor)
        assert isinstance(post_activation_hw, torch.Tensor)
        assert isinstance(assignment, LabelAssignment)
        assert isinstance(yx_annotation_encoding, PointAnnotationEncoding)
        assert isinstance(hw_annotation_encoding, SizeAnnotationEncoding)
        assert hw_annotation_encoding.has_size_hw.all()

        # Get assignment/matching-subset where all targets have valid height and width, and where they don't
        valid_hw_bitmap = hw_annotation_encoding.has_size_hw[assignment.object_idxs]
        check_tensor(valid_hw_bitmap, (assignment.num_assignments, ), torch.int64)
        valid_hw_bitmap = valid_hw_bitmap.type(torch.bool)
        valid_hw_assignment = assignment.extract_bitmap(valid_hw_bitmap)
        invalid_hw_assignment = assignment.extract_bitmap(~valid_hw_bitmap)

        # Get target yx and hw values for valid_hw subset
        true_hw = hw_annotation_encoding.size_hw[valid_hw_assignment.object_idxs]
        true_yx = yx_annotation_encoding.center_yx[valid_hw_assignment.object_idxs]

        # Get predicted yx and hw values for valid_hw subset
        (img_idxs, head_idxs, grid_y_idxs, grid_x_idxs) = valid_hw_assignment.get_grid_idx_vectors()
        predicted_hw = post_activation_hw[img_idxs, head_idxs, :, grid_y_idxs, grid_x_idxs]
        predicted_yx = post_activation_yx[img_idxs, head_idxs, :, grid_y_idxs, grid_x_idxs]

        # Get CIoU loss for the valid_hw subset
        ciou_loss = get_ciou_grid_loss(
            predicted_yx[:, None, :],
            predicted_hw[:, None, :],
            true_yx[:, None, :],
            true_hw[:, None, :],
            valid_hw_assignment.num_assignments,
            1,
            self.stable,
        )

        check_tensor(ciou_loss, (valid_hw_assignment.num_assignments, 1))

        # Compute L2 difference between center points matches without ground-truth height/width
        # distances = matched_predicted_yx - matched_true_yx.
        # Multiply by 1/(sqrt(2)) to contain in (0,1) range
        # center_yx_loss = get_center_yx_loss_(
        #     post_activation_yx, yx_annotation_encoding, invalid_hw_assignment
        # )

        # Return complete loss
        # loss = ciou_loss.sum() + center_yx_loss * DEVICE_INV_SQRT2
        loss = ciou_loss.sum()
        return loss


class DIoUBoxLoss(BoxLoss):
    """Computes DIoU as a "final" output loss, assuming all objects have known height and width"""

    def __init__(self, do_detach: bool) -> None:
        super().__init__()
        self.do_detach = do_detach

    def forward(
        self,
        post_activation_yx: torch.Tensor,
        post_activation_hw: torch.Tensor,
        assignment: LabelAssignment,
        yx_annotation_encoding: PointAnnotationEncoding,
        hw_annotation_encoding: SizeAnnotationEncoding,
    ) -> torch.Tensor:
        assert isinstance(post_activation_yx, torch.Tensor)
        assert isinstance(post_activation_hw, torch.Tensor)
        assert isinstance(assignment, LabelAssignment)
        assert isinstance(yx_annotation_encoding, PointAnnotationEncoding)
        assert isinstance(hw_annotation_encoding, SizeAnnotationEncoding)
        assert hw_annotation_encoding.has_size_hw.all()

        # Get assignment/matching-subset where all targets have valid height and width, and where they don't
        valid_hw_bitmap = hw_annotation_encoding.has_size_hw[assignment.object_idxs]
        check_tensor(valid_hw_bitmap, (assignment.num_assignments, ), torch.int64)
        valid_hw_bitmap = valid_hw_bitmap.type(torch.bool)
        valid_hw_assignment = assignment.extract_bitmap(valid_hw_bitmap)

        # Get target yx and hw values for valid_hw subset
        true_hw = hw_annotation_encoding.size_hw[valid_hw_assignment.object_idxs]
        true_yx = yx_annotation_encoding.center_yx[valid_hw_assignment.object_idxs]

        # Get predicted yx and hw values for valid_hw subset
        (img_idxs, head_idxs, grid_y_idxs, grid_x_idxs) = valid_hw_assignment.get_grid_idx_vectors()
        predicted_hw = post_activation_hw[img_idxs, head_idxs, :, grid_y_idxs, grid_x_idxs]
        predicted_yx = post_activation_yx[img_idxs, head_idxs, :, grid_y_idxs, grid_x_idxs]

        # Get DIoU loss for the valid_hw subset
        diou_loss = get_diou_grid_loss(
            predicted_yx[:, None, :],
            predicted_hw[:, None, :],
            true_yx[:, None, :],
            true_hw[:, None, :],
            valid_hw_assignment.num_assignments,
            1,
            self.do_detach,
        )

        check_tensor(diou_loss, (valid_hw_assignment.num_assignments, 1))

        # Return complete loss
        loss = diou_loss.sum()
        return loss

