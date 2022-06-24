import torch
from yolo_lib.cfg import SAFE_MODE
from yolo_lib.detectors.yolo_heads.label_assigner.label_assigner import LabelAssignment
from yolo_lib.detectors.yolo_heads.annotation_encoding import PointAnnotationEncoding, SizeAnnotationEncoding
from yolo_lib.detectors.yolo_heads.losses.center_yx_losses import CenterYXLoss
from yolo_lib.util import check_tensor
from .complete_box_losses import BoxLoss
from yolo_lib.iou import get_centered_iou


class SIoUBoxLoss(BoxLoss):
    """
    Computes "SIoU" for rows with target HW, and computes L2
    center distance for ALL rows (with AND without height and width)
    """

    def __init__(self, yx_weight: float, hw_weight: float, yx_loss_fn: CenterYXLoss) -> None:
        super().__init__()
        assert isinstance(yx_weight, float)
        assert isinstance(hw_weight, float)
        assert isinstance(yx_loss_fn, CenterYXLoss)
        self.yx_weight = yx_weight
        self.hw_weight = hw_weight
        self.yx_loss_fn = yx_loss_fn

    def _get_centered_iou_loss(
        self,
        post_activation_hw: torch.Tensor,
        assignment: LabelAssignment,
        hw_annotation_encoding: SizeAnnotationEncoding,
    ) -> torch.Tensor:
        """Computes only "Centered IoU", ignoring all rows with missing target-HW"""

        # Get assignment/matching-subset where all targets have valid height and width, and where they don't
        valid_hw_bitmap = hw_annotation_encoding.has_size_hw[assignment.object_idxs]
        check_tensor(valid_hw_bitmap, (assignment.num_assignments, ), torch.int64)
        valid_hw_bitmap = valid_hw_bitmap.type(torch.bool)
        valid_hw_assignment = assignment.extract_bitmap(valid_hw_bitmap)

        # Get target yx and hw values for valid_hw subset
        true_hw = hw_annotation_encoding.size_hw[valid_hw_assignment.object_idxs]

        # Get predicted yx and hw values for valid_hw subset
        (img_idxs, head_idxs, grid_y_idxs, grid_x_idxs) = valid_hw_assignment.get_grid_idx_vectors()
        predicted_hw = post_activation_hw[img_idxs, head_idxs, :, grid_y_idxs, grid_x_idxs]

        # Get centered IoU
        centered_iou = get_centered_iou(
            predicted_hw[:, None, :],
            true_hw[:, None, :],
            valid_hw_assignment.num_assignments,
            1
        )

        if centered_iou.isnan().any():
            isnan = centered_iou.isnan()
            print("centered_iou isnan indexes", isnan.nonzero())
            print("predicted_hw[isnan]", predicted_hw[:, None, :][isnan])
            print("true_hw[isnan]", true_hw[:, None, :][isnan])
            assert False


        # Get centered IoU-loss (1 - IoU)
        centered_iou_loss = 1 - centered_iou

        if SAFE_MODE:
            assert centered_iou_loss.shape == (valid_hw_assignment.num_assignments, 1)

        # Return complete loss
        return centered_iou_loss.sum()

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

        centered_iou_loss = self._get_centered_iou_loss(post_activation_hw, assignment, hw_annotation_encoding)
        yx_center_distance = self.yx_loss_fn.get_loss(post_activation_yx, yx_annotation_encoding, assignment)
        assert centered_iou_loss.shape == ()
        assert yx_center_distance.shape == ()
        return centered_iou_loss * self.hw_weight + yx_center_distance * self.yx_weight

