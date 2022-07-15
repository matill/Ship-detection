import torch
from yolo_lib.detectors.yolo_heads.label_assignment.label_assignment import LabelAssignment
from yolo_lib.detectors.yolo_heads.annotation_encoding import PointAnnotationEncoding


class CenterYXLoss(torch.nn.Module):
    def __core_loss_implementation__(self, n: int, m: int, diffs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def core_loss_implementation(self, n: int, m: int, diffs: torch.Tensor) -> torch.Tensor:
        assert diffs.shape == (n, m, 2)
        loss = self.__core_loss_implementation__(n, m, diffs)
        assert loss.shape == (n, m)
        return loss

    def get_local_cell_matchloss(
        self,
        yx_annotation_encoding: PointAnnotationEncoding,
        post_activation_yx: torch.Tensor,
        num_heads: int,
    ) -> torch.Tensor:
        assert isinstance(yx_annotation_encoding, PointAnnotationEncoding)
        assert isinstance(post_activation_yx, torch.Tensor)
        assert isinstance(num_heads, int)
        num_objects = yx_annotation_encoding.num_annotations

        # For each object in the sequence of (y_idx, x_idx, img_idx) index tuples, get
        # the "num_heads" candidate predicted anchors.
        # Produces a [num_objects, num_heads, {y,x}] shaped output
        (y_idxs, x_idxs, img_idxs) = yx_annotation_encoding.get_annotation_idxs()
        predicted_yx = post_activation_yx[img_idxs, :, :, y_idxs, x_idxs]
        assert predicted_yx.shape == (num_objects, num_heads, 2)

        # Similar to in_cell_pos_predictions, produce an yx_annotation_encoding_repeated with the
        # shape. yx_annotation_encoding has a [num_objects, 2] shape. Add an axis in the middle
        # which repeats the yx_annotation_encoding matrix "num_heads" times
        target_yx = yx_annotation_encoding.center_yx[:, None, :]
        assert target_yx.shape == (num_objects, 1, 2)

        # Compute element-wise difference, and compute loss
        diffs = predicted_yx - target_yx
        loss = self.core_loss_implementation(num_objects, num_heads, diffs)
        return loss

    def get_loss(
        self,
        post_activation_yx: torch.Tensor,
        yx_annotation_encoding: PointAnnotationEncoding,
        assignment: LabelAssignment
    ) -> torch.Tensor:
        num_assignments = assignment.num_assignments

        # Index grids to get predicted and true yx values
        (img_idxs, head_idxs, grid_y_idxs, grid_x_idxs) = assignment.get_grid_idx_vectors()
        predicted_yx = post_activation_yx[img_idxs, head_idxs, :, grid_y_idxs, grid_x_idxs]
        true_yx = yx_annotation_encoding.center_yx[assignment.object_idxs]
        assert predicted_yx.shape == (num_assignments, 2)
        assert true_yx.shape == (num_assignments, 2)

        # Compute the summed l2 loss of all regressors.
        # assigned_predicted_offsets[i] corresponds to matched_true_yx[i]
        diffs = (predicted_yx - true_yx)[:, None, :]
        loss = self.core_loss_implementation(num_assignments, 1, diffs)
        return loss.sum()


class CenterYXSmoothL1(CenterYXLoss):
    def __core_loss_implementation__(self, n: int, m: int, diffs: torch.Tensor) -> torch.Tensor:
        diffs_abs = diffs.abs()
        diffs_abs_below_1 = diffs_abs < 1
        diffs_sqrd = diffs ** 2
        smooth_l1 = torch.where(diffs_abs_below_1, diffs_sqrd, diffs_abs)
        loss = smooth_l1[:, :, 0] + smooth_l1[:, :, 1]
        assert diffs_abs.shape == (n, m, 2)
        assert diffs_abs_below_1.shape == (n, m, 2)
        assert diffs_abs.shape == (n, m, 2)
        assert smooth_l1.shape == (n, m, 2)
        return loss


class CenterYXSquaredError(CenterYXLoss):
    def __core_loss_implementation__(self, n: int, m: int, diffs: torch.Tensor) -> torch.Tensor:
        diffs_sqrd = (diffs ** 2)
        loss = diffs_sqrd[:, :, 0] + diffs_sqrd[:, :, 1]
        assert diffs_sqrd.shape == (n, m, 2)
        return loss


class CenterYXL1(CenterYXLoss):
    def __core_loss_implementation__(self, n: int, m: int, diffs: torch.Tensor) -> torch.Tensor:
        diffs_abs = diffs.abs()
        loss = diffs_abs[:, :, 0] + diffs_abs[:, :, 1]
        assert diffs_abs.shape == (n, m, 2)
        return loss

