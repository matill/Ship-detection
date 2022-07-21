from typing import Optional
import torch
from torch import Tensor
from yolo_lib.cfg import SAFE_MODE
from yolo_lib.data.annotation import AnnotationBlock
from yolo_lib.detectors.yolo_heads.label_assignment.similarity_metrics.similarity_metric_base import SimilarityMetric
from yolo_lib.detectors.yolo_heads.output_vector_format import YOLO_Y, YOLO_X, YOLO_H, YOLO_W, YOLO_O, YOLO_SIN, YOLO_COS, YOLO_YX, YOLO_HW, YOLO_SINCOS
from yolo_lib.util.check_tensor import check_tensor


class PointPotoMatchloss(SimilarityMetric):
    def __init__(
        self,
        num_anchors: int,
        matchloss_objectness_weight: float,
        matchloss_yx_weight: float,
        max_distance: float,
        matchloss_hw_weight: Optional[float],
    ) -> None:
        super().__init__()
        self.use_centered_iou = matchloss_hw_weight is not None
        self.matchloss_hw_weight = matchloss_hw_weight
        if not self.use_centered_iou:
            assert (matchloss_objectness_weight + matchloss_yx_weight) == 1
        else:
            assert (matchloss_objectness_weight + matchloss_yx_weight + matchloss_hw_weight) == 1

        self.num_anchors = num_anchors
        self.matchloss_objectness_weight = matchloss_objectness_weight
        self.matchloss_yx_weight = matchloss_yx_weight
        self.max_distance = max_distance
        self.max_distance_sqrd = max_distance ** 2

    def get_matchloss(
        self,
        post_activation_b_posi: Tensor, # post_acivation grid, indexed by prior_mutliplier_b_posi_idx
        prior_mutliplier_b_posi_idx: Tensor, # Indices of grid cells with positive centerness priors
        num_posi_b: int, # Number of grid-cells with positive centerness prior
        annotations_b: AnnotationBlock, # Annotations within the image (not the entire batch)
        downsample_factor: float,
    ) -> Tensor:
        check_tensor(post_activation_b_posi, (self.num_anchors, 7, num_posi_b))
        check_tensor(prior_mutliplier_b_posi_idx, (num_posi_b, 2), torch.int64)

        # y_idxs_b = prior_mutliplier_b_posi_idx[:, 0]
        # x_idxs_b = prior_mutliplier_b_posi_idx[:, 1]
        # post_activation_yx_b_posi_offsets = post_activation_yx[b][:, :, y_idxs_b, x_idxs_b]
        post_activation_yx_b_posi_offsets = post_activation_b_posi[:, YOLO_YX, :]
        prior_mutliplier_b_posi_idx_r = prior_mutliplier_b_posi_idx.T[None, :, :]
        post_activation_yx_b_posi_absolute_p = post_activation_yx_b_posi_offsets + prior_mutliplier_b_posi_idx_r
        post_activation_yx_b_posi_absolute = post_activation_yx_b_posi_absolute_p.permute(2, 0, 1)
        check_tensor(post_activation_yx_b_posi_offsets, (self.num_anchors, 2, num_posi_b))
        check_tensor(prior_mutliplier_b_posi_idx_r, (1, 2, num_posi_b), torch.int64)
        check_tensor(post_activation_yx_b_posi_absolute_p, (self.num_anchors, 2, num_posi_b))
        check_tensor(post_activation_yx_b_posi_absolute, (num_posi_b, self.num_anchors, 2))

        # Get the squared difference between predicted center-yx and true center.
        # Then compute the pairwise center-yx matchloss for prediction+target pairs.
        # distances: [num_posi_b, num_anchors, num_objects_b, 2]
        # norms_sqrd: [num_posi_b, num_anchors, num_objects_b]
        target_yx_b = annotations_b.center_yx / downsample_factor
        num_objects_b = annotations_b.size
        distances = post_activation_yx_b_posi_absolute[:, :, None, :] - target_yx_b[None, None, :, :]
        check_tensor(distances, (num_posi_b, self.num_anchors, num_objects_b, 2))
        distances_sqrd = distances ** 2
        norms_sqrd = distances_sqrd[:, :, :, 0] + distances_sqrd[:, :, :, 1]
        yx_matchloss = 1 - (norms_sqrd / self.max_distance_sqrd)
        check_tensor(yx_matchloss, (num_posi_b, self.num_anchors, num_objects_b))
        if SAFE_MODE:
            assert (0 <= yx_matchloss).any(dim=2).all()
            assert (yx_matchloss <= 1).all()

        # CenteredIoU
        if self.use_centered_iou:
            # Predicted and target sizes
            predicted_hw = post_activation_b_posi[:, YOLO_HW, :].permute(0, 2, 1)[:, :, None, :]
            target_hw = (annotations_b.size_hw / downsample_factor)[None, None, :, :]
            assert annotations_b.has_size_hw.bool().all(), f"All targets must have known HW to compute CenteredIoU match-loss"
            check_tensor(predicted_hw, (self.num_anchors, num_posi_b, 1, 2))
            check_tensor(target_hw, (1, 1, num_objects_b, 2))

            # Intersection height, width, and area
            intersection_hw = torch.min(predicted_hw, target_hw)
            check_tensor(intersection_hw, (self.num_anchors, num_posi_b, num_objects_b, 2))
            intersection_area = intersection_hw.prod(dim=3)
            check_tensor(intersection_area, (self.num_anchors, num_posi_b, num_objects_b))

            # Union area: Predicted area + Target area - intersection area
            target_area = target_hw.prod(dim=3)
            predicted_area = predicted_hw.prod(dim=3)
            union_area = predicted_area + target_area - intersection_area
            check_tensor(union_area, (self.num_anchors, num_posi_b, num_objects_b))

            # CenteredIoU
            centered_iou = (intersection_area / union_area).permute(1, 0, 2)
            assert ((0 <= centered_iou) & (centered_iou <= 1)).all()
            check_tensor(centered_iou, (num_posi_b, self.num_anchors, num_objects_b))

        # Predicted objectness scores
        post_activation_o_b_posi = post_activation_b_posi[:, YOLO_O, :].T[:, :, None]
        check_tensor(post_activation_o_b_posi, (num_posi_b, self.num_anchors, 1))

        # Match loss
        if self.use_centered_iou:

            check_tensor(post_activation_o_b_posi, (num_posi_b, self.num_anchors, 1))
            check_tensor(yx_matchloss, (num_posi_b, self.num_anchors, num_objects_b))
            check_tensor(centered_iou, (num_posi_b, self.num_anchors, num_objects_b))
            match_loss_before_prior: Tensor = (
                post_activation_o_b_posi * self.matchloss_objectness_weight
                + yx_matchloss * self.matchloss_yx_weight
                + centered_iou * self.matchloss_hw_weight
            )

        else:
            match_loss_before_prior: Tensor = (
                post_activation_o_b_posi * self.matchloss_objectness_weight
                + yx_matchloss * self.matchloss_yx_weight
            )
        check_tensor(match_loss_before_prior, (num_posi_b, self.num_anchors, num_objects_b))
        return match_loss_before_prior