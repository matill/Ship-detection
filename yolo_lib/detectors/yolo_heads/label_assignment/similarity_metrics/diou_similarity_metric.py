import torch
from torch import Tensor
from yolo_lib.data.annotation import AnnotationBlock
from yolo_lib.detectors.yolo_heads.label_assignment.similarity_metrics.similarity_metric_base import SimilarityMetric
from yolo_lib.detectors.yolo_heads.output_vector_format import YOLO_O, YOLO_YX, YOLO_HW
from yolo_lib.util.check_tensor import check_tensor


class DIoUPotoMatchloss(SimilarityMetric):
    def __init__(self, num_anchors: int, matchloss_objectness_weight: float, matchloss_box_weight: float) -> None:
        super().__init__()
        assert isinstance(num_anchors, int)
        assert isinstance(matchloss_objectness_weight, float) and (0.0 < matchloss_objectness_weight < 1.0)
        assert isinstance(matchloss_objectness_weight, float) and (0.0 < matchloss_objectness_weight < 1.0)
        assert matchloss_box_weight + matchloss_objectness_weight == 1.0
        self.num_anchors = num_anchors
        self.matchloss_objectness_weight = matchloss_objectness_weight
        self.matchloss_box_weight = matchloss_box_weight

    def _get_top_left_and_bottom_right(self, yx, hw):
        half_hw = hw * 0.5
        top_left = yx - half_hw
        bottom_right = yx + half_hw
        return top_left, bottom_right

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

        # Predicted absolute center positions
        post_activation_yx_b_posi_offsets = post_activation_b_posi[:, YOLO_YX, :]
        prior_mutliplier_b_posi_idx_r = prior_mutliplier_b_posi_idx.T[None, :, :]
        post_activation_yx_b_posi_absolute_p = post_activation_yx_b_posi_offsets + prior_mutliplier_b_posi_idx_r
        post_activation_yx_b_posi_absolute = post_activation_yx_b_posi_absolute_p.permute(2, 0, 1)
        check_tensor(post_activation_yx_b_posi_offsets, (self.num_anchors, 2, num_posi_b))
        check_tensor(prior_mutliplier_b_posi_idx_r, (1, 2, num_posi_b), torch.int64)
        check_tensor(post_activation_yx_b_posi_absolute_p, (self.num_anchors, 2, num_posi_b))
        check_tensor(post_activation_yx_b_posi_absolute, (num_posi_b, self.num_anchors, 2))

        # post_activation_hw_b_posi
        post_activation_hw_b_posi = post_activation_b_posi[:, YOLO_HW, :].permute(2, 0, 1)
        check_tensor(post_activation_hw_b_posi, (num_posi_b, self.num_anchors, 2))

        # Target size and position
        assert annotations_b.has_size_hw.bool().all(), f"DIoUPotoMatchloss expects all annotations to have known HW"
        target_yx_b = annotations_b.center_yx / downsample_factor
        target_hw_b = annotations_b.size_hw / downsample_factor
        num_objects_b = annotations_b.size

        # True and predicted corners
        predicted_top_left, predicted_bottom_right = self._get_top_left_and_bottom_right(post_activation_yx_b_posi_absolute, post_activation_hw_b_posi)
        true_top_left, true_bottom_right = self._get_top_left_and_bottom_right(target_yx_b, target_hw_b)
        check_tensor(predicted_top_left, (num_posi_b, self.num_anchors, 2))
        check_tensor(predicted_bottom_right, (num_posi_b, self.num_anchors, 2))
        check_tensor(true_top_left, (num_objects_b, 2))
        check_tensor(true_bottom_right, (num_objects_b, 2))

        # Smallest and largest top-left and bottom-right "corners"
        smallest_bottom_right = torch.min(predicted_bottom_right[:, :, None, :], true_bottom_right[None, None, :, :])
        largest_bottom_right = torch.max(predicted_bottom_right[:, :, None, :], true_bottom_right[None, None, :, :])
        largest_top_left = torch.max(predicted_top_left[:, :, None, :], true_top_left[None, None, :, :])
        smallest_top_left = torch.min(predicted_top_left[:, :, None, :], true_top_left[None, None, :, :])
        check_tensor(smallest_bottom_right, (num_posi_b, self.num_anchors, num_objects_b, 2))
        check_tensor(largest_bottom_right, (num_posi_b, self.num_anchors, num_objects_b, 2))
        check_tensor(largest_top_left, (num_posi_b, self.num_anchors, num_objects_b, 2))
        check_tensor(smallest_top_left, (num_posi_b, self.num_anchors, num_objects_b, 2))

        # Intersection area
        intersections_hw = (smallest_bottom_right - largest_top_left).relu()
        intersections_area = intersections_hw.prod(dim=3)
        check_tensor(intersections_hw, (num_posi_b, self.num_anchors, num_objects_b, 2))
        check_tensor(intersections_area, (num_posi_b, self.num_anchors, num_objects_b))

        # Union area: predicted area + true area - intersection area
        predicted_area = post_activation_hw_b_posi.prod(dim=2)
        true_area = target_hw_b.prod(dim=1)
        union_area = predicted_area[:, :, None] + true_area[None, None, :] - intersections_area
        check_tensor(predicted_area, (num_posi_b, self.num_anchors))
        check_tensor(true_area, (num_objects_b, ))
        check_tensor(union_area, (num_posi_b, self.num_anchors, num_objects_b))

        # IoU: intersection area / union area
        iou = intersections_area / union_area
        check_tensor(iou, (num_posi_b, self.num_anchors, num_objects_b))

        # largest_bottom_right:      (num_posi_b  self.num_anchors, num_objects_b, 2)
        # smallest_top_left:         (num_posi_b  self.num_anchors, num_objects_b, 2)
        # big_box_diagonal:          (num_posi_b, self.num_anchors, num_objects_b, 2)
        # big_box_diagonal_len_sqrd: (num_posi_b, self.num_anchors, num_objects_b   )
        big_box_diagonal = largest_bottom_right - smallest_top_left
        big_box_diagonal_len_sqrd = big_box_diagonal.square().sum(dim=3)
        check_tensor(big_box_diagonal, (num_posi_b, self.num_anchors, num_objects_b, 2))
        check_tensor(big_box_diagonal_len_sqrd, (num_posi_b, self.num_anchors, num_objects_b))

        # Get the square of the distance between the center points of the annotation and
        # the predicted box
        # target_yx_b:                        (                              num_objects_b, 2)
        # post_activation_yx_b_posi_absolute: (num_posi_b, self.num_anchors,                2)
        # center_differences:                 (num_posi_b, self.num_anchors, num_objects_b, 2)
        # center_differences_len_sqrd:        (num_posi_b, self.num_anchors, num_objects_b   )
        center_differences = target_yx_b[None, None, :, :] - post_activation_yx_b_posi_absolute[:, :, None, :]
        center_differences_len_sqrd = center_differences.square().sum(dim=3)
        check_tensor(center_differences, (num_posi_b, self.num_anchors, num_objects_b, 2))
        check_tensor(center_differences_len_sqrd, (num_posi_b, self.num_anchors, num_objects_b))

        # Compute normalized L2 difference between predicted and true positions
        normalized_l2 = center_differences_len_sqrd / big_box_diagonal_len_sqrd
        check_tensor(normalized_l2, (num_posi_b, self.num_anchors, num_objects_b))

        # Compute DIoU-loss and DIoU-similarity
        # 0 <= DIoU-loss <= 2 
        # 0 <= DIoU-similarity <= 1
        diou_loss       = (1 - iou) + normalized_l2
        diou_similarity = (2 - diou_loss) / 2
        check_tensor(diou_loss, (num_posi_b, self.num_anchors, num_objects_b))
        check_tensor(diou_similarity, (num_posi_b, self.num_anchors, num_objects_b))

        # Predicted positivity / objectness
        post_activation_o_b_posi = post_activation_b_posi[:, YOLO_O, :].T[:, :, None]
        check_tensor(post_activation_o_b_posi, (num_posi_b, self.num_anchors, 1))

        # Match quality: Weighted sum of IoU + objectness
        match_loss_before_prior = (diou_similarity * self.matchloss_box_weight + post_activation_o_b_posi * self.matchloss_objectness_weight)
        check_tensor(match_loss_before_prior, (num_posi_b, self.num_anchors, num_objects_b))
        return match_loss_before_prior

