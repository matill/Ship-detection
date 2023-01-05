from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
from torch import nn, Tensor
from yolo_lib.cfg import DEVICE
from yolo_lib.data.annotation import AnnotationBlock
from yolo_lib.data.detection import DetectionGrid
from yolo_lib.data.dataclasses import YOLOTileStack
from yolo_lib.detectors.yolo_heads.label_assignment.label_assignment import LabelAssignment
from yolo_lib.detectors.yolo_heads.label_assignment.label_assignment_cfg import AssignmentLossCfg
from yolo_lib.detectors.yolo_heads.label_assignment.spatial_prior.spatial_prior_base import SpatialPrior
from yolo_lib.detectors.yolo_heads.label_assignment.similarity_metrics.similarity_metric_base import SimilarityMetric
from yolo_lib.detectors.yolo_heads.losses.adv_loss import ADVLoss
from yolo_lib.detectors.yolo_heads.losses.complete_box_losses import BoxLoss
from yolo_lib.detectors.yolo_heads.losses.objectness_loss import ConfidenceUnawareObjectnessLoss
from yolo_lib.detectors.yolo_heads.annotation_encoding import PointAnnotationEncoding, SizeAnnotationEncoding
from yolo_lib.models.blocks.conv5d import Conv5D
from yolo_lib.util.check_tensor import check_tensor
from scipy.optimize import linear_sum_assignment
from yolo_lib.detectors.yolo_heads.output_vector_format import YOLO_O, YOLO_YX, YOLO_HW, YOLO_SINCOS



@dataclass
class YOLOHeadCfg:
    assignment_loss_cfg: AssignmentLossCfg
    yx_multiplier: float

    num_anchors: int

    adv_loss_fn: ADVLoss
    complete_box_loss_fn: BoxLoss
    objectness_loss_fn: ConfidenceUnawareObjectnessLoss

    loss_objectness_weight: float
    loss_box_weight: float
    loss_sincos_weight: float

    def build(self, in_channels: int) -> YOLOHead:
        return YOLOHead(
            in_channels,
            self.num_anchors,

            self.yx_multiplier,
            self.assignment_loss_cfg,

            self.adv_loss_fn,
            self.complete_box_loss_fn,
            self.objectness_loss_fn,

            self.loss_objectness_weight,
            self.loss_box_weight,
            self.loss_sincos_weight,
        )


class YOLOHead(nn.Module):
    OUTPUTS_PER_ANCHOR = 7

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,

        yx_multiplier: float,
        assignment_loss_cfg: AssignmentLossCfg,

        adv_loss_fn: ADVLoss,
        complete_box_loss_fn: BoxLoss,
        objectness_loss_fn: ConfidenceUnawareObjectnessLoss,

        loss_objectness_weight: float,
        loss_box_weight: float,
        loss_sincos_weight: float,

        num_classes: int=0,
    ):
        assert isinstance(in_channels, int)
        assert isinstance(num_anchors, int)
        assert isinstance(yx_multiplier, float)
        assert isinstance(assignment_loss_cfg, AssignmentLossCfg)
        assert isinstance(adv_loss_fn, ADVLoss)
        assert isinstance(complete_box_loss_fn, BoxLoss)
        assert isinstance(objectness_loss_fn, ConfidenceUnawareObjectnessLoss)
        assert isinstance(loss_objectness_weight, float)
        assert isinstance(loss_box_weight, float)
        assert isinstance(loss_sincos_weight, float)
        assert (loss_objectness_weight + loss_box_weight + loss_sincos_weight) == 1
        assert isinstance(num_classes, int) and 0 <= num_classes
        super().__init__()
        self.complete_box_loss_fn = complete_box_loss_fn
        self.adv_loss_fn = adv_loss_fn
        self.objectness_loss_fn = objectness_loss_fn
        self.matchloss_fn: SimilarityMetric = assignment_loss_cfg.get_similarity_metric_fn(yx_multiplier, num_anchors)
        self.spatial_prior_fn: SpatialPrior = assignment_loss_cfg.get_spatial_prior_fn(yx_multiplier)
        self.loss_objectness_weight = loss_objectness_weight
        self.loss_box_weight = loss_box_weight
        self.loss_sincos_weight = loss_sincos_weight
        self.yx_multiplier = yx_multiplier
        self.num_classes = num_classes
        self.outputs_per_anchor = self.OUTPUTS_PER_ANCHOR + num_classes
        self.num_anchors = num_anchors
        self.yolo_class_channels = list(range(self.OUTPUTS_PER_ANCHOR, self.outputs_per_anchor))
        self.wrapped_yolo_head = Conv5D(
            in_channels,
            num_anchors,
            self.outputs_per_anchor
        )

    def absolute_yx_helper(self, post_activation_yx: Tensor, downsample_factor: int) -> Tensor:
        batch_size, num_anchors, two, h, w = post_activation_yx.shape
        assert two == 2
        y_idxs = torch.arange(0, h, 1, dtype=torch.float32, device=DEVICE)[None, None, :, None]
        x_idxs = torch.arange(0, w, 1, dtype=torch.float32, device=DEVICE)[None, None, None, :]
        check_tensor(y_idxs, (1, 1, h, 1))
        check_tensor(x_idxs, (1, 1, 1, w))
        absolute_yx = torch.empty((batch_size, num_anchors, 2, h, w), dtype=torch.float32, device=DEVICE)
        absolute_yx[:, :, 0, :, :] = (post_activation_yx[:, :, 0, :, :] + y_idxs) * downsample_factor
        absolute_yx[:, :, 1, :, :] = (post_activation_yx[:, :, 1, :, :] + x_idxs) * downsample_factor
        return absolute_yx

    def decode_sincos_grid(self, direction_vectors: Tensor) -> Tensor:
        batch_size, num_anchors, two, h, w = direction_vectors.shape
        assert two == 2
        unit_vectors = direction_vectors / torch.norm(direction_vectors, dim=2, keepdim=True)
        sin, cos = unit_vectors[:, :, 0], unit_vectors[:, :, 1]
        arcsin_radians = torch.arcsin(sin)
        arcsin_01 = arcsin_radians / (2 * 3.14159)

        # Depending on which quadrant the angle is within (depending on the sign of sin and cos),
        # get the angle value of all detections. Implements the following pseudocode, using
        # a "branchless" implementation
        # if cos < 0:
        #     angles = 0.5 - arcsin_01
        # else:
        #     if sin > 0:
        #         angles = arcsin_01
        #     else:
        #         angles = 1 + arcsin_01
        angles_cos_below_0 = (0.5 - arcsin_01) * (cos < 0)
        angles_cos_above_0 = (arcsin_01 + (sin < 0)) * (cos >= 0)
        angles = angles_cos_above_0 + angles_cos_below_0
        check_tensor(angles, (batch_size, num_anchors, h, w))
        return angles

    def detect_objects(self, post_activation: Tensor, downsample_factor: int) -> DetectionGrid:
        batch_size, num_anchors, _, h, w = post_activation.shape
        return DetectionGrid.new(
            (batch_size, num_anchors, h, w),
            self.absolute_yx_helper(post_activation[:, :, YOLO_YX, :, :], downsample_factor),
            post_activation[:, :, YOLO_O, :, :],
            post_activation[:, :, YOLO_HW, :, :] * downsample_factor,
            self.decode_sincos_grid(post_activation[:, :, YOLO_SINCOS, :, :]),
            post_activation[:, :, self.yolo_class_channels, :, :] if self.num_classes else None,
        )

    def get_pre_activation(self, feature_map: Tensor) -> Tensor:
        return self.wrapped_yolo_head(feature_map)

    def get_post_activation(self, pre_activation: Tensor) -> Tensor:
        post_activation = torch.cat([
            torch.tanh(pre_activation[:, :, YOLO_YX, :, :]) * self.yx_multiplier + 0.5,
            torch.exp(pre_activation[:, :, YOLO_HW, :, :]),
            torch.sigmoid(pre_activation[:, :, [YOLO_O], :, :]),
            pre_activation[:, :, YOLO_SINCOS, :, :],
            torch.softmax(pre_activation[:, :, self.yolo_class_channels, :, :], dim=2),
        ], dim=2)

        assert post_activation.shape == pre_activation.shape
        return post_activation

    def get_post_activation_o(self, post_activation: Tensor) -> Tensor:
        return post_activation[:, :, YOLO_O, :, :]

    @torch.no_grad()
    def get_assignment(self,
        pre_activation: Tensor,
        post_activation: Tensor,
        tiles: YOLOTileStack,
        downsample_factor: int
    ) -> LabelAssignment:
        # Get annotation encodings
        annotations = tiles.annotations
        true_center_yx = annotations.center_yx / downsample_factor

        # Get batch-size, height, width, and number of objects
        batch_size = pre_activation.shape[0]
        height = pre_activation.shape[3]
        width = pre_activation.shape[4]
        exp_shape = (batch_size, self.num_anchors, self.outputs_per_anchor, height, width)
        check_tensor(pre_activation, exp_shape)

        # Compute spatial prior
        prior_multiplier = self.spatial_prior_fn.compute(height, width, annotations, downsample_factor)

        # For each image in the batch
        post_activation_yx = post_activation[:, :, YOLO_YX, :, :]
        post_activation_o = pre_activation[:, :, YOLO_O, :, :]
        check_tensor(post_activation_yx, (batch_size, self.num_anchors, 2, height, width))
        check_tensor(post_activation_o, (batch_size, self.num_anchors, height, width))
        assignments_per_image = []
        for b in range(batch_size):
            # prior_multiplier_b: a subset of prior_multiplier for the objects
            # in the current image in the batch
            batch_annotation_bitmap_nonzero_p = (tiles.img_idxs == b).nonzero()
            batch_annotation_bitmap_nonzero = batch_annotation_bitmap_nonzero_p[:, 0]
            num_objects_b = int(batch_annotation_bitmap_nonzero_p.shape[0])
            target_yx_b = true_center_yx[batch_annotation_bitmap_nonzero]
            prior_multiplier_b = prior_multiplier[batch_annotation_bitmap_nonzero]
            check_tensor(batch_annotation_bitmap_nonzero_p, (num_objects_b, 1), torch.int64)
            check_tensor(batch_annotation_bitmap_nonzero, (num_objects_b, ), torch.int64)
            check_tensor(target_yx_b, (num_objects_b, 2))
            check_tensor(prior_multiplier_b, (num_objects_b, height, width))

            # Skip if the image is empty
            if num_objects_b == 0:
                continue

            # prior_mutliplier_b_posi: Subset of prior_multiplier_b where the entrires are positive / nonzero.
            # prior_mutliplier_b_posi_idx: Index tensor such that
            # prior_mutliplier_b_posi[n] = prior_multiplier_b[prior_mutliplier_b_posi_idx[n]]
            prior_mutliplier_b_posi_idx_bitmap = (prior_multiplier_b > 0).any(dim=0)
            prior_mutliplier_b_posi_idx = prior_mutliplier_b_posi_idx_bitmap.nonzero()
            num_posi_b = prior_mutliplier_b_posi_idx.shape[0]
            prior_mutliplier_b_posi = prior_multiplier_b[:, prior_mutliplier_b_posi_idx[:, 0], prior_mutliplier_b_posi_idx[:, 1]].T
            check_tensor(prior_mutliplier_b_posi_idx_bitmap, (height, width), torch.bool)
            check_tensor(prior_mutliplier_b_posi_idx, (num_posi_b, 2), torch.int64)
            check_tensor(prior_mutliplier_b_posi, (num_posi_b, num_objects_b))

            # Match-loss before and after applying prior multiplier
            # post_activation[b]: [num_anchors, 7, h, w]
            # post_activation[b][:, :, y_idxs_b, x_idxs_b]: [num_anchors, 7, num_posi_b]
            y_idxs_b = prior_mutliplier_b_posi_idx[:, 0]
            x_idxs_b = prior_mutliplier_b_posi_idx[:, 1]
            post_activation_b_posi = post_activation[b][:, :, y_idxs_b, x_idxs_b]
            check_tensor(post_activation_b_posi, (self.num_anchors, self.outputs_per_anchor, num_posi_b))
            annotations_b = annotations.extract_index_tensor(batch_annotation_bitmap_nonzero)
            match_loss_before_prior = self.matchloss_fn.get_matchloss(
                post_activation_b_posi,
                prior_mutliplier_b_posi_idx,
                num_posi_b,
                annotations_b,
                downsample_factor
            )
            check_tensor(match_loss_before_prior, (num_posi_b, self.num_anchors, num_objects_b))
            prior_mutliplier_b_posi_r = prior_mutliplier_b_posi[:, None, :]
            match_loss = match_loss_before_prior * prior_mutliplier_b_posi_r
            check_tensor(prior_mutliplier_b_posi_r, (num_posi_b, 1, num_objects_b))
            check_tensor(match_loss, (num_posi_b, self.num_anchors, num_objects_b))

            # Flatten matchloss to a matrix shape before applying Hungarian matching
            # match_loss_flattened: [num_posi_b * num_anchors, num_objects_b]
            match_loss_idxs_n = torch.arange(0, num_posi_b, 1, device=DEVICE)[:, None, None].expand(-1, self.num_anchors, -1)
            match_loss_idxs_anchor = torch.arange(0, self.num_anchors, 1, device=DEVICE)[None, :, None].expand(num_posi_b, -1, -1)
            match_loss_idxs = torch.cat([match_loss_idxs_n, match_loss_idxs_anchor], dim=2)
            check_tensor(match_loss_idxs, (num_posi_b, self.num_anchors, 2))
            match_loss_idxs_flattened = match_loss_idxs.flatten(start_dim=0, end_dim=1)
            check_tensor(match_loss_idxs_flattened, (num_posi_b * self.num_anchors, 2))
            match_loss_flattened = match_loss.flatten(start_dim=0, end_dim=1)
            check_tensor(match_loss_flattened, (num_posi_b * self.num_anchors, num_objects_b))

            # Match-loss to numpy array and run Hungarian algorithm
            # match_anchor_idxs: Indexes into the first axis of match_loss_flattened
            # match_obj_idxs: Indexes into the first axis of match_loss_flattened
            match_loss_flattened_np = match_loss_flattened.cpu().detach().numpy()
            match = linear_sum_assignment(match_loss_flattened_np, maximize=True)
            match_anchor_idxs_np, match_obj_idxs_np = match
            match_anchor_idxs = torch.tensor(match_anchor_idxs_np, dtype=torch.int64, device=DEVICE)
            match_obj_idxs = torch.tensor(match_obj_idxs_np, dtype=torch.int64, device=DEVICE)
            check_tensor(match_anchor_idxs, (num_objects_b, ))
            check_tensor(match_obj_idxs, (num_objects_b, ))

            # Convert match_anchor_idxs into y_ixs, x_idxs and anchor_idxs.
            posi_and_anchor_idxs = match_loss_idxs_flattened[match_anchor_idxs]
            check_tensor(posi_and_anchor_idxs, (num_objects_b, 2), torch.int64)
            posi_idxs = posi_and_anchor_idxs[:, 0]
            anchor_idxs = posi_and_anchor_idxs[:, 1]
            check_tensor(posi_idxs, (num_objects_b, ), torch.int64)
            check_tensor(anchor_idxs, (num_objects_b, ), torch.int64)
            yx_idxs = prior_mutliplier_b_posi_idx[posi_idxs]
            check_tensor(yx_idxs, (num_objects_b, 2), torch.int64)
            y_idxs = yx_idxs[:, 0]
            x_idxs = yx_idxs[:, 1]

            # Convert match_obj_idxs (indices into match_loss_flattened matrix) into
            # object_idxs (indices into tiles.annotations)
            object_idxs = batch_annotation_bitmap_nonzero[match_obj_idxs]
            check_tensor(object_idxs, (num_objects_b, ), torch.int64)

            # Append assignment-segment for this image
            img_idxs = torch.tensor(b, dtype=torch.int64, device=DEVICE)[None].expand(num_objects_b)
            assignments_per_image.append(
                LabelAssignment.new(num_objects_b, img_idxs, anchor_idxs, y_idxs, x_idxs, object_idxs)
            )

        # Stack assignments computed for each image
        return LabelAssignment.stack(assignments_per_image)

    def compute_loss(
        self,
        pre_activation: Tensor,
        post_activation: Tensor,
        tiles: YOLOTileStack,
        downsample_factor: int
    ) -> Tuple[Tensor, Dict[str, float]]:
        # Compute assignment
        assignment = self.get_assignment(
            pre_activation,
            post_activation,
            tiles,
            downsample_factor,
        )

        # Reorder the tiles.annotations block to have the same ordering as the assignment object_idxs.
        # This way we can redefine assignment to have object_idxs = [0, 1, 2, 3 ...], since this matches
        # the ordering of the annotations. This also removes elements from tiles.annotations that are unassigned,
        # in any case where that would happen.
        # This makes it a lot easier to compute a PointAnnotationEncoding
        annotations = tiles.annotations.extract_index_tensor(assignment.object_idxs)
        assignment = LabelAssignment(
            assignment.num_assignments,
            assignment.full_head_idxs,
            torch.arange(0, assignment.num_assignments, 1, device=DEVICE, dtype=torch.int64)
        )

        # Get a PointAnnotationEncoding, customized specifically for this model.
        # The default constructor [PointAnnotationEncoding.encode(tiles, downsample_factor)],
        # assumes each object is assigned to an anchor within the same cell, which is
        # wrong for this model.
        # Instead, compute yx_annotation_encoding as an offset from which
        # based on which y_idxs, x_idxs is assigned to the object.
        # The PointAnnotationEncoding's img_idxs, y_idxs, x_idxs variables are only used to
        # compute match loss, but we don't do that for this model, so instead, we just use
        # None values for those.
        (img_idxs, head_idxs, grid_y_idxs, grid_x_idxs) = assignment.get_grid_idx_vectors()
        center_yx_downsampled = annotations.center_yx / downsample_factor
        grid_yx_idxs = torch.cat([grid_y_idxs[:, None], grid_x_idxs[:, None]], dim=1)
        check_tensor(grid_yx_idxs, (assignment.num_assignments, 2), torch.int64)
        center_yx = center_yx_downsampled - grid_yx_idxs.float()
        yx_annotation_encoding = PointAnnotationEncoding(annotations.size, center_yx, None, None, None)

        # Get SizeAnnotationEncoding, using default constructors
        hw_annotation_encoding = SizeAnnotationEncoding.encode(annotations, downsample_factor)

        # Compute objectness loss
        objectness_loss = self.objectness_loss_fn(
            pre_activation[:, :, YOLO_O, :, :],
            assignment
        )

        # Get yx and hw loss
        try:
            box_loss = self.complete_box_loss_fn(
                post_activation[:, :, YOLO_YX, :, :],
                post_activation[:, :, YOLO_HW, :, :],
                assignment,
                yx_annotation_encoding,
                hw_annotation_encoding
            )
        except Exception as e:
            print("post_activation.isinf()", (post_activation.isnan() | post_activation.isinf()).nonzero())
            raise e

        # Get direction loss
        sincos_loss = self.adv_loss_fn(
            post_activation[:, :, YOLO_SINCOS, :, :],
            annotations,
            assignment,
        )

        # Compute sub of subterms
        loss = (
            objectness_loss * self.loss_objectness_weight
            + box_loss * self.loss_box_weight
            + sincos_loss * self.loss_sincos_weight
        )
        subterms = {
            "objectness_loss": objectness_loss,
            "box_loss": box_loss,
            "sincos_loss": sincos_loss,
        }
        return loss, subterms


