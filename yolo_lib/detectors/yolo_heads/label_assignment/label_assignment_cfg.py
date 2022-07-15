from dataclasses import dataclass
from typing import Optional
from yolo_lib.detectors.yolo_heads.label_assignment.similarity_metrics.diou_similarity_metric import DIoUPotoMatchloss
from yolo_lib.detectors.yolo_heads.label_assignment.spatial_prior.spatial_prior_base import SpatialPrior
from yolo_lib.detectors.yolo_heads.label_assignment.spatial_prior.circular_spatial_prior import CircularSpatialPrior
from yolo_lib.detectors.yolo_heads.label_assignment.spatial_prior.non_overlapping_spatial_prior import NonOverlappingSpatialPrior
from yolo_lib.detectors.yolo_heads.label_assignment.similarity_metrics.distance_similarity_metric import PointPotoMatchloss
from yolo_lib.detectors.yolo_heads.label_assignment.similarity_metrics.iou_similarity_metric import IoUPotoMatchloss
from yolo_lib.detectors.yolo_heads.label_assignment.similarity_metrics.similarity_metric_base import SimilarityMetric


class AssignmentLossCfg:
    def get_spatial_prior_fn(self, yx_multiplier: float) -> SpatialPrior:
        raise NotImplementedError

    def get_similarity_metric_fn(self, yx_multiplier: float, num_anchors: int) -> SimilarityMetric:
        raise NotImplementedError


@dataclass
class DistanceBasedOverlappingAssignmentLossCfg(AssignmentLossCfg):
    yx_match_threshold: float
    flat_prior: bool

    matchloss_objectness_weight: float
    matchloss_yx_weight: float
    matchloss_hw_weight: Optional[float] = None

    def get_spatial_prior_fn(self, yx_multiplier: float) -> SpatialPrior:
        return CircularSpatialPrior(self.flat_prior, self.yx_match_threshold)

    def get_similarity_metric_fn(self, yx_multiplier: float, num_anchors: int) -> SimilarityMetric:
        assert yx_multiplier > self.yx_match_threshold
        max_distance = (2 ** 0.5) * yx_multiplier  + self.yx_match_threshold
        return PointPotoMatchloss(
            num_anchors,
            self.matchloss_objectness_weight,
            self.matchloss_yx_weight,
            max_distance,
            self.matchloss_hw_weight
        )


@dataclass
class IoUBasedOverlappingAssignmentLossCfg(AssignmentLossCfg):
    yx_match_threshold: float
    flat_prior: bool

    matchloss_objectness_weight: float
    matchloss_box_weight: float

    def get_spatial_prior_fn(self, yx_multiplier: float) -> SpatialPrior:
        return CircularSpatialPrior(self.flat_prior, self.yx_match_threshold)

    def get_similarity_metric_fn(self, yx_multiplier: float, num_anchors: int) -> SimilarityMetric:
        return IoUPotoMatchloss(num_anchors, self.matchloss_objectness_weight, self.matchloss_box_weight)


@dataclass
class DIoUBasedOverlappingAssignmentLossCfg(AssignmentLossCfg):
    yx_match_threshold: float
    flat_prior: bool

    matchloss_objectness_weight: float
    matchloss_box_weight: float

    def get_spatial_prior_fn(self, yx_multiplier: float) -> SpatialPrior:
        return CircularSpatialPrior(self.flat_prior, self.yx_match_threshold)

    def get_similarity_metric_fn(self, yx_multiplier: float, num_anchors: int) -> SimilarityMetric:
        return DIoUPotoMatchloss(num_anchors, self.matchloss_objectness_weight, self.matchloss_box_weight)


@dataclass
class DistanceBasedNonOverlappingAssignmentLossCfg(AssignmentLossCfg):
    yx_match_threshold: float
    flat_prior: bool

    matchloss_objectness_weight: float
    matchloss_yx_weight: float

    def get_spatial_prior_fn(self, yx_multiplier: float) -> SpatialPrior:
        assert yx_multiplier > 0.5
        return NonOverlappingSpatialPrior()

    def get_similarity_metric_fn(self, yx_multiplier: float, num_anchors: int) -> SimilarityMetric:
        assert yx_multiplier > 0.5
        max_distance = (2 ** 0.5) * (yx_multiplier + 0.5)
        print("max_distance", max_distance)
        print("yx_multiplier", yx_multiplier)
        return PointPotoMatchloss(
            num_anchors,
            self.matchloss_objectness_weight,
            self.matchloss_yx_weight,
            max_distance,
            matchloss_hw_weight=None,
        )

