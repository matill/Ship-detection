from typing import List
import torch
from yolo_lib.performance_metrics.base_ap import BaseAveragePrecision
from yolo_lib.data.dataclasses import DetectionBlock, AnnotationBlock
from yolo_lib.performance_metrics.base_performance_metric import BasePerformanceMetric
from yolo_lib.performance_metrics.regression_metrics import RegressionMetric


class DistanceAP(BaseAveragePrecision):
    def __init__(
        self,
        distance_threshold: float,
        max_detections: int,
        regression_metrics: List[RegressionMetric],
        include_f2: bool,
    ):
        assert isinstance(distance_threshold, (float, int))
        super().__init__(max_detections, regression_metrics, include_f2)
        self.distance_threshold = distance_threshold
        self.reset()

    def __get_similarity_matrix__(
        self,
        detections: DetectionBlock,
        annotations: AnnotationBlock,
    ) -> torch.Tensor:
        distances_euclid = detections.center_yx[:, None, :] - annotations.center_yx[None, :, :]
        assert distances_euclid.shape == (detections.size, annotations.size, 2)
        distances_l2 = distances_euclid.norm(dim=2)
        assert distances_l2.shape == (detections.size, annotations.size)
        similarity_matrix = -distances_l2
        return similarity_matrix

    def __threshold_similarity_matrix__(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        distances_l2 = -similarity_matrix
        return distances_l2 < self.distance_threshold


class MeanDistanceAP(BasePerformanceMetric):
    def __init__(self, max_detections: int, distance_thresholds: List[float]):
        self.sub_metrics = [
            DistanceAP(distance_threshold, max_detections, [])
            for distance_threshold in distance_thresholds
        ]
        self.distance_thresholds = distance_thresholds
        self.reset()

    def reset(self):
        for sub_metric in self.sub_metrics:
            sub_metric.reset()
    
    def increment(
        self,
        detections: DetectionBlock,
        annotations: AnnotationBlock,
    ):
        for sub_metric in self.sub_metrics:
            sub_metric.increment(detections, annotations)

    def finalize(self):
        performance = {
            f"AP_{t}": sub_metric.finalize()["AP"]
            for (sub_metric, t) in zip(self.sub_metrics, self.distance_thresholds)
        }

        ap = torch.tensor([val for val in performance.values()]).mean()
        performance["AP"] = float(ap)
        return performance



