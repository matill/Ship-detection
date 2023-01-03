
from typing import List
import torch
from yolo_lib.util.iou import get_iou
from yolo_lib.performance_metrics.base_ap import BaseAveragePrecision
from yolo_lib.data.dataclasses import DetectionBlock, AnnotationBlock
from yolo_lib.performance_metrics.base_performance_metric import BasePerformanceMetric
from yolo_lib.performance_metrics.regression_metrics import RegressionMetric

class IoUAP(BaseAveragePrecision):
    def __init__(
        self,
        iou_threshold: float,
        max_detections: int,
        regression_metrics: List[RegressionMetric]
    ):
        assert isinstance(iou_threshold, float)
        super().__init__(max_detections, regression_metrics)
        self.iou_threshold = iou_threshold
        self.reset()

    def __get_similarity_matrix__(
        self,
        detections: DetectionBlock,
        annotations: AnnotationBlock,
    ) -> torch.Tensor:
        assert annotations.has_size_hw.bool().all(), "Cannot compute IoU when annotations have unknown size"
        assert detections.size_hw is not None, "Cannot compute IoU when detections do not include size"
        return get_iou(
            detections.center_yx[:, None, :].expand(-1, annotations.size, -1),
            detections.size_hw[:, None, :].expand(-1, annotations.size, -1),
            annotations.center_yx[None, :, :].expand(detections.size, -1, -1),
            annotations.size_hw[None, :, :].expand(detections.size, -1, -1),
            detections.size,
            annotations.size
        )

    def __threshold_similarity_matrix__(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        return similarity_matrix > self.iou_threshold
        

class MeanIoUAP(BasePerformanceMetric):
    def __init__(self, max_detections: int, iou_thresholds: List[float]):
        self.sub_metrics = [
            IoUAP(iou_threshold, max_detections, [])
            for iou_threshold in iou_thresholds
        ]
        self.iou_thresholds = iou_thresholds
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
            for (sub_metric, t) in zip(self.sub_metrics, self.iou_thresholds)
        }

        ap = torch.tensor([val for val in performance.values()]).mean()
        performance["AP"] = float(ap)
        return performance

