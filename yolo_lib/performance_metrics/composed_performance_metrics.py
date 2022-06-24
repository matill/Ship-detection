from typing import Dict
from yolo_lib.data.annotation import AnnotationBlock
from yolo_lib.data.detection import DetectionBlock
from yolo_lib.performance_metrics.base_performance_metric import BasePerformanceMetric


class ComposedPerformanceMetrics(BasePerformanceMetric):
    def __init__(self, sub_metrics: Dict[str, BasePerformanceMetric]):
        self.sub_metrics = sub_metrics

    def reset(self):
        for sub_metric in self.sub_metrics.values():
            sub_metric.reset()
    
    def increment(
        self,
        detections: DetectionBlock,
        annotations: AnnotationBlock,
    ):
        for sub_metric in self.sub_metrics.values():
            sub_metric.increment(detections, annotations)

    def finalize(self):
        return {
            metric_name: metric.finalize()
            for (metric_name, metric) in self.sub_metrics.items()
        }
