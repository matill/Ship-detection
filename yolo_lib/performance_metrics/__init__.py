from .base_ap import BaseAveragePrecision
from .base_performance_metric import BasePerformanceMetric
from .composed_performance_metrics import ComposedPerformanceMetrics
from .distance_ap import DistanceAP, MeanDistanceAP
from .iou_ap import IoUAP, MeanIoUAP
from .linear_sum_assignment_performance import LinSumPerformance
from .regression_metrics import CenterDistaneMetric, RegressionMetric, RotationMetric, CenteredIoUMetric

def get_default_performance_metrics() -> BasePerformanceMetric:
    iou_steps = [
        0.50, 0.55,
        0.60, 0.65,
        0.70, 0.75,
        0.80, 0.85,
        0.90, 0.95
    ]
    return ComposedPerformanceMetrics({
        "IoU-AP": MeanIoUAP(100, iou_steps),
        "Distance-AP": DistanceAP(
            distance_threshold=50.0,
            max_detections=100,
            regression_metrics=[CenteredIoUMetric(), RotationMetric(), CenterDistaneMetric()],
            include_f2=True
        )
    })

