from .base_ap import BaseAveragePrecision
from .base_performance_metric import BasePerformanceMetric
from .composed_performance_metrics import ComposedPerformanceMetrics
from .distance_ap import DistanceAP, MeanDistanceAP
from .iou_ap import IoUAP, MeanIoUAP
from .linear_sum_assignment_performance import LinSumPerformance
from .regression_metrics import CenterDistaneMetric, RegressionMetric, RotationMetric, CenteredIoUMetric, SubclassificationMetrics

def get_default_performance_metrics(
    include_iou_ap: bool=False,
    dap_distance_threshold: float=50.0,
    dap_max_detections_per_tile: int=100,
    num_classes: int=0,
) -> BasePerformanceMetric:
    cfg_dict = {
        "Distance-AP": DistanceAP(
            distance_threshold=dap_distance_threshold,
            max_detections=dap_max_detections_per_tile,
            regression_metrics=[CenteredIoUMetric(), RotationMetric(), CenterDistaneMetric(), SubclassificationMetrics(num_classes)],
            include_f2=True
        )
    }

    if include_iou_ap:
        iou_steps = [
            0.50, 0.55,
            0.60, 0.65,
            0.70, 0.75,
            0.80, 0.85,
            0.90, 0.95
        ]
        cfg_dict["Iou-AP"] = MeanIoUAP(100, iou_steps)

    return ComposedPerformanceMetrics(cfg_dict)

