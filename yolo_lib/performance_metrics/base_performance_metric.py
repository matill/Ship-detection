


from typing import Any, Dict
from yolo_lib.data.detection import DetectionBlock
from yolo_lib.data.annotation import AnnotationBlock

class BasePerformanceMetric:
    def reset(self):
        raise NotImplementedError

    def increment(
        self,
        detections: DetectionBlock,
        annotations: AnnotationBlock,
    ):
        raise NotImplementedError

    def divide(self, upper, lower):
        if lower == 0:
            return None
        else:
            return float(upper / lower)

    def finalize(self) -> Dict[str, Any]:
        raise NotImplementedError

