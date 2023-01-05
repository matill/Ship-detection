from __future__ import annotations
from torch import nn, Tensor
from typing import Dict, Tuple
from yolo_lib.data.dataclasses import YOLOTileStack
from yolo_lib.data.detection import DetectionGrid, Detection
from typing import List


class DetectorCfg:
    def build(self) -> BaseDetector:
        pass


class BaseDetector(nn.Module):
    def detect_objects(
        self,
        images: Tensor,
    ) -> DetectionGrid:
        """
        Must be implemented by sub classes.
        Accepts an image as input. Returns a list of Detection objects
        tuples. size, class_probability and mask elements can be None.
        """
        print(f"ERROR: {self.__class__.__name__} does not implement self.detect_objects")

    def compute_loss(self, tiles: YOLOTileStack) -> Tuple[Tensor, Dict[str, float]]:
        print(f"ERROR: {self.__class__.__name__} does not implement self.compute_loss")

    def detect_objects_simple(self, image: Tensor, min_positivity: float) -> List[Detection]:
        """
        Return list of all detections in an image where positivity (confidence) is greater than the given threshold
        """
        assert image.shape[0] == 1, f"detect_objects_simple() expects a batch containing one image. Got ({int(image.shape[0])})"
        return self.detect_objects(image) \
                .as_detection_block() \
                .filter_min_positivity(min_positivity) \
                .as_detection_list()


