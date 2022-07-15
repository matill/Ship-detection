from __future__ import annotations
from torch import nn, Tensor
from typing import Dict, Tuple
from yolo_lib.data.dataclasses import YOLOTileStack
from yolo_lib.data.detection import DetectionGrid


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

