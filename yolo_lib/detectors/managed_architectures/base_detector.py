from dataclasses import dataclass
import torch
from torch import nn
from typing import Dict, List, Optional, Tuple
from yolo_lib.data.dataclasses import Detection, DetectionBlock, YOLOTile, YOLOTileStack
from yolo_lib.data.detection import DetectionGrid


class BaseDetector(nn.Module):
    def set_detection_tags(self, detection_tags: List[str]) -> None:
        self.detection_tags = detection_tags

    def get_detection_tags(self, detection_tags: List[str]) -> None:
        self.detection_tags = detection_tags

    def detect_objects(
        self,
        images: torch.Tensor,
    ) -> DetectionGrid:
        """
        Must be implemented by sub classes.
        Accepts an image as input. Returns a list of Detection objects
        tuples. size, class_probability and mask elements can be None.
        """
        print(f"ERROR: {self.__class__.__name__} does not implement self.detect_objects")

    def compute_loss(self, tiles: YOLOTileStack) -> Tuple[torch.Tensor, Dict[str, float]]:
        print(f"ERROR: {self.__class__.__name__} does not implement self.compute_loss")


class PostProcessor:
    def __init__(self) -> None:
        pass

    def get_all_tags(self) -> List[str]:
        print(f"ERROR: {self.__class__.__name__}.get_all_tags() not implemented")

    def apply_post_processing(self, detection_batch: List[List[Detection]]):
        print(f"ERROR: {self.__class__.__name__}.apply_post_processing() not implemented")

    def filtered(self, detection_set: List[Detection], tag: str) -> List[Detection]:
        return [
            detection for detection in detection_set
            if detection.tags is not None and tag in detection.tags
        ]

