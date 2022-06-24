from __future__ import annotations
from typing import List, Tuple
from yolo_lib.data.dataclasses import YOLOTileStack
from yolo_lib.detectors.managed_architectures.base_detector import BaseDetector
import torch


class DataAugmentation:
    """
    Base class for data augmentations.
    Implementors need to implement get_probability() and apply(). 
    """

    @staticmethod
    def apply_list(data_augmentations: List[DataAugmentation], tiles: YOLOTileStack, model: BaseDetector, epoch: int) -> Tuple[YOLOTileStack, int]:
        num_applied = 0
        for data_augmentation in data_augmentations:
            tiles, is_applied = data_augmentation.apply_given_p(tiles, model, epoch)
            num_applied += int(is_applied)

        return tiles, num_applied

    def apply_given_p(self, tiles: YOLOTileStack, model: BaseDetector, epoch: int) -> Tuple[YOLOTileStack, bool]:
        # Return input tiles directly with some probability
        p = self.get_probability(epoch)
        assert isinstance(p, float) and 0 <= p <= 1, f"Expected p to be a float 0 <= p <= 1. Got {p}"
        if torch.rand(1) >= p:
            return tiles, False

        after_apply = self.apply(tiles, model)
        assert isinstance(after_apply, YOLOTileStack)
        return after_apply, True

    def get_probability(self, epoch: int) -> float:
        raise NotImplementedError

    def apply(self, tiles: YOLOTileStack, model: BaseDetector) -> YOLOTileStack:
        raise NotImplementedError


