from typing import List, Tuple
import torch
from yolo_lib.cfg import DEVICE
from yolo_lib.data.dataclasses import AnnotationBlock, YOLOTileStack
from yolo_lib.data.yolo_tile import YOLOTile
from yolo_lib.detectors.managed_architectures.base_detector import BaseDetector
from yolo_lib.util.check_tensor import check_tensor
from .data_augmentation import DataAugmentation


class RandomCrop(DataAugmentation):
    def __init__(self, crop_sizes: List[int], static_p: float, epochs_without: int) -> None:
        assert 0 <= static_p <= 1
        self.crop_sizes = crop_sizes
        self.static_p = static_p
        self.epochs_without = epochs_without

    def get_probability(self, epoch: int) -> float:
        if epoch < self.epochs_without:
            return 0.0
        else:
            return self.static_p

    @staticmethod
    def get_crop_positions(old_w: int, new_w: int) -> Tuple[int, int]:
        randint_range_size = old_w - new_w
        crop_ofst_min = torch.randint(low=0, high=randint_range_size, size=(1, ))
        crop_ofst_max = crop_ofst_min + new_w
        return (crop_ofst_min, crop_ofst_max)

    @staticmethod
    def random_crop_tile(tile: YOLOTile, new_h: int, new_w: int) -> YOLOTile:
        image_old = tile.image
        annotations_old = tile.annotations
        _, channels, old_h, old_w = image_old.shape

        # Get crop positions
        crop_y_min, crop_y_max = RandomCrop.get_crop_positions(old_h, new_h)
        crop_x_min, crop_x_max = RandomCrop.get_crop_positions(old_w, new_w)
        crop_yx_min = torch.tensor([crop_y_min, crop_x_min], dtype=torch.float32, device=DEVICE)
        crop_yx_max = torch.tensor([crop_y_max, crop_x_max], dtype=torch.float32, device=DEVICE)

        # Crop image
        image_new = image_old[:, :, crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        check_tensor(image_new, (1, channels, new_h, new_w))

        # Find which objects are within the crop. Extract entries from AnnotationBlock that are within,
        # and compute a new center_yx offset from the new crop
        center_yx = annotations_old.center_yx
        center_yx_within_bitmap = ((crop_yx_min[None, :] <= center_yx) & (center_yx < crop_yx_max[None, :])).all(1)
        check_tensor(center_yx_within_bitmap, (annotations_old.size, ), torch.bool)
        annotations_extracted = annotations_old.extract_bitmap(center_yx_within_bitmap)
        annotations_new = AnnotationBlock(
            annotations_extracted.size,
            annotations_extracted.center_yx - crop_yx_min,
            annotations_extracted.is_high_confidence,
            annotations_extracted.size_hw,
            annotations_extracted.has_size_hw,
            annotations_extracted.rotation,
            annotations_extracted.has_rotation,
            annotations_extracted.is_rotation_360,
        )

        # Create a YOLOTile
        assert annotations_new.size <= annotations_old.size
        return YOLOTile(image_new, annotations_new)

    def apply(self, tiles: YOLOTileStack, model: BaseDetector) -> YOLOTileStack:
        # Decide crop sizes
        new_h = self.crop_sizes[torch.randint(0, len(self.crop_sizes), ())]
        new_w = self.crop_sizes[torch.randint(0, len(self.crop_sizes), ())]
        _, _, old_h, old_w = tiles.images.shape
        assert old_h >= new_h and old_w >= new_w, f"Got h={old_h} and w={old_w}, when new_h={new_h} and new_w={new_w}"

        # Crop each tile
        return YOLOTileStack.stack_tiles([
            RandomCrop.random_crop_tile(tile, new_h, new_w)
            for tile in tiles.split_into_tiles()
        ])

