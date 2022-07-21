from __future__ import annotations
from dataclasses import dataclass
import random
from typing import List, Optional, Tuple
import torch
from yolo_lib.cfg import DEVICE
from yolo_lib.data.dataclasses import AnnotationBlock, YOLOTileStack
from yolo_lib.detectors.managed_architectures.base_detector import BaseDetector
from yolo_lib.util.check_tensor import check_tensor
from .data_augmentation import DataAugmentation
import numpy as np


class Mosaic(DataAugmentation):
    def __init__(
        self,
        static_p: float,
        bbox_size_pixels: int,
        epochs_without: int,
        object_p: float,
        num_retries: int = 5
    ) -> None:
        assert isinstance(bbox_size_pixels, int) and bbox_size_pixels % 2 == 0, f"bbox_size_pixels must be an even number. got {bbox_size_pixels}"
        assert isinstance(static_p, float) and 0 <= static_p <= 1
        assert isinstance(epochs_without, int)
        assert isinstance(object_p, float)
        assert isinstance(num_retries, int)
        self.static_p = static_p
        self.bbox_size_pixels = bbox_size_pixels
        self.half_bbox_size_pixels = int(bbox_size_pixels / 2)
        self.epochs_without = epochs_without
        self.object_p = object_p
        self.num_retries = num_retries

    def get_probability(self, epoch: int) -> float:
        if epoch < self.epochs_without:
            return 0.0
        else:
            return self.static_p

    def apply(self, tiles: YOLOTileStack, model: BaseDetector) -> YOLOTileStack:
        # Number of images
        num_images, c, h, w = tiles.images.shape

        # List of bounding boxes for all objects
        object_bboxes: List[BBox] = []
        img_idxs = tiles.img_idxs.cpu()
        for i, annotation in enumerate(tiles.annotations.as_annotation_list()):
            center_yx = annotation.center_yx.astype(np.int32)
            yx_min = center_yx - self.half_bbox_size_pixels
            yx_max = center_yx + self.half_bbox_size_pixels
            img_idx = int(img_idxs[i]) # Casting is crucial to make a copy, and avoid shared mutable reference bugs
            object_bboxes.append(BBox(yx_min[0], yx_max[0], yx_min[1], yx_max[1], img_idx, center_yx))

        # Determine a set of targets: Objects that we attempt to move to other images
        target_obj_idxs = []
        for i, obj_bbox in enumerate(object_bboxes):
            num_overlaps = (-1) + sum((BBox.overlap(obj_bbox, other_bbox) for other_bbox in object_bboxes))
            no_overlaps = (num_overlaps == 0)
            if no_overlaps and random.uniform(0.0, 1.0) < self.static_p:
                target_obj_idxs.append(i)

        # Shuffle the order of targets.
        # Try to find a new location in a new image for the object.
        # Make sure the new location does not overlap any other objects. No object should
        # originate from that position, and no other object should be moved to that location.
        shuffled_target_obj_idxs = [i for i in target_obj_idxs]
        destination_bboxes: List[Tuple[int, BBox]] = []
        random.shuffle(shuffled_target_obj_idxs)
        for obj_idx in shuffled_target_obj_idxs:
            # obj_bbox = object_bboxes[obj_idx]
            dest_bbox = self.try_find_free_bbox(object_bboxes, destination_bboxes, num_images, h, w)
            if dest_bbox is not None:
                destination_bboxes.append((obj_idx, dest_bbox))

        # swaps: The set of bbox-pairs to swap between different images, and
        # the corresponding index (into the AnnotationBloxk) of the object
        # that is being moved
        # TODO: Implemented this subroutine. At this point we have defined how the swap should
        # be executed. What is missing is to perform the swap.
        # 1: Implement a function that swaps small boxes in the image
        # 2: Implement a function that updates img_idx and center_yx in AnnotationBlock.
        # 3: Check if (2) works well in "litterally" edge cases where the obj_bbox is slightly outside the image
        # 4: Display results
        swaps: List[Tuple[int, BBox, BBox]] = []
        for obj_idx, dest_bbox in destination_bboxes:
            # Destination bbox is assumed to be fully contained in the image
            assert 0 <= dest_bbox.ymin < dest_bbox.ymax < h
            assert 0 <= dest_bbox.xmin < dest_bbox.xmax < w

            # The object bbox may be slightly outside the image. If so, we increment
            # the edge that is outside, both for the object-bbox and the destination-bbox
            # so they stay at the same size.
            obj_bbox = object_bboxes[obj_idx]
            ymin_pad = max(-obj_bbox.ymin, 0)
            xmin_pad = max(-obj_bbox.xmin, 0)
            ymax_pad = max(1 + obj_bbox.ymax - h, 0)
            xmax_pad = max(1 + obj_bbox.xmax - w, 0)
            obj_bbox = obj_bbox.apply_pad(ymin_pad, ymax_pad, xmin_pad, xmax_pad)
            dest_bbox = dest_bbox.apply_pad(ymin_pad, ymax_pad, xmin_pad, xmax_pad)
            swaps.append((obj_idx, obj_bbox, dest_bbox))

        # Apply swaps, and return result
        self.apply_swaps(swaps, tiles)
        return tiles

    def apply_swaps(
        self,
        swaps: List[Tuple[int, BBox, BBox]],
        tiles: YOLOTileStack,
    ):
        if len(swaps) == 0:
            return

        # Get list/vector representation of new positions, image indices, and indices into AnnotationBlock
        obj_idxs = []
        new_img_idxs = []
        new_center_y = []
        new_center_x = []
        for (idx, obj_bbox, dest_bbox) in swaps:
            new_center_y.append((dest_bbox.ymin + dest_bbox.ymax) / 2)
            new_center_x.append((dest_bbox.xmin + dest_bbox.xmax) / 2)
            obj_idxs.append(idx)
            new_img_idxs.append(dest_bbox.img_idx)

        # Update center_yx positions and img_idxs
        obj_idxs = torch.tensor(obj_idxs, device=DEVICE)
        new_center_yx = torch.tensor([new_center_y, new_center_x], dtype=tiles.annotations.center_yx.dtype, device=DEVICE).T
        tiles.annotations.center_yx[obj_idxs] = new_center_yx
        tiles.img_idxs[obj_idxs] = torch.tensor(new_img_idxs, device=DEVICE)

        # Swap regions from images
        num_images, c, h, w = tiles.images.shape
        buffer = torch.empty(c, self.bbox_size_pixels, self.bbox_size_pixels, device=DEVICE)
        for (idx, obj_bbox, dest_bbox) in swaps:
            # Get equally shaped views into swap-buffer, and two regions in the image
            obj_bbox.assert_equal_size(dest_bbox)
            crt_box_h, crt_box_w = obj_bbox.get_hw()
            buffer_view = buffer[:, :crt_box_h, :crt_box_w]
            obj_bbox_view = obj_bbox.index_images(tiles.images)
            dest_bbox_view = dest_bbox.index_images(tiles.images)
            assert buffer_view.shape == (c, crt_box_h, crt_box_w)
            assert obj_bbox_view.shape == (c, crt_box_h, crt_box_w)
            assert dest_bbox_view.shape == (c, crt_box_h, crt_box_w)

            # Perform triangle swap
            buffer_view[:] = obj_bbox_view
            obj_bbox_view[:] = dest_bbox_view
            dest_bbox_view[:] = buffer_view
            # buffer_view[:] = 0
            # obj_bbox_view[:] = 0
            # dest_bbox_view[:] = 0

    def try_find_free_bbox(
        self,
        object_bboxes: List[BBox],
        destination_bboxes: List[Tuple[int, BBox]],
        num_images: int,
        h: int,
        w: int
    ) -> Optional[BBox]:
        """
        Tries to find a bounding box that does not overlap any objects, and does not overlap
        the destination of any objects that are being moved.
        Tests a random location N times and checks for overlap. Gives up after N failures
        """
        for _ in range(self.num_retries):
            dest_img_idx = random.randint(0, num_images-1)
            dest_center_y = random.randint(self.half_bbox_size_pixels, h - self.half_bbox_size_pixels - 1)
            dest_center_x = random.randint(self.half_bbox_size_pixels, w - self.half_bbox_size_pixels - 1)
            dest_bbox = BBox(
                dest_center_y - self.half_bbox_size_pixels,
                dest_center_y + self.half_bbox_size_pixels,
                dest_center_x - self.half_bbox_size_pixels,
                dest_center_x + self.half_bbox_size_pixels,
                dest_img_idx
            )

            overlaps_any_obj = any((BBox.overlap(dest_bbox, bbox) for bbox in object_bboxes))
            overlaps_any_dest = any((BBox.overlap(dest_bbox, bbox) for (_idx, bbox) in destination_bboxes))
            
            if overlaps_any_obj or overlaps_any_dest:
                continue
            else:
                return dest_bbox

        return None


@dataclass
class BBox:
    ymin: int
    ymax: int
    xmin: int
    xmax: int
    img_idx: int
    center_yx: np.ndarray = None

    @staticmethod
    def _overlap_helper(a_min, a_max, b_min, b_max) -> bool:
        return not (a_min >= b_max or b_min >= a_max)

    @staticmethod
    def overlap(a: BBox, b: BBox) -> bool:
        if a.img_idx != b.img_idx:
            return False

        overlap_y = BBox._overlap_helper(a.ymin, a.ymax, b.ymin, b.ymax)
        overlap_x = BBox._overlap_helper(a.xmin, a.xmax, b.xmin, b.xmax)
        return overlap_y and overlap_x

    def apply_pad(self, ymin_pad, ymax_pad, xmin_pad, xmax_pad) -> BBox:
        return BBox(
            self.ymin + ymin_pad,
            self.ymax - ymax_pad,
            self.xmin + xmin_pad,
            self.xmax - xmax_pad,
            self.img_idx,
            self.center_yx
        )

    def index_images(self, images: torch.Tensor) -> torch.Tensor:
        """Returns a view into an image tensor"""
        return images[int(self.img_idx)][:, self.ymin:self.ymax, self.xmin:self.xmax]

    def get_hw(self) -> Tuple[int, int]:
        return (int(self.ymax - self.ymin), int(self.xmax - self.xmin))

    def assert_equal_size(self, other: BBox):
        assert self.get_hw() == other.get_hw(), f"Got shapes. self {self.get_hw()} other {other.get_hw}"
