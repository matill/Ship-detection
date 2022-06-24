"""
Self adversarial training
"""

from typing import Tuple
import random
import torch
from yolo_lib.cfg import DEVICE
from yolo_lib.data.dataclasses import AnnotationBlock, YOLOTileStack
from yolo_lib.detectors.managed_architectures.base_detector import BaseDetector
from yolo_lib.util import check_tensor
from .data_augmentation import DataAugmentation


_PRIME_FACTORIZATIONS = [
    None,
    None,
    [2],
    [3],
    [2, 2],
    [5],
    [2, 3],
    [7],
    [2, 2, 2],
    [3, 3],
    [2, 5],
    [11],
    [2, 2, 3],
    [13],
    [2, 7],
    [3, 5],
    [2, 2, 2, 2],
    [17],
    [2, 3, 3],
    [19],
    [2, 2, 5],
    [3, 7],
    [2, 11],
    [23],
    [2, 2, 2, 3],
    [5, 5],
    [2, 13],
    [27],
    [2, 2, 7],
    [29],
    [2, 3, 5],
    [31],
    [2, 2, 2, 2, 2],
    [3, 11],
    [2, 17],
    [5, 7],
    [2, 2, 3, 3],
    [37],
]

for x, factorization in enumerate(_PRIME_FACTORIZATIONS):
    if x < 2:
        continue
    product = 1
    for factor in factorization:
        product *= factor

    assert product == x


class FlatBatch(DataAugmentation):
    def __init__(self, static_p: float) -> None:
        assert 0 <= static_p <= 1
        self.static_p = static_p

    def get_probability(self, epoch: int) -> float:
        return self.static_p

    def _get_stack_shape(self, num_images) -> Tuple[int, int, int]:
        if num_images == 1:
            return (1, 1, 1)

        factorization = _PRIME_FACTORIZATIONS[num_images]
        n_height = 1
        n_width = 1
        n_batch = 1
        for factor in factorization:
            random_int = random.randint(0, 2)
            if random_int == 0:
                n_height *= factor
            elif random_int == 1:
                n_width *= factor
            else:
                n_batch *= factor

        return (n_height, n_width, n_batch)

    def apply(self, tiles: YOLOTileStack, model: BaseDetector) -> YOLOTileStack:
        # Decide how the new image should be shaped: How many images stacked in
        # the height dimension, how many stacked in the width, and how many batches?
        num_images, channels, h, w = tiles.images.shape
        n_height, n_width, n_batch = self._get_stack_shape(num_images)

        # Put images into new tensor
        new_img = torch.empty((n_batch, channels, h * n_height, w * n_width), device=DEVICE)
        i = -1
        y_offsets = []
        x_offsets = []
        img_idxs_img = []
        for i_batch in range(n_batch):
            for i_height in range(n_height):
                h_min = i_height * h
                h_max = h_min + h
                for i_width in range(n_width):
                    i += 1
                    w_min = i_width * w
                    w_max = w_min + w
                    src_image = tiles.images[i]
                    dst_image = new_img[i_batch, :, h_min:h_max, w_min:w_max]
                    dst_image[:] = src_image
                    y_offsets.append(h_min)
                    x_offsets.append(w_min)
                    img_idxs_img.append(i_batch)

        # Move annotations according to their new position
        # yx_offsets_img: [num_images, 2] shaped tensor describing the offset of each image
        # yx_offsets_obj: [num_objects, 2] shapes tensor describing the offset of each object
        yx_offsets_img = torch.tensor([y_offsets, x_offsets], dtype=torch.float32, device=DEVICE).T
        yx_offsets_obj = yx_offsets_img[tiles.img_idxs]
        tiles.annotations.center_yx += yx_offsets_obj

        # Get index of the images each object belongs to.
        # img_idxs_img: [num_images] shaped tensor corresponding to 
        # image-indices (into the new image-stack) that images from the old
        # image-stack belong to.
        # img_idxs_obj: [num_objects] shaped tensor corresponding to which objects
        # correspond to which image in the new image stack
        img_idxs_obj = torch.tensor(img_idxs_img, device=DEVICE)[tiles.img_idxs]

        # Return YOLOTileStack with new image and updated annotations
        return YOLOTileStack(
            new_img,
            img_idxs_obj,
            tiles.annotations
        )

