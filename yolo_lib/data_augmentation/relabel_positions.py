"""
Self adversarial training
"""

import torch
import torch.nn.functional as F
from yolo_lib.cfg import DEVICE
from yolo_lib.data.dataclasses import AnnotationBlock, YOLOTileStack
from yolo_lib.detectors.managed_architectures.base_detector import BaseDetector
from yolo_lib.util import check_tensor
from yolo_lib.cfg import DEVICE
from .data_augmentation import DataAugmentation
import numpy as np
from scipy.optimize import linear_sum_assignment


class RelabelPositions(DataAugmentation):
    def __init__(self, kernel_size: int, distance_threshold_px: float) -> None:
        assert kernel_size % 2 == 1, f"kernel_size must be an odd number. Got {kernel_size}"
        self.kernel = torch.tensor(1.0, device=DEVICE)[None, None, None, None].expand(-1, -1, kernel_size, kernel_size)
        self.kernel_sub = - int((kernel_size - 1) / 2)
        self.kernel_add = + int((kernel_size + 1) / 2)
            # self.kernel = torch.tensor([[[
            #     [1.0, 1.0, 1.0],
            #     [1.0, 1.0, 1.0],
            #     [1.0, 1.0, 1.0],
            # ]]])

    def get_probability(self, epoch: int) -> float:
        return 1.0

    def apply(self, tiles: YOLOTileStack, model: BaseDetector) -> YOLOTileStack:
        images = tiles.images
        num_images, _, h, w = images.shape

        new_annotation_blocks = []
        
        # Get smoothed image
        smoothing = F.conv2d(images, self.kernel, bias=None, padding="same")

        # For each image, get the 2N highest peaks in the smoothing, where
        # N is the number of objects in the image
        for img_idx in range(num_images):
            annotations = tiles.annotations.extract_bitmap(tiles.img_idxs == img_idx)
            smoothing_i = smoothing[img_idx]
            num_objects = annotations.size
            if num_objects == 0:
                new_annotation_blocks.append(AnnotationBlock.empty())
                continue

            # Get the N candidate peaks
            peak_idxs = []
            peak_scores = []
            for i in range(num_objects):
                # peak_idx = torch.argmax(smoothing_i)
                peak_idx = np.unravel_index(smoothing_i.argmax(), smoothing_i.shape)
                _, peak_idx_y, peak_idx_x = peak_idx
                peak_idxs.append(np.array([peak_idx_y, peak_idx_x]))
                peak_square_ymin = peak_idx_y + self.kernel_sub
                peak_square_ymax = peak_idx_y + self.kernel_add
                peak_square_xmin = peak_idx_x + self.kernel_sub
                peak_square_xmax = peak_idx_x + self.kernel_add
                smoothing_i[:, peak_square_ymin:peak_square_ymax, peak_square_xmin:peak_square_xmax] = 0

            # Apply a linear sum assignment to match annotations with peaks
            peak_positions_yx = np.array(peak_idxs)
            annotation_positions_yx = annotations.center_yx.cpu().detach().numpy()
            print("peak_positions_yx", peak_positions_yx)
            print("peak_idxs", peak_idxs)
            assert peak_positions_yx.shape == (num_objects, 2)
            assert annotation_positions_yx.shape == (num_objects, 2)
            distance_matrix_yx = peak_positions_yx[None, :, :] - annotation_positions_yx[:, None, :]
            assert distance_matrix_yx.shape == (num_objects, num_objects, 2)
            distance_matrix = np.linalg.norm(distance_matrix_yx, axis=2)
            assert distance_matrix.shape == (num_objects, num_objects)
            assignment = linear_sum_assignment(distance_matrix, maximize=False)
            obj_idxs, peak_idxs = assignment

            # Create a new center_yx matrix
            peak_positions_torch = torch.tensor(peak_positions_yx, dtype=torch.float32, device=DEVICE)
            new_center_yx = torch.empty((num_objects, 2), device=DEVICE)
            new_center_yx[obj_idxs] = peak_positions_torch[peak_idxs]

            # distance_matrix[i,j] = norm(annotation_positions_yx[i] - peak_positions_yx[j])
            annotations.center_yx = new_center_yx
            new_annotation_blocks.append(annotations)

        # Stack the new annotation blocks, and create a new YOLOTileStack
        return YOLOTileStack(
            # images,
            smoothing,
            YOLOTileStack.get_img_idxs(new_annotation_blocks),
            AnnotationBlock.stack_blocks(new_annotation_blocks)
        )

