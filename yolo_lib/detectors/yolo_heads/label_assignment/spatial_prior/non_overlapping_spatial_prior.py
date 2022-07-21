from dataclasses import dataclass
from torch import Tensor
import torch
from yolo_lib.data.annotation import AnnotationBlock
from yolo_lib.detectors.yolo_heads.label_assignment.spatial_prior.spatial_prior_base import SpatialPrior
from yolo_lib.cfg import DEVICE
from yolo_lib.util.check_tensor import check_tensor


class NonOverlappingSpatialPrior(SpatialPrior):
    def compute(self, img_h: int, img_w: int, annotations: AnnotationBlock, downsample_factor: int) -> Tensor:
        num_objects = annotations.size

        # Index of region each object is contained in
        true_center_yx = annotations.center_yx / downsample_factor
        true_center_yx_idxs = true_center_yx.floor().type(torch.int64)
        true_center_y_idxs = true_center_yx_idxs[:, 0]
        true_center_x_idxs = true_center_yx_idxs[:, 1]

        # Index of each object
        obj_idxs = torch.arange(0, num_objects, 1, dtype=torch.int64, device=DEVICE)

        # Prior multiplier:
        # 1 for the region that contains each object
        # 0 for all others
        prior_multiplier = torch.zeros((num_objects, img_h, img_w), dtype=torch.float32, device=DEVICE)
        prior_multiplier[obj_idxs, true_center_y_idxs, true_center_x_idxs] = 1.0
        return prior_multiplier


