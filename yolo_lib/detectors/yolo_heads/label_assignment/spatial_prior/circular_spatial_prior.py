from dataclasses import dataclass
from torch import Tensor
import torch
from yolo_lib.data.annotation import AnnotationBlock
from yolo_lib.detectors.yolo_heads.label_assignment.spatial_prior.spatial_prior_base import SpatialPrior
from yolo_lib.cfg import DEVICE
from yolo_lib.util import check_tensor


class CircularSpatialPrior(SpatialPrior):
    def __init__(self, flat_prior: bool, yx_match_threshold: float) -> None:
        self.flat_prior = flat_prior
        self.yx_match_threshold_sqrd = yx_match_threshold ** 2

    def compute(self, img_h: int, img_w: int, annotations: AnnotationBlock, downsample_factor: int) -> Tensor:
        true_center_yx = annotations.center_yx / downsample_factor

        # Get batch-size, height, width, and number of objects
        num_objects = annotations.size

        # Get prior multiplier [num_objects, height, width]
        # y_dists_sqrd: Distance from regression-target to cell-centers in y-direction, squared
        # x_dists_sqrd: Distance from regression-target to cell-centers in x-direction, squared
        y_dists_sqrd = (true_center_yx[:, 0][:, None] - (torch.arange(0, img_h, 1, device=DEVICE) + 0.5)[None, :]) ** 2
        x_dists_sqrd = (true_center_yx[:, 1][:, None] - (torch.arange(0, img_w, 1, device=DEVICE) + 0.5)[None, :]) ** 2
        check_tensor(y_dists_sqrd, (num_objects, img_h))
        check_tensor(x_dists_sqrd, (num_objects, img_w))
        y_dists_sqrd_normalized = (y_dists_sqrd / self.yx_match_threshold_sqrd)[:, :, None]
        x_dists_sqrd_normalized = (x_dists_sqrd / self.yx_match_threshold_sqrd)[:, None, :]

        # prior_multiplier: Either 0 or 1 if using a flat prior, or exactly 0 or 1, if using a smooth prior
        prior_multiplier_pre_activation = 1 - (y_dists_sqrd_normalized + x_dists_sqrd_normalized)
        if self.flat_prior:
            prior_multiplier = (0 <= prior_multiplier_pre_activation).type(torch.float32)
        else:
            prior_multiplier = torch.relu(prior_multiplier_pre_activation)
        assert y_dists_sqrd_normalized.shape == (num_objects, img_h, 1)
        assert x_dists_sqrd_normalized.shape == (num_objects, 1, img_w)
        assert prior_multiplier.shape == (num_objects, img_h, img_w)
        return prior_multiplier


