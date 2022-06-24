"""
Utility functions to generate fake "SAR images" with vessels.
Really just very noisy greyscale images with some clearly visible blobs representing vessels.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, List, Tuple
import numpy as np
from numpy.random import uniform
import torch
from yolo_lib.cfg import DEVICE, SAFE_MODE
from yolo_lib.data.dataclasses import Annotation, AnnotationBlock, YOLOTile
from torch.utils.data import IterableDataset


__all__ = [
    "SyntheticTrainDataSet",
    "SyntheticTestDataSet",
    "FakeDataCfg"
]

np.random.seed(30)

BACKGROUND_LOW = 0
BACKGROUND_HIGH = 3
VESSEL_LOW = 4.0
VESSEL_HIGH = 10


@dataclass
class VesselShapeCfg:
    vessel_length_low: int
    vessel_length_high: int
    vessel_width_low: int
    vessel_width_high: int
    rear_end_width_multiplier: float
    middle_width_multiplier: float


@dataclass
class ImgShapeCfg:
    img_h: int
    img_w: int
    max_fake_vessels: int
    max_real_vessels: int
    num_channels: int=1


@dataclass
class DsWeaknessCfg:
    rotation_known_probability: float
    hw_known_probability: float
    max_yx_error_px: float


@dataclass
class FakeDataCfg:
    img_h: int
    img_w: int

    # Max number of objects that look like vessels that aren't vessels:
    min_fake_vessels: int
    max_fake_vessels: int

    # Probability that a fake vessel is detected as a vessel by an operator:
    fake_vessel_detection_probability: float

    # Probability that a fake vessel gets a correlated "high confidence" status
    fake_vessel_high_confidence_probability: float

    # Max vessels in an image
    min_real_vessels: int
    max_real_vessels: int

    # Probability that a vessel is undetected
    real_vessel_detection_probability: float

    # Probability that a vessel gets a correlated "high confidence" status
    real_vessel_high_confidence_probability: float

    # Min and max length and width of vessels
    vessel_length_low: int
    vessel_length_high: int
    vessel_width_low: int
    vessel_width_high: int

    # Radius of the circle on the rear end of real vessels, relative to the width of the vessel
    rear_end_width_multiplier: float

    # Radius of the circle on the width of fake vessels, relative to the width of the vessel
    middle_width_multiplier: float

    # Probability that the rotation or size of a vessel is known
    rotation_known_probability: float
    hw_known_probability: float

    # Expected error in position (Standard deviation)
    max_yx_error_px: float
    # yx_standard_deviation_px: float

    # Number of channels in output image
    num_channels: int = 1

    def __post_init__(self):
        self.vessel_length_range = self.vessel_length_high - self.vessel_length_low
        self.vessel_width_range = self.vessel_width_high - self.vessel_width_low

    @staticmethod
    def make(
        vessel_shape_cfg: VesselShapeCfg,
        img_shape_cfg: ImgShapeCfg,
        ds_weakness_cfg: DsWeaknessCfg,
    ) -> FakeDataCfg:
        return FakeDataCfg(
                img_shape_cfg.img_h,
                img_shape_cfg.img_w,
                min_fake_vessels=0,
                max_fake_vessels=img_shape_cfg.max_fake_vessels,
                fake_vessel_detection_probability=0.0,
                fake_vessel_high_confidence_probability=0.0,
                min_real_vessels=0,
                max_real_vessels=img_shape_cfg.max_real_vessels,
                real_vessel_detection_probability=1.0,
                real_vessel_high_confidence_probability=1.0,
                vessel_length_low=vessel_shape_cfg.vessel_length_low,
                vessel_length_high=vessel_shape_cfg.vessel_length_high,
                vessel_width_low=vessel_shape_cfg.vessel_width_low,
                vessel_width_high=vessel_shape_cfg.vessel_width_high,
                rear_end_width_multiplier=vessel_shape_cfg.rear_end_width_multiplier,
                middle_width_multiplier=vessel_shape_cfg.middle_width_multiplier,
                rotation_known_probability=ds_weakness_cfg.rotation_known_probability,
                hw_known_probability=ds_weakness_cfg.hw_known_probability,
                max_yx_error_px=ds_weakness_cfg.max_yx_error_px,
                num_channels=img_shape_cfg.num_channels,
            )


class SyntheticDs(IterableDataset):
    def __init__(self, epoch_size: int, cfg: FakeDataCfg, repeat: bool) -> None:
        assert isinstance(epoch_size, int)
        assert isinstance(cfg, FakeDataCfg)
        assert isinstance(repeat, bool)
        super().__init__()
        self.epoch_size = epoch_size
        self.cfg = cfg
        self.repeat = repeat
        if self.repeat:
            self.dataset = [
                generate_image(cfg)
                for _ in range(epoch_size)
            ]

    def __iter__(self) -> Iterator[YOLOTile]:
        if self.repeat:
            return self.dataset.__iter__()
        else:
            return (
                generate_image(self.cfg)
                for _ in range(self.epoch_size)
            )

    def __len__(self) -> int:
        return self.epoch_size


@dataclass
class ObjectDescriptionBlock:
    object_type: torch.Tensor
    rotation: torch.Tensor
    size_hw: torch.Tensor
    yx: torch.Tensor
    known_yx: torch.Tensor
    is_detected: torch.Tensor
    is_high_confidence: torch.Tensor
    is_rotation_known: torch.Tensor
    is_hw_known: torch.Tensor

    def get_annotation_block(self) -> AnnotationBlock:
        num_detected = self.is_detected.sum()
        return AnnotationBlock(
            size=int(num_detected),
            center_yx=self.known_yx[self.is_detected],
            is_high_confidence=self.is_high_confidence[self.is_detected],
            size_hw=self.size_hw[self.is_detected],
            has_size_hw=self.is_hw_known[self.is_detected],
            rotation=self.rotation[self.is_detected],
            has_rotation=self.is_rotation_known[self.is_detected],
            is_rotation_360=torch.tensor(True, device=DEVICE)[None].expand(num_detected),
        )

def uniform_int(low, high) -> int:
    uniform_float = np.random.uniform(low, high+1, ())
    uniform_int = np.floor(uniform_float)
    return int(uniform_int)

@torch.no_grad()
def repeat_float(num_repetitions: int, value: float) -> torch.Tensor:
    if num_repetitions == 0:
        return torch.zeros(num_repetitions, device=DEVICE)
    else:
        return torch.tensor(value, device=DEVICE)[None].expand(num_repetitions)

@torch.no_grad()
def stack(value_1: float, num_1: int, value_2: float, num_2: int) -> torch.Tensor:
    return torch.cat((repeat_float(num_1, value_1), repeat_float(num_2, value_2)))

@torch.no_grad()
def torch_bernulli_scalar(num_elements: int, probability: float) -> torch.Tensor:
    return torch.rand(num_elements, device=DEVICE) < probability

@torch.no_grad()
def torch_bernulli_vector(probability: torch.Tensor) -> torch.Tensor:
    return torch.rand(probability.shape, device=DEVICE) < probability

@torch.no_grad()
def generate_image_description(cfg: FakeDataCfg) -> ObjectDescriptionBlock:
    num_fake_vessels = uniform_int(cfg.min_fake_vessels, cfg.max_fake_vessels)
    num_real_vessels = uniform_int(cfg.min_real_vessels, cfg.max_real_vessels)
    num_vessels = num_fake_vessels + num_real_vessels

    # Positions
    known_yx_min = cfg.max_yx_error_px
    known_y_max = cfg.img_h - cfg.max_yx_error_px
    known_x_max = cfg.img_w - cfg.max_yx_error_px
    known_yx = torch.empty((num_vessels, 2), device=DEVICE)
    known_yx[:, 0] = torch.rand(num_vessels, out=known_yx[:, 0]) * (known_y_max - known_yx_min) + known_yx_min
    known_yx[:, 1] = torch.rand(num_vessels, out=known_yx[:, 1]) * (known_x_max - known_yx_min) + known_yx_min


    center_yx_offset_distances = torch.rand((num_vessels, ), device=DEVICE) * cfg.max_yx_error_px
    center_yx_offset_angles = torch.rand((num_vessels, ), device=DEVICE) * 3.14159 * 2
    center_yx = known_yx.clone()
    center_yx[:, 0] += center_yx_offset_angles.sin() * center_yx_offset_distances
    center_yx[:, 1] += center_yx_offset_angles.cos() * center_yx_offset_distances
    # known_y_offsets = torch.sim(known_y_offsets)
    # known_yx = torch.normal(center_yx, cfg.yx_standard_deviation_px)
    # known_yx = known_yx.relu()
    # known_yx[:, 0] = known_yx[:, 0].min(torch.tensor(cfg.img_h-1.001, device=DEVICE))
    # known_yx[:, 1] = known_yx[:, 1].min(torch.tensor(cfg.img_w-1.001, device=DEVICE))

    # Length and width
    size_hw = torch.zeros((num_vessels, 2), device=DEVICE)
    size_hw[:, 0] = (torch.rand(num_vessels, out=size_hw[:, 0]) * cfg.vessel_length_range) + cfg.vessel_length_low
    size_hw[:, 1] = (torch.rand(num_vessels, out=size_hw[:, 1]) * cfg.vessel_width_range) + cfg.vessel_width_low

    # Flag probabilities for fake vessels
    fake_is_detected_probability = cfg.fake_vessel_detection_probability
    fake_is_high_confidence_probability = cfg.fake_vessel_high_confidence_probability

    # Flag probabilities for real vessels
    real_is_detected_probability = cfg.real_vessel_detection_probability
    real_is_high_confidence_probability = cfg.real_vessel_high_confidence_probability

    # Stacked combined probabilities
    is_detected_probability = stack(
        fake_is_detected_probability, num_fake_vessels,
        real_is_detected_probability, num_real_vessels,
    )
    is_high_confidence_probability = stack(
        fake_is_high_confidence_probability, num_fake_vessels,
        real_is_high_confidence_probability, num_real_vessels,
    )

    # Return result
    return ObjectDescriptionBlock(
        object_type=stack(0, num_fake_vessels, 1, num_real_vessels),
        rotation=torch.rand(num_vessels, device=DEVICE),
        size_hw=size_hw,
        yx=center_yx,
        known_yx=known_yx,
        is_detected=torch_bernulli_vector(is_detected_probability),
        is_high_confidence=torch_bernulli_vector(is_high_confidence_probability),
        is_rotation_known=torch_bernulli_scalar(num_vessels, cfg.rotation_known_probability),
        is_hw_known=torch_bernulli_scalar(num_vessels, cfg.hw_known_probability),
    )

def generate_image(cfg: FakeDataCfg) -> YOLOTile:
    image_description = generate_image_description(cfg)
    image = description_to_image(cfg.img_h, cfg.img_w, image_description, cfg)
    annotation_block = image_description.get_annotation_block()
    return YOLOTile(image[None], annotation_block)

@torch.no_grad()
def create_object_mask(img_h: int, img_w: int, object_descriptions: ObjectDescriptionBlock, cfg: FakeDataCfg) -> torch.Tensor:
    num_vessels = object_descriptions.object_type.shape[0]
    rotation = object_descriptions.rotation
    size_hw = object_descriptions.size_hw
    yx = object_descriptions.yx

    if num_vessels == 0:
        return torch.zeros((img_h, img_w), device=DEVICE, dtype=torch.bool)

    # Generate a rotated ogrid with the vessel-center as origo
    # (map yx coordinates of pixels into the rotated cordinate system)

    # Create a rotation_matrix
    # [num_vessels, 2, 2]
    rotation_radians = rotation * 2 * 3.14159
    sin = torch.sin(rotation_radians)
    cos = torch.cos(rotation_radians)
    assert sin.shape == (num_vessels, )
    assert cos.shape == (num_vessels, )
    # rotation_matrix = torch.tensor([
    #     [-cos, sin],
    #     [sin, cos]
    # ])
    # assert rotation_matrix.shape == (num_vessels, 2, 2)

    # Cntered y_grid (num_vessels, img_h)
    y_grid = torch.arange(0, img_h, 1, device=DEVICE)[None, :].expand(num_vessels, -1)
    center_y_reshaped = yx[:, 0][:, None].expand(-1, img_h)
    centered_y_grid = y_grid - center_y_reshaped
    assert centered_y_grid.shape == (num_vessels, img_h)

    # Centered x grid (num_vessels, img_w)
    x_grid = torch.arange(0, img_w, 1, device=DEVICE)[None, :].expand(num_vessels, -1)
    centered_x_reshaped = yx[:, 1][:, None].expand(-1, img_w)
    centered_x_grid = x_grid - centered_x_reshaped
    assert centered_x_grid.shape == (num_vessels, img_w)

    # Rotated y grid (num_vessels, img_h)
    y_rotated_grid_y = -(centered_y_grid * cos[:, None].expand(-1, img_h))
    y_rotated_grid_x = +(centered_x_grid * sin[:, None].expand(-1, img_w))
    y_rotated_grid = y_rotated_grid_x[:, None, :] + y_rotated_grid_y[:, :, None]
    assert y_rotated_grid.shape == (num_vessels, img_h, img_w)

    # Rotated x grid (num_vessels, img_w)
    x_rotated_grid_y = +(centered_y_grid * sin[:, None].expand(-1, img_h))
    x_rotated_grid_x = +(centered_x_grid * cos[:, None].expand(-1, img_w))
    x_rotated_grid = x_rotated_grid_x[:, None, :] + x_rotated_grid_y[:, :, None]
    assert x_rotated_grid.shape == (num_vessels, img_h, img_w)

    # Bitmap of pixels in ellipsis
    ellipse_h = size_hw[:, 0] / 2
    ellipse_w = size_hw[:, 1] / 2
    assert ellipse_h.shape == (num_vessels, )
    assert ellipse_w.shape == (num_vessels, )
    scores_y = (y_rotated_grid / ellipse_h[:, None, None].expand(-1, img_h, img_w)) ** 2
    scores_x = (x_rotated_grid / ellipse_w[:, None, None].expand(-1, img_h, img_w)) ** 2
    assert scores_y.shape == (num_vessels, img_h, img_w)
    assert scores_x.shape == (num_vessels, img_h, img_w)
    scores = scores_y + scores_x
    ellipsis_mask = torch.max(scores <= 1, dim=0).values
    assert ellipsis_mask.shape == (img_h, img_w)
    assert ellipsis_mask.dtype == torch.bool

    # Get circle center and radius for real and fake vessels "branchless"
    is_real_mask = object_descriptions.object_type
    is_fake_mask = 1 - is_real_mask

    # Circle center and radius for real vessels
    rear_end_direction = -torch.stack([-cos, sin], dim=1)
    assert rear_end_direction.shape == (num_vessels, 2)
    ellipse_h_reshaped = ellipse_h[:, None].expand(-1, 2)
    assert ellipse_h_reshaped.shape == (num_vessels, 2)
    rear_end_pos = rear_end_direction * ellipse_h_reshaped
    rear_end_pos = yx + rear_end_pos
    assert rear_end_pos.shape == (num_vessels, 2)
    circle_center_real = rear_end_pos * is_real_mask[:, None].expand(-1, 2)
    circle_radius_real = ellipse_w * cfg.rear_end_width_multiplier * is_real_mask
    assert circle_center_real.shape == (num_vessels, 2)
    assert circle_radius_real.shape == (num_vessels, )

    # Circle center and radius for fake vessels
    circle_center_fake = yx * is_fake_mask[:, None].expand(-1, 2)
    circle_radius_fake = ellipse_w * cfg.middle_width_multiplier * is_fake_mask
    assert circle_center_fake.shape == (num_vessels, 2)
    assert circle_radius_fake.shape == (num_vessels, )

    # Circle radius and center for all vessels
    circle_center = circle_center_real + circle_center_fake
    circle_radius = circle_radius_real + circle_radius_fake

    # Distance from center in x direction
    # (num_vessels, img_w)
    circle_center_x_reshaped = circle_center[:, 1][:, None].expand(-1, img_w)
    assert circle_center_x_reshaped.shape == (num_vessels, img_w)
    x_dist = x_grid - circle_center_x_reshaped
    x_dist_sqrd = x_dist ** 2
    x_dist_sqrd_reshaped = x_dist_sqrd[:, None, :]
    assert x_dist_sqrd_reshaped.shape == (num_vessels, 1, img_w)

    # Distence from center in y direction
    # (num_vessels, img_h)
    circle_center_y_reshaped = circle_center[:, 0][:, None].expand(-1, img_h)
    assert circle_center_y_reshaped.shape == (num_vessels, img_h)
    y_dist = y_grid - circle_center_y_reshaped
    y_dist_sqrd = y_dist ** 2
    y_dist_sqrd_reshaped = y_dist_sqrd[:, :, None]
    assert y_dist_sqrd_reshaped.shape == (num_vessels, img_h, 1)

    # Bitmap over which pixels are within the radius
    yx_dist_sqrd = x_dist_sqrd_reshaped + y_dist_sqrd_reshaped
    circle_radius_sqrd = circle_radius ** 2
    circle_radius_sqrd_reshaped = circle_radius_sqrd[:, None, None].expand(-1, img_h, img_w)
    assert circle_radius_sqrd_reshaped.shape == (num_vessels, img_h, img_w)
    assert yx_dist_sqrd.shape == (num_vessels, img_h, img_w)
    circle_mask = (yx_dist_sqrd <= (circle_radius_sqrd_reshaped)).max(dim=0).values
    assert circle_mask.shape == (img_h, img_w)

    # Combined mask
    mask = torch.max(circle_mask, ellipsis_mask)
    return mask


@torch.no_grad()
def description_to_image(
    img_h: int,
    img_w: int,
    object_decriptions: ObjectDescriptionBlock,
    cfg: FakeDataCfg,
) -> torch.Tensor:
    img_shape = (cfg.num_channels, img_h, img_w)

    vessel_mask = create_object_mask(img_h, img_w, object_decriptions, cfg)
    vessel_image_base = torch.rand(size=img_shape, device=DEVICE) * (VESSEL_HIGH - VESSEL_LOW) + VESSEL_LOW
    vessel_image = vessel_mask[None] * vessel_image_base

    background_image_base = torch.rand(size=img_shape, device=DEVICE) * (BACKGROUND_HIGH - BACKGROUND_LOW) + BACKGROUND_LOW
    background_mask = ~vessel_mask
    background_image = background_image_base * background_mask[None]

    image = vessel_image + background_image
    assert image.shape == (img_shape)
    return image

