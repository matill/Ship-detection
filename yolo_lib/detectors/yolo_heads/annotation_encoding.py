from __future__ import annotations
from typing import Tuple
import torch
import numpy as np
import unittest
from dataclasses import dataclass
from yolo_lib.cfg import DEVICE
from yolo_lib.data.dataclasses import Annotation, YOLOTileStack, AnnotationBlock
from yolo_lib.torch_consts import DEVICE_TWO_PI


@dataclass
class PointAnnotationEncoding:
    num_annotations: int
    center_yx: torch.Tensor
    img_idxs: torch.Tensor
    y_idxs: torch.Tensor
    x_idxs: torch.Tensor

    @staticmethod
    @torch.no_grad()
    def encode(tiles: YOLOTileStack, downsample_factor: torch.Tensor) -> PointAnnotationEncoding:
        assert isinstance(downsample_factor, torch.Tensor)
        assert isinstance(tiles, YOLOTileStack)

        # Get yx offsets and yx idxs
        annotations = tiles.annotations
        center_yx_divided = annotations.center_yx / downsample_factor
        yx_idxs_float = torch.floor(center_yx_divided)
        center_yx_offsets = center_yx_divided - yx_idxs_float
        yx_idxs_int = yx_idxs_float.type(torch.int64)

        # Return result
        return PointAnnotationEncoding(
            annotations.size,
            center_yx_offsets,
            tiles.img_idxs,
            yx_idxs_int[:, 0],
            yx_idxs_int[:, 1]
        )

    @torch.no_grad()
    def get_annotation_idxs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.y_idxs, self.x_idxs, self.img_idxs)


@dataclass
class SizeAnnotationEncoding:
    size_hw: torch.Tensor
    has_size_hw: torch.Tensor

    @staticmethod
    @torch.no_grad()
    def encode(annotations: AnnotationBlock, downsample_factor: torch.Tensor) -> SizeAnnotationEncoding:
        assert isinstance(downsample_factor, torch.Tensor)
        assert isinstance(annotations, AnnotationBlock)

        # Cast has_size_hw to int64 and divide size_hw by downsample_factor
        size_hw = annotations.size_hw / downsample_factor
        has_size_hw = annotations.has_size_hw.type(torch.int64)
        return SizeAnnotationEncoding(size_hw, has_size_hw)


@dataclass
class SinCosAnnotationEncoding:
    sincos: torch.Tensor
    has_rotation: torch.Tensor
    is_rotation_360: torch.Tensor

    @staticmethod
    @torch.no_grad()
    def encode(annotations: AnnotationBlock) -> SinCosAnnotationEncoding:
        assert isinstance(annotations, AnnotationBlock)

        # Sin and cos of rotation
        rotation_radians = annotations.rotation * DEVICE_TWO_PI
        sincos = torch.empty((annotations.size, 2), dtype=torch.float64, device=DEVICE)
        torch.sin(rotation_radians, out=sincos[:, 0])
        torch.cos(rotation_radians, out=sincos[:, 1])
        return SinCosAnnotationEncoding(sincos, annotations.has_rotation, annotations.is_rotation_360)


@dataclass
class AngleAnnotationEncoding:
    rotation_180: torch.Tensor
    rotation_360: torch.Tensor
    has_rotation: torch.Tensor
    is_rotation_360: torch.Tensor
    num_with_rotation: torch.Tensor

    @staticmethod
    @torch.no_grad()
    def encode(annotations: AnnotationBlock) -> AngleAnnotationEncoding:
        assert isinstance(annotations, AnnotationBlock)
        rotation_360 = annotations.rotation
        rotation_180 = (rotation_360 % 0.5) * 2
        has_rotation = annotations.has_rotation
        is_rotation_360 = annotations.is_rotation_360
        num_with_rotation = has_rotation.sum().cpu()
        return AngleAnnotationEncoding(rotation_180, rotation_360, has_rotation, is_rotation_360, num_with_rotation)


# def get_hw_annotation_encoding(flat_annotation_list, downsample_factor):
#     # Create a flat vector of height+width values in pixels, and a corresponding vector
#     # of has_hw (0 or 1) values, that denote if the annotated objects have a grund-truth size
#     size_hw_list = []
#     has_size_hw_list = []
#     for annotation in flat_annotation_list:
#         has_size_hw = annotation.size_hw is not None
#         size_hw = annotation.size_hw if has_size_hw else np.array([0,0])
#         size_hw_list.append(size_hw)
#         has_size_hw_list.append(has_size_hw)

#     # Convert to torch tensors and return
#     size_hw_tensor = torch.tensor(size_hw_list, device=DEVICE) / downsample_factor
#     has_size_hw_tensor = torch.tensor(has_size_hw_list, dtype=torch.int64, device=DEVICE)
#     value = (size_hw_tensor, has_size_hw_tensor)
#     return value


# # TODO: Annotation.rotation should maybe just be a float, and not a numpy float.
# # Then we don't need the tensor = torch.tensor(np.array(rotation_list_01))
# # TODO: This belongs in torch_constants.py
# TWO_PI = 2 * 3.14150
# zero_np = np.array(0)
# def get_sincos_annotation_encoding(flat_annotation_list):

#     # Extract rotation values into a torch tensor
#     has_rotation_list = []
#     rotation_list_01 = []
#     for annotation in flat_annotation_list:
#         has_rotation = annotation.rotation is not None
#         rotation = annotation.rotation if has_rotation else zero_np
#         rotation_list_01.append(rotation)
#         has_rotation_list.append(has_rotation)

#     rotation_tensor_01 = torch.tensor(np.array(rotation_list_01), device=DEVICE)
#     has_sincos_tensor = torch.tensor(has_rotation_list, dtype=torch.int64, device=DEVICE)

#     # Convert to radians, and get sin and cos
#     rotation_tensor_radians = rotation_tensor_01 * TWO_PI
#     sin_tensor = torch.sin(rotation_tensor_radians)
#     cos_tensor = torch.cos(rotation_tensor_radians)
#     sincos_tensor = torch.cat(
#         (sin_tensor[:, None], cos_tensor[:, None]),
#         dim=1
#     )
#     num_annotation = len(rotation_list_01)
#     if SAFE_MODE:
#         assert sin_tensor.shape == (num_annotation, )
#         assert cos_tensor.shape == (num_annotation, )
#         assert sincos_tensor.shape == (num_annotation, 2)

#     value = (sincos_tensor, has_sincos_tensor)
#     return value


def get_confidence_bitmap_annotation_encoding(flat_annotation_list):
    """
    Returns a boolean tensor "bitmap", representing which annotations are correct with high confidence.
    Could be replaced with a score between 0 and 1 for varying confidence.
    """
    return torch.tensor(
        [annotation.is_high_confidence for annotation in flat_annotation_list],
        device=DEVICE
    )



class TestPointAnnotationEncoding(unittest.TestCase):
    def assert_tensors_equal(self, a, b, approx=None):
        self.assertEqual(a.shape, b.shape)
        if approx:
            equal = (a - b).abs() < approx
        else:
            equal = a == b
        self.assertTrue(equal.all())

    def make_image(self, shape, value):
        img = torch.tensor(value)[None, None, None].expand(shape)
        return img

    def test_self(self):
        centers = [
            [10, 5],        # --> [0,  0] [0.312500000, 0.15625]
            [40, 40],       # --> [1,  1] [0.240000000, 0.25000]
            [10, 100],      # --> [0,  3] [0.312500000, 0.12500]
            [500, 300],     # --> [15, 9] [0.625000000, 0.37500]
            [32.0001, 200], # --> [1,  6] [0.000003125, 0.25000]
            [33.0001, 200], # --> [1,  6] [0.031253125, 0.25000]
        ]

        annotation_block = AnnotationBlock.from_annotation_list([
            Annotation(center_yx=np.array(center_yx), is_high_confidence=True)
            for center_yx in centers
        ])

        # Input
        downsample_factor = torch.tensor(32)
        tiles = YOLOTileStack(
            self.make_image((4, 1, 100, 100), 1),
            torch.tensor([0, 0, 0, 1, 1, 2]),
            annotation_block,
        )

        # Output
        encoding = PointAnnotationEncoding.encode(tiles, downsample_factor)

        # Expected output
        expected_encoding = PointAnnotationEncoding(
            num_annotations=6,
            center_yx=torch.tensor([
                [0.312500000, 0.15625],
                [0.250000000, 0.25000],
                [0.312500000, 0.12500],
                [0.625000000, 0.37500],
                [0.000003125, 0.25000],
                [0.031253125, 0.25000],
            ]),
            img_idxs=torch.tensor([0, 0, 0, 1, 1, 2]),
            y_idxs=torch.tensor([0, 1, 0, 15, 1, 1]),
            x_idxs=torch.tensor([0, 1, 3,  9, 6, 6]),
        )

        # Check for equality
        self.assertEqual(expected_encoding.num_annotations, encoding.num_annotations)
        self.assert_tensors_equal(expected_encoding.center_yx, encoding.center_yx, approx=0.01)
        self.assert_tensors_equal(expected_encoding.img_idxs, encoding.img_idxs)
        self.assert_tensors_equal(expected_encoding.y_idxs, encoding.y_idxs)
        self.assert_tensors_equal(expected_encoding.x_idxs, encoding.x_idxs)

