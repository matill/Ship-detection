from __future__ import annotations
import torch
import numpy as np
from typing import Any, Dict, List
from dataclasses import dataclass
from yolo_lib.util.check_tensor import check_tensor
import json


@dataclass
class AnnotationBlock:
    size: int
    center_yx: torch.Tensor
    is_high_confidence: torch.Tensor
    size_hw: torch.Tensor
    has_size_hw: torch.Tensor
    rotation: torch.Tensor
    has_rotation: torch.Tensor
    is_rotation_360: torch.Tensor
    max_class: torch.Tensor
    has_max_class: torch.Tensor

    def __post_init__(self):
        assert isinstance(self.size, int) and self.size >= 0
        assert self.center_yx.shape == (self.size, 2)
        assert self.is_high_confidence.shape == (self.size, )
        assert self.size_hw.shape == (self.size, 2)
        assert self.has_size_hw.shape == (self.size, )
        assert self.rotation.shape == (self.size, )
        assert self.has_rotation.shape == (self.size, )
        assert self.is_rotation_360.shape == (self.size, )
        assert self.max_class.shape == (self.size, )
        assert self.has_max_class.shape == (self.size, )

    @staticmethod
    @torch.no_grad()
    def stack_blocks(annotation_blocks: List[AnnotationBlock]) -> AnnotationBlock:
        return AnnotationBlock(
            size=sum((block.size for block in annotation_blocks)),
            center_yx=torch.cat([block.center_yx for block in annotation_blocks]),
            is_high_confidence=torch.cat([block.is_high_confidence for block in annotation_blocks]),
            size_hw=torch.cat([block.size_hw for block in annotation_blocks]),
            has_size_hw=torch.cat([block.has_size_hw for block in annotation_blocks]),
            rotation=torch.cat([block.rotation for block in annotation_blocks]),
            has_rotation=torch.cat([block.has_rotation for block in annotation_blocks]),
            is_rotation_360=torch.cat([block.is_rotation_360 for block in annotation_blocks]),
            max_class=torch.cat([block.max_class for block in annotation_blocks]),
            has_max_class=torch.cat([block.has_max_class for block in annotation_blocks]),
        )

    @torch.no_grad()
    def to_device(self, use_gpu: bool) -> AnnotationBlock:
        if use_gpu:
            return AnnotationBlock(
                self.size,
                self.center_yx.cuda(),
                self.is_high_confidence.cuda(),
                self.size_hw.cuda(),
                self.has_size_hw.cuda(),
                self.rotation.cuda(),
                self.has_rotation.cuda(),
                self.is_rotation_360.cuda(),
                self.max_class.cuda(),
                self.has_max_class.cuda(),
            )
        else:
            return self

    @torch.no_grad()
    def as_annotation_list(self) -> List[Annotation]:
        center_yx = self.center_yx.cpu().detach().numpy()
        is_high_confidence = self.is_high_confidence.cpu().detach().numpy()
        size_hw = self.size_hw.cpu().detach().numpy()
        has_size_hw = self.has_size_hw.cpu().detach().numpy()
        rotation = self.rotation.cpu().detach().numpy()
        has_rotation = self.has_rotation.cpu().detach().numpy()
        is_rotation_360 = self.is_rotation_360.cpu().detach().numpy()
        max_class = self.max_class.cpu().detach().numpy()
        has_max_class = self.has_max_class.cpu().detach().numpy()
        return [
            Annotation(
                center_yx=center_yx[i],
                is_high_confidence=is_high_confidence[i],
                size_hw=(size_hw[i] if has_size_hw[i] else None),
                rotation=(rotation[i] if has_rotation[i] else None),
                is_rotation_360=is_rotation_360[i],
                max_class=None,
                max_class=(max_class[i] if has_max_class[i] else None),
            )
            for i in range(self.size)
        ]

    @staticmethod
    @torch.no_grad()
    def empty() -> AnnotationBlock:
        return AnnotationBlock(
            size=0,
            center_yx=torch.zeros((0, 2), dtype=torch.float64),
            is_high_confidence=torch.zeros((0, ), dtype=torch.bool),
            size_hw=torch.zeros((0, 2), dtype=torch.float64),
            has_size_hw=torch.zeros((0, ), dtype=torch.bool),
            rotation=torch.zeros((0, ), dtype=torch.float64),
            has_rotation=torch.zeros((0, ), dtype=torch.bool),
            is_rotation_360=torch.zeros((0, ), dtype=torch.bool),
            max_class=torch.zeros((0, 2), dtype=torch.float64),
            has_max_class=torch.zeros((0, ), dtype=torch.bool),
        )

    @staticmethod
    @torch.no_grad()
    def from_annotation_list(annotation_list: List[Annotation]) -> AnnotationBlock:
        size = len(annotation_list)
        if size == 0:
            return AnnotationBlock.empty()
        else:
            return AnnotationBlock(
                size=size,
                center_yx=torch.tensor(np.array([a.center_yx for a in annotation_list])),
                is_high_confidence=torch.tensor([a.is_high_confidence for a in annotation_list]),
                size_hw=torch.tensor(np.array([
                    a.size_hw if a.size_hw is not None else np.array([0.0, 0.0])
                    for a in annotation_list
                ])),
                has_size_hw=torch.tensor([a.size_hw is not None for a in annotation_list]),
                rotation=torch.tensor(np.array([
                    a.rotation if a.rotation is not None else np.array(0.0)
                    for a in annotation_list
                ])),
                has_rotation=torch.tensor([a.rotation is not None for a in annotation_list]),
                is_rotation_360=torch.tensor([a.is_rotation_360 for a in annotation_list]),
                max_class=torch.tensor(np.array([
                    a.max_class if a.max_class is not None else np.array([0.0, 0.0])
                    for a in annotation_list
                ])),
                has_max_class=torch.tensor([a.max_class is not None for a in annotation_list]),
            )

    def index(self, index: Any) -> AnnotationBlock:
        center_yx = self.center_yx[index]
        size = center_yx.shape[0]
        return AnnotationBlock(
            size,
            center_yx,
            self.is_high_confidence[index],
            self.size_hw[index],
            self.has_size_hw[index],
            self.rotation[index],
            self.has_rotation[index],
            self.is_rotation_360[index],
            self.max_class[index],
            self.has_max_class[index],
        )

    @torch.no_grad()
    def extract_bitmap(self, bitmap: torch.Tensor) -> AnnotationBlock:
        """Extract rows from self according to a bitmap"""
        check_tensor(bitmap, (self.size, ), torch.bool)
        new = self.index(bitmap)
        assert new.size <= self.size
        return new

    @torch.no_grad()
    def extract_index_tensor(self, index_tensor: torch.Tensor) -> AnnotationBlock:
        """Extract rows from self according to a 'list' of integer indices"""
        assert isinstance(index_tensor, torch.Tensor)
        assert index_tensor.dtype == torch.int64
        (size, ) = index_tensor.shape
        new = self.index(index_tensor)
        assert new.size == size
        return new

    def to_dict_list(self) -> List[Dict[str, Any]]:
        annotation_list = self.as_annotation_list()
        return [a.to_dict() for a in annotation_list]

    @staticmethod
    def from_dict_list(dict_list: List[Dict[str, Any]]) -> AnnotationBlock:
        annotation_list = [Annotation.from_dict(as_dict) for as_dict in dict_list]
        return AnnotationBlock.from_annotation_list(annotation_list)

    def to_str(self) -> str:
        dict_list = self.to_dict_list()
        return json.dumps(dict_list, indent=None)

    @staticmethod
    def from_str(as_str: str) -> AnnotationBlock:
        assert isinstance(as_str, str)
        return AnnotationBlock.from_dict_list(json.loads(as_str))


@dataclass
class Annotation:
    """
    A "ground-truth" annotation dataclass
    center_yx: Center position of a vessel, in pixels
    is_high_confidence: True if the vessel has an assumed ~100% confidence of being a true positive
        (ex. if correlated with AES data)
    size_hw: Length and width of a vessel. Not available for all annotations. TODO: Define the unit of this one.
    rotation: A number between 0 and 1. In [0, 1) range. Not available for all annotations
        0.000 --> Pointing upwards
        0.250 --> Pointing right
        0.500 --> Pointing downwards
        0.750 --> Pointing left
        0.999 --> Also pointing upwards ;-)
    max_class: An integer, representing a class index. Not available for all annotations
    """
    center_yx: np.ndarray
    is_high_confidence: bool
    size_hw: np.ndarray = None
    rotation: np.ndarray = None
    is_rotation_360: bool = True
    max_class: int = None

    def __post_init__(self):
        if self.rotation is not None:
            assert 0.0 <= self.rotation <= 1.0, f"Annotation.rotation out of range. Got ({self.rotation}). Out of [0, 1) range"

    def to_dict(self) -> Dict[str, Any]:
        as_dict = {"yx": [float(x) for x in self.center_yx]}
        if self.size_hw is not None:
            as_dict["hw"] = [float(x) for x in self.size_hw]
        if self.rotation is not None:
            as_dict["r"] = float(self.rotation)
        if self.max_class is not None:
            as_dict["c"] = float(self.max_class)
        return as_dict

    @staticmethod
    def from_dict(as_dict: Dict[str, Any]) -> Annotation:
        return Annotation(
            center_yx=np.array(as_dict["yx"], dtype=np.float32),
            size_hw=(np.array(as_dict["hw"], dtype=np.float32) if "hw" in as_dict else None),
            rotation=(np.array(as_dict["r"], dtype=np.float32) if "r" in as_dict else None),
            is_high_confidence=True,
            is_rotation_360=True,
            max_class=(np.array(as_dict["c"], dtype=np.float32) if "c" in as_dict else None),
        )

