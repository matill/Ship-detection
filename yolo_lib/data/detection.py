from __future__ import annotations
import torch
import numpy as np
from typing import Any, List, Optional, Tuple
from dataclasses import dataclass
from yolo_lib.util.check_tensor import check_tensor


@dataclass
class DetectionBlock:
    size: int
    center_yx: torch.Tensor
    objectness: torch.Tensor
    size_hw: None | torch.Tensor
    rotation: None | torch.Tensor
    class_probabilities: None | torch.Tensor

    def __post_init__(self):
        assert isinstance(self.size, int)
        assert self.center_yx.shape == (self.size, 2)
        assert self.objectness.shape == (self.size, )
        assert self.size_hw is None or self.size_hw.shape == (self.size, 2)
        assert self.rotation is None or self.rotation.shape == (self.size, )
        if self.class_probabilities is not None:
            num_classes = int(self.class_probabilities.shape[1])
            assert self.class_probabilities.shape == (self.size, num_classes)

    @torch.no_grad()
    def as_detection_list(self) -> List[Detection]:
        center_yx = self.center_yx.cpu().detach().numpy()
        objectness = self.objectness.cpu().detach().numpy()
        size_hw = None if self.size_hw is None else self.size_hw.cpu().detach().numpy()
        rotation = None if self.rotation is None else self.rotation.cpu().detach().numpy()
        class_probabilities = None if self.class_probabilities is None else self.class_probabilities.cpu().detach().numpy()
        max_class = None if self.class_probabilities is None else self.class_probabilities.argmax(1)
        if max_class is not None:
            assert max_class.shape == (self.size, )
        return [
            Detection(
                center_yx=center_yx[i],
                objectness=objectness[i],
                size_hw=(None if size_hw is None else size_hw[i]),
                rotation=(None if rotation is None else rotation[i]),
                class_probabilities=(None if class_probabilities is None else class_probabilities[i]),
                max_class=(None if max_class is None else max_class[i]),
            )
            for i in range(self.size)
        ]

    def index(self, index: Any) -> DetectionBlock:
        center_yx = self.center_yx[index]
        objectness = self.objectness[index]
        size_hw = None if self.size_hw is None else self.size_hw[index]
        rotation = None if self.rotation is None else self.rotation[index]
        class_probabilities = None if self.class_probabilities is None else self.class_probabilities[index]
        size = center_yx.shape[0]
        return DetectionBlock(size, center_yx, objectness, size_hw, rotation, class_probabilities)

    @torch.no_grad()
    def extract_bitmap(self, bitmap: torch.Tensor) -> DetectionBlock:
        """Extract rows from self according to a bitmap"""
        check_tensor(bitmap, (self.size, ), torch.bool)
        new = self.index(bitmap)
        assert new.size <= self.size
        return new

    @torch.no_grad()
    def extract_index_tensor(self, index_tensor: torch.Tensor) -> DetectionBlock:
        """Extract rows from self according to a 'list' of integer indices"""
        assert isinstance(index_tensor, torch.Tensor)
        assert index_tensor.dtype == torch.int64
        (size, ) = index_tensor.shape
        new = self.index(index_tensor)
        assert new.size == size
        return new

    def filter_min_positivity(self, min_positivity: float) -> DetectionBlock:
        """Return only detections that have confidence score above threshold"""
        return self.extract_bitmap(self.objectness >= min_positivity)

    def get_top_n(self, max_detections: int) -> DetectionBlock:
        if self.size <= max_detections:
            return self
        else:
            topk_idxs = self.objectness.topk(max_detections).indices
            check_tensor(topk_idxs, (max_detections, ))
            return self.extract_index_tensor(topk_idxs)

    def order_by_objectness(self, max_detections: int) -> DetectionBlock:
        max_detections = min(self.size, max_detections)
        topk_idxs = self.objectness.topk(max_detections).indices
        check_tensor(topk_idxs, (max_detections, ))
        return self.extract_index_tensor(topk_idxs)


@dataclass
class DetectionGrid:
    """
    Similar to DetectionBlock, but including all detections in a YOLO
    grid, without applying a minimum objectness threshold.
    """
    size: Tuple[int, int, int, int]
    center_yx: torch.Tensor
    objectness: torch.Tensor
    size_hw: None | torch.Tensor
    rotation: None | torch.Tensor
    class_probabilities: None | torch.Tensor

    @staticmethod
    def new(
        size: Tuple[int, int, int, int],
        center_yx: torch.Tensor,
        objectness: torch.Tensor,
        size_hw: None | torch.Tensor,
        rotation: None | torch.Tensor,
        class_probabilities: None | torch.Tensor,
    ) -> DetectionGrid:
        return DetectionGrid(
            size,
            DetectionGrid._permute(center_yx),
            objectness,
            DetectionGrid._permute(size_hw) if size_hw is not None else None,
            rotation,
            DetectionGrid._permute(class_probabilities) if class_probabilities is not None else None,
        )

    @staticmethod
    def _permute(tensor: torch.Tensor) -> torch.Tensor:
        # return tensor.permute((2, 3)).permute(3, 4)
        return tensor.permute((0, 1, 3, 4, 2))

    def __post_init__(self):
        batch_size, num_anchors, h, w = self.size
        check_tensor(self.center_yx, (batch_size, num_anchors, h, w, 2))
        check_tensor(self.objectness, (batch_size, num_anchors, h, w))
        if self.size_hw is not None:
            check_tensor(self.size_hw, (batch_size, num_anchors, h, w, 2))
        if self.rotation is not None:
            check_tensor(self.rotation, (batch_size, num_anchors, h, w))
        if self.class_probabilities is not None:
            num_classes = self.class_probabilities.shape[4]
            check_tensor(self.class_probabilities, (batch_size, num_anchors, h, w, num_classes))

    def _flatten(self, grid: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if grid is None:
            return None
        else:
            return grid.flatten(start_dim=0, end_dim=3)

    def as_detection_block(self) -> DetectionBlock:
        """Flatten to a DetectionBlock"""
        batch_size, num_anchors, h, w = self.size
        assert batch_size == 1, f"Cannot convert DetectionGrid to DetectionBlock if it contains detections from more than one image"
        return DetectionBlock(
            size = num_anchors * h * w,
            center_yx = self._flatten(self.center_yx),
            objectness = self._flatten(self.objectness),
            size_hw = self._flatten(self.size_hw),
            rotation = self._flatten(self.rotation),
            class_probabilities = self._flatten(self.class_probabilities),
        )

    def _index_into_tensor_by_image(self, tensor: Optional[torch.Tensor], img_idx: int) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        else:
            return tensor[[img_idx]]

    def index_by_image(self, img_idx: int) -> DetectionGrid:
        _, num_anchors, h, w = self.size
        return DetectionGrid(
            size=(1, num_anchors, h, w),
            center_yx=self._index_into_tensor_by_image(self.center_yx, img_idx),
            objectness=self._index_into_tensor_by_image(self.objectness, img_idx),
            size_hw=self._index_into_tensor_by_image(self.size_hw, img_idx),
            rotation=self._index_into_tensor_by_image(self.rotation, img_idx),
            class_probabilities=self._index_into_tensor_by_image(self.class_probabilities, img_idx),
        )

    def split_by_image(self) -> List[DetectionGrid]:
        """Split the DetectionGrid into N parts, for N images"""
        batch_size, _, _, _ = self.size
        return [self.index_by_image(img_idx) for img_idx in range(batch_size)]


@dataclass
class Detection:
    """
    A predicted detection dataclass
    center_yx: 2d vector (y, x) representing pixel offset from the top-left corner of the image.
    objectness: Confidence in the object, between 0 and 1.
    size_hw: 2d vector (h, w) representing height and width of object in pixels.
    class_probabilities: C-dimensional vector: One-hot encoded.
    max_class: Integer in range(0, C), representing a hard class.
    """
    center_yx: np.ndarray
    objectness: float
    size_hw: np.ndarray = None
    rotation: np.ndarray = None
    class_probabilities: np.ndarray = None
    max_class: int = None

    def __post_init__(self):
        print(self)
        assert 0 <= self.objectness <= 1
        if self.class_probabilities is not None:
            assert abs(self.class_probabilities.sum() - 1) < 0.02, f"Detection.class_probabilities should sum to 1. Got ({float(self.class_probabilities.sum())})"


