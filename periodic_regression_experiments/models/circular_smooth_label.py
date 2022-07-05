from typing import Dict, Tuple
from models.base_model import PeriodicRegression
import torch
from torch import nn, Tensor

from models.classification_model_base import ClassificationModelBase, focal_loss

USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"

class WindowFn:
    def __get_weights__(self, diff_wrap: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_weights(self, diff_wrap: torch.Tensor) -> torch.Tensor:
        assert isinstance(diff_wrap, torch.Tensor)
        out = self.__get_weights__(diff_wrap)
        assert out.shape == diff_wrap.shape
        return out


class TriangleWindowFn(WindowFn):
    def __init__(self, window_fn_width: float) -> None:
        assert isinstance(window_fn_width, float)
        assert 0.0 <= window_fn_width < 0.5
        self.window_fn_width = window_fn_width

    def __get_weights__(self, diff_wrap: torch.Tensor) -> torch.Tensor:
        return torch.relu(1 - diff_wrap / self.window_fn_width)


class GaussianWindowFn(WindowFn):
    def __init__(self, window_fn_width: float) -> None:
        assert isinstance(window_fn_width, float)
        assert 0.0 <= window_fn_width < 0.5
        self.window_fn_width = window_fn_width

    def __get_weights__(self, diff_wrap: torch.Tensor) -> torch.Tensor:
        diff_wrap / self.window_fn_width
        return torch.exp(-(diff_wrap / (0.5*self.window_fn_width)) ** 2)
        # e ** (-(d / (0.5*r))**2)
        # return torch.relu(1 - diff_wrap / self.window_fn_width)



class CircularSmoothN(ClassificationModelBase):
    def __init__(self, num_bits: int, window_fn: WindowFn, include_offset_regression: bool):
        assert isinstance(num_bits, int) and num_bits > 0
        assert isinstance(window_fn, WindowFn)
        assert isinstance(include_offset_regression, bool)
        num_classes = num_bits
        super().__init__(num_bits, num_classes, include_offset_regression)
        self.window_fn = window_fn

    def __classification_loss__(
        self,
        bit_logits: Tensor,
        labels: Tensor,
        batch_size: int
    ) -> Tensor:

        # Get absolute difference between labels and the class centers.
        diff_nonwrap = torch.abs(labels[:, None] - self.class_centers[None, :])
        diff_wrap = torch.min(diff_nonwrap, 1.0 - diff_nonwrap)
        assert ((0 <= diff_wrap) & (diff_wrap <= 1)).all(), diff_wrap
        assert diff_nonwrap.shape == (batch_size, self.num_bits)
        assert diff_wrap.shape == (batch_size, self.num_bits)

        # Get classification ("ground truth" weights)
        class_weights = self.window_fn.get_weights(diff_wrap)
        assert class_weights.shape == (batch_size, self.num_bits)
        assert ((0 <= class_weights) & (class_weights <= 1)).all(), class_weights

        # Binary cross entropy for each entry in the matrix
        loss_grid = focal_loss(bit_logits, class_weights)
        # loss_grid = torch.binary_cross_entropy_with_logits(bit_logits, class_weights)
        assert loss_grid.shape == (batch_size, self.num_bits)
        return loss_grid.sum()

    def __decode_bits__(self, bit_logits: Tensor, batch_size: int) -> Tensor:
        # Get class index of each prediction
        # bit_logits: [batch_size, num_bits]
        assert bit_logits.shape == (batch_size, self.num_bits)
        predicted_class_idxs = bit_logits.topk(k=1, dim=1).indices
        assert predicted_class_idxs.shape == (batch_size, 1), predicted_class_idxs.shape
        assert predicted_class_idxs.dtype == torch.int64
        return predicted_class_idxs[:, 0]


class CSL_256Bit_03WindowWidth_PlusOffset(CircularSmoothN):
    PRETTY_NAME = "CSL (256 bit, window-radius = 0.3)"

    def __init__(self):
        num_bits = 256
        window_width = 0.3
        include_offset_regression = True
        window_fn = TriangleWindowFn(window_width)
        super().__init__(num_bits, window_fn, include_offset_regression)

class CSL_256Bit_01WindowWidth_PlusOffset(CircularSmoothN):
    PRETTY_NAME = "CSL (256 bit, window-radius = 0.1)"

    def __init__(self):
        num_bits = 256
        window_width = 0.1
        include_offset_regression = True
        window_fn = TriangleWindowFn(window_width)
        super().__init__(num_bits, window_fn, include_offset_regression)

class CSL_128Bit_03WindowWidth_PlusOffset(CircularSmoothN):
    PRETTY_NAME = "CSL (128 bit, window-radius = 0.3)"

    def __init__(self):
        num_bits = 128
        window_width = 0.3
        include_offset_regression = True
        window_fn = TriangleWindowFn(window_width)
        super().__init__(num_bits, window_fn, include_offset_regression)

class CSL_128Bit_01WindowWidth_PlusOffset(CircularSmoothN):
    PRETTY_NAME = "CSL (128 bit, window-radius = 0.1)"

    def __init__(self):
        num_bits = 128
        window_width = 0.1
        include_offset_regression = True
        window_fn = TriangleWindowFn(window_width)
        super().__init__(num_bits, window_fn, include_offset_regression)

class CSL_32Bit_03WindowWidth_PlusOffset(CircularSmoothN):
    PRETTY_NAME = "CSL (32 bit, window-radius = 0.3)"

    def __init__(self):
        num_bits = 32
        window_width = 0.3
        include_offset_regression = True
        window_fn = TriangleWindowFn(window_width)
        super().__init__(num_bits, window_fn, include_offset_regression)

class CSL_32Bit_01WindowWidth_PlusOffset(CircularSmoothN):
    PRETTY_NAME = "CSL (32 bit, window-radius = 0.1)"

    def __init__(self):
        num_bits = 32
        window_width = 0.1
        include_offset_regression = True
        window_fn = TriangleWindowFn(window_width)
        super().__init__(num_bits, window_fn, include_offset_regression)


class GaussianCSLBase(CircularSmoothN):
    NUM_BITS: int = None
    WINDOW_WIDTH: int = None

    def __init__(self):
        include_offset_regression = True
        window_fn = GaussianWindowFn(self.WINDOW_WIDTH)
        super().__init__(self.NUM_BITS, window_fn, include_offset_regression)

class CSL_32Bit_01Gaussian(GaussianCSLBase):
    PRETTY_NAME = "CSL (32 bit, 0.1 radius)"
    NUM_BITS = 32
    WINDOW_WIDTH = 0.1

class CSL_128Bit_01Gaussian(GaussianCSLBase):
    PRETTY_NAME = "CSL (128 bit, 0.1 radius)"
    NUM_BITS = 128
    WINDOW_WIDTH = 0.1

class CSL_256Bit_01Gaussian(GaussianCSLBase):
    PRETTY_NAME = "CSL (256 bit, 0.1 radius)"
    NUM_BITS = 256
    WINDOW_WIDTH = 0.1

class CSL_32Bit_03Gaussian(GaussianCSLBase):
    PRETTY_NAME = "CSL (32 bit, 0.3 radius)"
    NUM_BITS = 32
    WINDOW_WIDTH = 0.3

class CSL_128Bit_03Gaussian(GaussianCSLBase):
    PRETTY_NAME = "CSL (128 bit, 0.3 radius)"
    NUM_BITS = 128
    WINDOW_WIDTH = 0.3

class CSL_256Bit_03Gaussian(GaussianCSLBase):
    PRETTY_NAME = "CSL (256 bit, 0.3 radius)"
    NUM_BITS = 256
    WINDOW_WIDTH = 0.3

