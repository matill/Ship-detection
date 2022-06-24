from typing import Dict, Tuple
from models.base_model import PeriodicRegression
import torch
from torch import nn, Tensor

USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"

GAMMA = 2

def focal_loss(logits: Tensor, ground_truth: Tensor) -> Tensor:
    assert isinstance(logits, Tensor)
    assert isinstance(ground_truth, Tensor)
    assert logits.dtype in [torch.float32, torch.float64]
    assert ground_truth.dtype in [torch.float32, torch.float64]
    assert logits.shape == ground_truth.shape

    pos_exp = 1 + torch.exp(logits)
    neg_exp = 1 + torch.exp(-logits)
    pos_loss = torch.log(neg_exp) * (pos_exp ** -GAMMA)
    neg_loss = torch.log(pos_exp) * (neg_exp ** -GAMMA)

    pos_assignment = ground_truth
    neg_assignment = 1 - ground_truth
    loss = pos_loss * pos_assignment + neg_loss * neg_assignment
    assert loss.shape == logits.shape
    return loss


class ClassificationModelBase(PeriodicRegression):
    def __init__(
        self,
        num_bits: int,
        num_classes: int,
        include_offset_regression: bool,
    ):
        assert isinstance(num_bits, int)
        assert isinstance(num_classes, int)
        assert isinstance(include_offset_regression, bool)
        out_features = num_bits + int(include_offset_regression)
        super().__init__(out_features)
        self.num_bits = num_bits
        self.num_classes = num_classes
        self.include_offset_regression = include_offset_regression

        # Centers of each class
        # range: Index of each class (float)
        # class_centers: Center value of each class
        # thresholds: Min value of each class
        # class_width: The width of the area covered by each class
        range = torch.arange(0, num_classes, 1, dtype=torch.float64, device=DEVICE)
        self.class_centers: Tensor = range / num_classes + 1 / (2 * num_classes)
        self.thresholds = range / self.num_classes
        self.class_width = 1 / num_classes
        assert self.class_centers.shape == (num_classes, )
        assert self.thresholds.shape == (num_classes, )

    def get_predicted_offsets(self, predictions: Tensor, batch_size: int):
        assert self.include_offset_regression
        predicted_offsets = (predictions[:, 0].sigmoid() - 0.5) * self.class_width
        assert predicted_offsets.shape == (batch_size, )
        return predicted_offsets

    def get_bit_logits(self, predictions: Tensor, batch_size: int) -> Tensor:
        if not self.include_offset_regression:
            bit_logits = predictions
        else:
            bit_logits = predictions[:, 1:]

        assert bit_logits.shape == (batch_size, self.num_bits)
        return bit_logits

    def __decode_bits__(self, bit_logits: Tensor, batch_size: int) -> Tensor:
        raise NotImplementedError

    def __classification_loss__(
        self,
        bit_logits: Tensor,
        labels: Tensor,
        batch_size: int
    ) -> Tensor:
        raise NotImplementedError

    def __loss_impl__(self, predictions: Tensor, labels: Tensor, batch_size: int) -> Tensor:
        # predictions: [batch_size, out_features]

        # Get bit_logits and compute classification loss
        bit_logits = self.get_bit_logits(predictions, batch_size)
        classification_loss = self.__classification_loss__(bit_logits, labels, batch_size)
        assert classification_loss.shape == ()

        # If using offset regression, get offset regression loss
        if self.include_offset_regression:
            # Predicted offset
            predicted_offsets = self.get_predicted_offsets(predictions, batch_size)
            assert predicted_offsets.shape == (batch_size, )

            # Correct offset (First get index of correct class, then center of that class)
            label_idxs_bitmap = labels[:, None] > self.thresholds[None, :]
            label_idxs = label_idxs_bitmap.sum(dim=1) - 1
            true_class_centers = self.class_centers[label_idxs]
            true_offsets = labels - true_class_centers
            assert label_idxs_bitmap.shape == (batch_size, self.num_classes)
            assert label_idxs.shape == (batch_size, )
            assert true_class_centers.shape == (batch_size, )
            assert true_offsets.shape == (batch_size, )

            # Offset loss
            offset_loss = (predicted_offsets - true_offsets).square()

            # Return sum of losses
            return offset_loss.sum() + classification_loss
        else:
            # Return classification loss only
            return classification_loss

    def __infer_impl__(self, predictions: Tensor, batch_size: int) -> Tensor:
        assert predictions.shape == (batch_size, self.out_features)

        # Get bit_logits and decode bits to get class indices.
        # Then get the center of the predicted classes
        bit_logits = self.get_bit_logits(predictions, batch_size)
        predicted_class_idxs = self.__decode_bits__(bit_logits, batch_size)
        assert predicted_class_idxs.shape == (batch_size, )
        assert predicted_class_idxs.dtype == torch.int64, predicted_class_idxs.dtype
        predicted_class_centers = self.class_centers[predicted_class_idxs]
        assert predicted_class_centers.shape == (batch_size, )
        assert predicted_class_centers.dtype == torch.float64

        # If using offset regression, add predicted offsets
        if self.include_offset_regression:
            predicted_offsets = self.get_predicted_offsets(predictions, batch_size)
            return (predicted_class_centers + predicted_offsets) % 1.0
        else:
            return predicted_class_centers

