
from models.base_model import PeriodicRegression
import torch
from torch import Tensor


class PDV(PeriodicRegression):
    PRETTY_NAME = "ADV (mse)"

    def __init__(self):
        super().__init__(2)

    def __loss_impl__(self, predictions: Tensor, labels: Tensor, batch_size: int) -> Tensor:
        assert ((0 <= labels) & (labels <= 1)).all(), labels
        labels_rad = labels * 2 * 3.14159

        # Get predicted sin and cos, and compute MSE
        predictions_sin = predictions[:, 0]
        predictions_cos = predictions[:, 1]
        labels_sin = labels_rad.sin()
        labels_cos = labels_rad.cos()
        loss_sin = (predictions_sin - labels_sin).square()
        loss_cos = (predictions_cos - labels_cos).square()
        loss = loss_sin + loss_cos
        assert loss.shape == (batch_size, )
        return loss.sum()

    def __infer_impl__(self, predictions: Tensor, batch_size: int) -> Tensor:
        # Get predicted sin and cos
        norm: Tensor = predictions.norm(dim=1, keepdim=True)
        assert norm.shape == (batch_size, 1)
        predictions_normalized = predictions / norm
        sin = predictions_normalized[:, 0]
        cos = predictions_normalized[:, 1]

        # Decode sin and cos to get a number in [0, 1] range
        arcsin_radians = sin.arcsin()
        arcsin_01 = arcsin_radians / (2 * 3.14159)
        angles_cos_below_0 = (0.5 - arcsin_01) * (cos < 0)
        angles_cos_above_0 = (arcsin_01 + (sin < 0)) * (cos >= 0)
        angles = angles_cos_above_0 + angles_cos_below_0
        assert angles.shape == (batch_size, )
        return angles


class NormalizePDV(PDV):
    PRETTY_NAME = "ADV (unit)"

    def __loss_impl__(self, predictions: Tensor, labels: Tensor, batch_size: int) -> Tensor:
        assert ((0 <= labels) & (labels <= 1)).all(), labels
        labels_rad = labels * 2 * 3.14159

        # Normalize predictions
        norm: Tensor = predictions.norm(dim=1, keepdim=True)
        assert norm.shape == (batch_size, 1)
        predictions = predictions / norm

        # Get sin and cos, and compute MSE
        predictions_sin = predictions[:, 0]
        predictions_cos = predictions[:, 1]
        labels_sin = labels_rad.sin()
        labels_cos = labels_rad.cos()
        loss_sin = (predictions_sin - labels_sin).square()
        loss_cos = (predictions_cos - labels_cos).square()
        loss = loss_sin + loss_cos
        assert loss.shape == (batch_size, )
        return loss.sum()


class ProjectedPDVBase(PDV):
    WEIGHT_1 = None
    WEIGHT_2 = None

    def __loss_impl__(self, predictions: Tensor, labels: Tensor, batch_size: int) -> Tensor:
        # Sin and cos of target angle
        assert ((0 <= labels) & (labels <= 1)).all(), labels
        labels_rad = labels * 2 * 3.14159
        labels_sin = labels_rad.sin()
        labels_cos = labels_rad.cos()
        assert labels_rad.shape == (batch_size, )
        assert labels_sin.shape == (batch_size, )
        assert labels_cos.shape == (batch_size, )

        # Project predictions to coordinate system where the target and
        # its perpendicular are the unit vectors.
        # projection_1: Projection into the "desired" direction
        # projection_2: Projection into the "undesired" direction
        prediction_1 = predictions[:, 0]
        prediction_2 = predictions[:, 1]
        projection_1 = prediction_1 * labels_sin + prediction_2 * labels_cos
        projection_2 = prediction_1 * labels_cos - prediction_2 * labels_sin
        assert prediction_1.shape == (batch_size, )
        assert prediction_2.shape == (batch_size, )
        assert projection_1.shape == (batch_size, )
        assert projection_2.shape == (batch_size, )

        # Loss:
        # projection_1 should be 1
        # projection_2 should be 0
        weight_1 = self.WEIGHT_1
        weight_2 = self.WEIGHT_1
        loss_1 = (projection_1 - 1).square()
        loss_2 = projection_2.square()
        loss = loss_1.sum() * weight_1 + loss_2.sum() * weight_2
        assert loss_1.shape == (batch_size, )
        assert loss_2.shape == (batch_size, )
        return loss




class ProjectedPDV(ProjectedPDVBase):
    PRETTY_NAME = "ADV (mse-w1)"
    WEIGHT_1 = 0.5
    WEIGHT_2 = 1.5


class ProjectedPDV_175(ProjectedPDVBase):
    PRETTY_NAME = "ADV (mse-w2)"
    WEIGHT_1 = 0.25
    WEIGHT_2 = 1.75
