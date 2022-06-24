
from models.base_model import PeriodicRegression
import torch
from torch import Tensor

class LogisticRegression(PeriodicRegression):
    PRETTY_NAME = "Logistic regression"

    def __init__(self):
        super().__init__(1)

    def __loss_impl__(self, predictions: Tensor, labels: Tensor, batch_size: int) -> Tensor:
        loss_grid = (predictions.sigmoid() - labels[:, None]).square()
        assert loss_grid.shape == (batch_size, 1), loss_grid.shape
        return loss_grid.sum()

    def __infer_impl__(self, predictions: Tensor, batch_size: int) -> Tensor:
        return predictions.sigmoid()[:, 0]

