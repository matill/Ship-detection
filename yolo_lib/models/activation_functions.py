import torch.nn as nn
import torch


class ActivationFn(nn.Module):
    def __impl_forward__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        y = self.__impl_forward__(x)
        assert y.shape == x.shape
        return y


class ReLU(ActivationFn):
    def __impl_forward__(self, x: torch.Tensor) -> torch.Tensor:
        return x.relu()


class Sigmoid(ActivationFn):
    def __impl_forward__(self, x: torch.Tensor) -> torch.Tensor:
        return x.sigmoid()


class Mish(ActivationFn):
    def __impl_forward__(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(torch.log(1 + torch.exp(x)))

