

from typing import Dict, List, Optional
from torchvision.models import ResNet, resnet18, resnet34
import torch.nn as nn
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import json


USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"


USE_RESNET_34 = False
if USE_RESNET_34:
    resnet_getter = resnet34
else:
    resnet_getter = resnet18


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet_getter(pretrained=True, progress=False)

    def forward(self, x):
        # "Manually" perform ResNet's forward pass, truncating the last layer
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class PeriodicRegression(nn.Module):
    PRETTY_NAME: str = None

    def __init__(self, out_features: int):
        super().__init__()
        self.backbone = Backbone()
        self.fc = torch.nn.Linear(512, out_features)
        self.out_features = out_features
        self.min_prediction = 1.0
        self.max_prediction = 0.0
        self.min_label = 1.0
        self.max_label = 0.0

    def forward(self, x: Tensor) -> Tensor:
        batch_size, c, h, w = x.shape
        assert (c, h, w) == (3, 512, 512)
        x = self.backbone(x)
        x = self.fc(x)
        assert x.shape == (batch_size, self.out_features)
        return x

    def __loss_impl__(self, predictions: Tensor, labels: Tensor, batch_size: int) -> Tensor:
        raise NotImplementedError

    def __infer_impl__(self, predictions: Tensor, batch_size: int) -> Tensor:
        raise NotImplementedError

    def loss(self, images: Tensor, labels: Tensor) -> Tensor:
        # Input valudation
        batch_size = images.shape[0]
        assert images.shape == (batch_size, 3, 512, 512)
        assert labels.shape == (batch_size, )
        assert ((0 <= labels) & (labels <= 1)).all(), labels
        predictions = self.forward(images)
        assert predictions.shape == (batch_size, self.out_features)
        self.min_label = min(labels.min(), self.min_label)
        self.max_label = max(labels.max(), self.max_label)


        # Output validation
        loss = self.__loss_impl__(predictions, labels, batch_size)
        assert loss.shape == ()
        return loss

    def infer(self, images: Tensor) -> Tensor:
        # Input validation
        batch_size = images.shape[0]
        assert images.shape == (batch_size, 3, 512, 512)
        predictions = self.forward(images)
        assert predictions.shape == (batch_size, self.out_features)

        # Output validation
        out = self.__infer_impl__(predictions, batch_size)
        assert out.shape == (batch_size, )
        assert ((0 <= out) & (out <= 1)).all(), out
        self.min_prediction = min(out.min(), self.min_prediction)
        self.max_prediction = max(out.max(), self.max_prediction)
        return out


