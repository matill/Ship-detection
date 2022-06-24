from __future__ import annotations
from dataclasses import dataclass
from torch import Tensor, nn
from torchvision.models import ResNet, resnet18, resnet34, resnet50
from typing import Callable, Dict


@dataclass
class BackboneCfg:
    in_channels: int
    variation: int

    def __post_init__(self):
        assert self.variation in [18, 34, 50]

    def build(self) -> BackBone:
        specs: Dict[int, BackBone] = {
            18: ResNet18,
            34: ResNet34,
            50: ResNet50,
        }
        ResNetClass: BackBone = specs[self.variation]
        return ResNetClass(self.in_channels)


class BackBone(nn.Module):
    RESNET_GETTER = None
    OUT_CHANNELS: int = None
    DOWNSAMPLE_FACTOR = None

    def __init__(self, in_channels: int):
        # Construct a pre-trained ResNet and replace the first convolutional layer
        # to have the right input channels
        super().__init__()
        self.resnet: ResNet = self.RESNET_GETTER(True)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.in_channels = in_channels
        self.out_channels = self.OUT_CHANNELS
        self.downsample_factor = self.DOWNSAMPLE_FACTOR

    def forward(self, x: Tensor) -> Tensor:
        (batch, in_channels, h, w) = x.shape
        assert in_channels == self.in_channels
        # Manually perform ResNet's forward pass, truncating the last layer
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        # x = self.resnet.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.resnet.fc(x)
        assert x.shape == (batch, self.OUT_CHANNELS, int(h/32), int(w/32))
        return x


class ResNet18(BackBone):
    RESNET_GETTER = resnet18
    OUT_CHANNELS = 512
    DOWNSAMPLE_FACTOR = 32


class ResNet34(BackBone):
    RESNET_GETTER = resnet34
    OUT_CHANNELS = 512
    DOWNSAMPLE_FACTOR = 32


class ResNet50(BackBone):
    RESNET_GETTER = resnet50
    OUT_CHANNELS = 2048
    DOWNSAMPLE_FACTOR = 32


