import torch
from .cfg import DEVICE


DEVICE_4_OVER_PI_SQUARED = torch.tensor(4 / (3.14159 ** 2), device=DEVICE, requires_grad=False)
DEVICE_INV_SQRT2 = 1 / torch.tensor(2.0, device=DEVICE, requires_grad=False).sqrt()
DEVICE_0 = torch.tensor(0, device=DEVICE, requires_grad=False)
DEVICE_1 = torch.tensor(1, device=DEVICE, requires_grad=False)
DEVICE_TWO_PI = torch.tensor(3.14159 * 2, device=DEVICE, requires_grad=False)

