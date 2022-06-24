import torch
import torch.nn as nn

from yolo_lib.cfg import SAFE_MODE


class LeakySigmoid(nn.Module):
    def __init__(self, overflow_amount=0.3) -> None:
        super().__init__()
        self.overflow_amount = overflow_amount
        self.scale_factor = 1.0 + overflow_amount
        self.offset_term = -0.5 * overflow_amount

    def forward(self, input):
        sigmoid = torch.sigmoid(input)
        output = torch.add(self.offset_term, sigmoid, alpha=self.scale_factor)
        if SAFE_MODE:
            assert input.shape == output.shape

        return output

