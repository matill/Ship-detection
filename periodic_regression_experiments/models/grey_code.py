from typing import Dict, List, Tuple
import torch
from torch import nn, Tensor

# from models.base_model import PeriodicRegression
from models.classification_model_base import ClassificationModelBase, focal_loss

USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"


class GreyCodeN(ClassificationModelBase):
    def __init__(
        self,
        num_bits: int,
        encodings: Tensor,
        include_offset_regression: bool
    ):
        num_classes = 2 ** num_bits
        super().__init__(num_bits, num_classes, include_offset_regression)
        self.encodings = encodings

        self.decoder = {}
        for (class_idx, encoding) in enumerate(self.encodings):
            assert encoding.shape == (num_bits, )
            encoding_tuple = tuple((int(x)) for x in encoding)
            self.decoder[encoding_tuple] = class_idx

    def __classification_loss__(
        self,
        bit_logits: Tensor,
        labels: Tensor,
        batch_size: int
    ) -> Tensor:
        assert bit_logits.shape == (batch_size, self.num_bits)
        assert labels.shape == (batch_size, )

        # Get grey encoding labels [batch_size, n]
        label_idxs_bitmap = labels[:, None] > self.thresholds[None, :]
        label_idxs = label_idxs_bitmap.sum(dim=1) - 1
        grey_encoding_labels = self.encodings[label_idxs]
        assert label_idxs_bitmap.shape == (batch_size, self.num_classes)
        assert label_idxs.shape == (batch_size, )
        assert grey_encoding_labels.shape == (batch_size, self.num_bits)

        # Compute predicted encoding [batch_size, 3]
        assert bit_logits.shape == (batch_size, self.num_bits)
        loss = focal_loss(bit_logits, grey_encoding_labels.float())
        # loss = nn.BCEWithLogitsLoss(reduction="none").forward(bit_logits, grey_encoding_labels.float())
        assert loss.shape == (batch_size, self.num_bits)
        return loss.sum()

    def __decode_bits__(self, bit_logits: Tensor, batch_size: int) -> Tensor:
        predicted_class_idxs = []
        encoding_bits = bit_logits >= 0.0
        for encoded_vec in encoding_bits:
            encoded_tuple = tuple((int(x) for x in encoded_vec))
            predicted_class_idxs.append(self.decoder[encoded_tuple])

        return torch.tensor(predicted_class_idxs)


class GreyCode3PlusOffset(GreyCodeN):
    PRETTY_NAME = "GCL (3 bit)"

    def __init__(self):
        encodings = torch.tensor(GREYCODE_3, dtype=torch.bool, device=DEVICE)
        super().__init__(3, encodings, True)


class GreyCode5PlusOffset(GreyCodeN):
    PRETTY_NAME = "GCL (5 bit)"
    def __init__(self):
        encodings = torch.tensor(GREYCODE_5, dtype=torch.bool, device=DEVICE)
        super().__init__(5, encodings, True)


class GreyCode7PlusOffset(GreyCodeN):
    PRETTY_NAME = "GCL (7 bit)"
    def __init__(self):
        encodings = torch.tensor(GREYCODE_5, dtype=torch.bool, device=DEVICE)
        super().__init__(5, encodings, True)


GREYCODE_3 = [
    [0,0,0],
    [1,0,0],
    [1,1,0],
    [0,1,0],
    [0,1,1],
    [1,1,1],
    [1,0,1],
    [0,0,1],
]
GREYCODE_5 = [
    [0,0,0,0,0],
    [0,0,0,0,1],
    [0,0,0,1,1],
    [0,0,0,1,0],
    [0,0,1,1,0],
    [0,0,1,1,1],
    [0,0,1,0,1],
    [0,0,1,0,0],
    [0,1,1,0,0],
    [0,1,1,0,1],
    [0,1,1,1,1],
    [0,1,1,1,0],
    [0,1,0,1,0],
    [0,1,0,1,1],
    [0,1,0,0,1],
    [0,1,0,0,0],
    [1,1,0,0,0],
    [1,1,0,0,1],
    [1,1,0,1,1],
    [1,1,0,1,0],
    [1,1,1,1,0],
    [1,1,1,1,1],
    [1,1,1,0,1],
    [1,1,1,0,0],
    [1,0,1,0,0],
    [1,0,1,0,1],
    [1,0,1,1,1],
    [1,0,1,1,0],
    [1,0,0,1,0],
    [1,0,0,1,1],
    [1,0,0,0,1],
    [1,0,0,0,0],
]
GREYCODE_7 = [
    [0,0,0,0,0,0,1],
    [0,0,0,0,0,1,1],
    [0,0,0,0,0,1,0],
    [0,0,0,0,1,1,0],
    [0,0,0,0,1,1,1],
    [0,0,0,0,1,0,1],
    [0,0,0,0,1,0,0],
    [0,0,0,1,1,0,0],
    [0,0,0,1,1,0,1],
    [0,0,0,1,1,1,1],
    [0,0,0,1,1,1,0],
    [0,0,0,1,0,1,0],
    [0,0,0,1,0,1,1],
    [0,0,0,1,0,0,1],
    [0,0,0,1,0,0,0],
    [0,0,1,1,0,0,0],
    [0,0,1,1,0,0,1],
    [0,0,1,1,0,1,1],
    [0,0,1,1,0,1,0],
    [0,0,1,1,1,1,0],
    [0,0,1,1,1,1,1],
    [0,0,1,1,1,0,1],
    [0,0,1,1,1,0,0],
    [0,0,1,0,1,0,0],
    [0,0,1,0,1,0,1],
    [0,0,1,0,1,1,1],
    [0,0,1,0,1,1,0],
    [0,0,1,0,0,1,0],
    [0,0,1,0,0,1,1],
    [0,0,1,0,0,0,1],
    [0,0,1,0,0,0,0],
    [0,1,1,0,0,0,0],
    [0,1,1,0,0,0,1],
    [0,1,1,0,0,1,1],
    [0,1,1,0,0,1,0],
    [0,1,1,0,1,1,0],
    [0,1,1,0,1,1,1],
    [0,1,1,0,1,0,1],
    [0,1,1,0,1,0,0],
    [0,1,1,1,1,0,0],
    [0,1,1,1,1,0,1],
    [0,1,1,1,1,1,1],
    [0,1,1,1,1,1,0],
    [0,1,1,1,0,1,0],
    [0,1,1,1,0,1,1],
    [0,1,1,1,0,0,1],
    [0,1,1,1,0,0,0],
    [0,1,0,1,0,0,0],
    [0,1,0,1,0,0,1],
    [0,1,0,1,0,1,1],
    [0,1,0,1,0,1,0],
    [0,1,0,1,1,1,0],
    [0,1,0,1,1,1,1],
    [0,1,0,1,1,0,1],
    [0,1,0,1,1,0,0],
    [0,1,0,0,1,0,0],
    [0,1,0,0,1,0,1],
    [0,1,0,0,1,1,1],
    [0,1,0,0,1,1,0],
    [0,1,0,0,0,1,0],
    [0,1,0,0,0,1,1],
    [0,1,0,0,0,0,1],
    [0,1,0,0,0,0,0],
    [1,1,0,0,0,0,0],
    [1,1,0,0,0,0,1],
    [1,1,0,0,0,1,1],
    [1,1,0,0,0,1,0],
    [1,1,0,0,1,1,0],
    [1,1,0,0,1,1,1],
    [1,1,0,0,1,0,1],
    [1,1,0,0,1,0,0],
    [1,1,0,1,1,0,0],
    [1,1,0,1,1,0,1],
    [1,1,0,1,1,1,1],
    [1,1,0,1,1,1,0],
    [1,1,0,1,0,1,0],
    [1,1,0,1,0,1,1],
    [1,1,0,1,0,0,1],
    [1,1,0,1,0,0,0],
    [1,1,1,1,0,0,0],
    [1,1,1,1,0,0,1],
    [1,1,1,1,0,1,1],
    [1,1,1,1,0,1,0],
    [1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,0,1],
    [1,1,1,1,1,0,0],
    [1,1,1,0,1,0,0],
    [1,1,1,0,1,0,1],
    [1,1,1,0,1,1,1],
    [1,1,1,0,1,1,0],
    [1,1,1,0,0,1,0],
    [1,1,1,0,0,1,1],
    [1,1,1,0,0,0,1],
    [1,1,1,0,0,0,0],
    [1,0,1,0,0,0,0],
    [1,0,1,0,0,0,1],
    [1,0,1,0,0,1,1],
    [1,0,1,0,0,1,0],
    [1,0,1,0,1,1,0],
    [1,0,1,0,1,1,1],
    [1,0,1,0,1,0,1],
    [1,0,1,0,1,0,0],
    [1,0,1,1,1,0,0],
    [1,0,1,1,1,0,1],
    [1,0,1,1,1,1,1],
    [1,0,1,1,1,1,0],
    [1,0,1,1,0,1,0],
    [1,0,1,1,0,1,1],
    [1,0,1,1,0,0,1],
    [1,0,1,1,0,0,0],
    [1,0,0,1,0,0,0],
    [1,0,0,1,0,0,1],
    [1,0,0,1,0,1,1],
    [1,0,0,1,0,1,0],
    [1,0,0,1,1,1,0],
    [1,0,0,1,1,1,1],
    [1,0,0,1,1,0,1],
    [1,0,0,1,1,0,0],
    [1,0,0,0,1,0,0],
    [1,0,0,0,1,0,1],
    [1,0,0,0,1,1,1],
    [1,0,0,0,1,1,0],
    [1,0,0,0,0,1,0],
    [1,0,0,0,0,1,1],
    [1,0,0,0,0,0,1],
    [1,0,0,0,0,0,0],
]