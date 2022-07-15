from __future__ import annotations
import torch
from torch import nn, Tensor
from dataclasses import dataclass
from yolo_lib.cfg import DEVICE
from yolo_lib.detectors.yolo_heads.label_assignment.label_assignment import LabelAssignment
from yolo_lib.models.binary_focal_loss import BinaryFocalLoss


@dataclass
class FocalLossCfg:
    neg_weight: float
    pos_weight: float
    gamma: int

    def build(self) -> ConfidenceUnawareObjectnessLoss:
        return ConfidenceUnawareObjectnessLoss(
            BinaryFocalLoss(self.gamma, self.pos_weight, self.neg_weight)
        )


class ConfidenceUnawareObjectnessLoss(nn.Module):
    def __init__(self, objectness_loss_function):
        super().__init__()
        self.objectness_loss_function = objectness_loss_function

    def forward(
        self,
        pre_activation_o: Tensor,
        assignment: LabelAssignment,
    ):
        # Set up ground-truth objectness probabilities
        ground_truth_objectness_bool = create_binary_mask(assignment, pre_activation_o.shape)

        # Compute loss
        objectness_loss = self.objectness_loss_function(pre_activation_o, ground_truth_objectness_bool)
        return objectness_loss


def create_binary_mask(assignment: LabelAssignment, shape):
    """
    Creates a binary mask where the indices specified in full_head_idxs are 1,
    and the rest are 0
    """
    assert len(shape) == 4
    assert isinstance(assignment, LabelAssignment)

    # Extract four "lists" of indices
    (img_idxs, head_idxs, grid_y_idxs, grid_x_idxs) = assignment.get_grid_idx_vectors()

    # Create a zero-only mask, and set all specified indices to 1
    binary_mask = torch.zeros(shape, dtype=torch.bool, device=DEVICE)
    binary_mask[img_idxs, head_idxs, grid_y_idxs, grid_x_idxs] = True
    return binary_mask

