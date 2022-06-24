import torch
import torch.nn as nn
from yolo_lib.cfg import DEVICE
from yolo_lib.detectors.yolo_heads.label_assigner.label_assigner import LabelAssignment


class ConfidenceUnawareObjectnessLoss(nn.Module):
    def __init__(self, objectness_loss_function):
        super().__init__()
        self.objectness_loss_function = objectness_loss_function

    def forward(
        self,
        pre_activation_o: torch.Tensor,
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

