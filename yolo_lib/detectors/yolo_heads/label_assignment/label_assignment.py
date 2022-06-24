from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from typing import List, Tuple
from yolo_lib.cfg import DEVICE
from yolo_lib.util import check_tensor


@dataclass
class LabelAssignment:
    num_assignments: int
    full_head_idxs: Tensor # [num_assignments, 4]
    object_idxs: Tensor # [num_assignments]

    def __post_init__(self):
        assert isinstance(self.num_assignments, int) and self.num_assignments >= 0
        check_tensor(self.full_head_idxs, (self.num_assignments, 4), torch.int64)
        check_tensor(self.object_idxs, (self.num_assignments, ), torch.int64)

    # Returns a reduced version of "self", but indexed according
    # to the provided bitmap. Used for extracting subsets of the assignments
    # where the target/truth has a known rotation, known height/width, etc...
    def extract_bitmap(self, bitmap: Tensor) -> LabelAssignment:
        # Check input validity
        assert isinstance(bitmap, Tensor)
        assert bitmap.shape == (self.num_assignments, )
        assert bitmap.dtype == torch.bool

        # Create new object
        full_head_idxs = self.full_head_idxs[bitmap, :]
        object_idxs = self.object_idxs[bitmap]
        (num_assignments, ) = object_idxs.shape
        assert full_head_idxs.shape == (num_assignments, 4)
        return LabelAssignment(num_assignments, full_head_idxs, object_idxs)

    def get_grid_idx_vectors(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        img_idxs = self.full_head_idxs[:, 0]
        head_idxs = self.full_head_idxs[:, 1]
        grid_y_idxs = self.full_head_idxs[:, 2]
        grid_x_idxs = self.full_head_idxs[:, 3]
        return (img_idxs, head_idxs, grid_y_idxs, grid_x_idxs)

    @staticmethod
    def new(
        num_assignments: int,
        img_idxs: Tensor,
        anchor_idxs: Tensor,
        y_idxs: Tensor,
        x_idxs: Tensor,
        object_idxs: Tensor,
    ) -> LabelAssignment:
        check_tensor(img_idxs, (num_assignments, ), torch.int64)
        check_tensor(anchor_idxs, (num_assignments, ), torch.int64)
        check_tensor(y_idxs, (num_assignments, ), torch.int64)
        check_tensor(x_idxs, (num_assignments, ), torch.int64)
        check_tensor(object_idxs, (num_assignments, ), torch.int64)
        return LabelAssignment(
            num_assignments,
            torch.cat(
                [img_idxs[:, None], anchor_idxs[:, None], y_idxs[:, None], x_idxs[:, None]],
                dim=1
            ),
            object_idxs,
        )

    @staticmethod
    def stack(assignments: List[LabelAssignment]) -> LabelAssignment:
        num_assignments = sum((a.num_assignments for a in assignments))
        if len(assignments) == 0:
            full_head_idxs = torch.empty((0, 4), dtype=torch.int64, device=DEVICE)
            object_idxs = torch.empty((0, ), dtype=torch.int64, device=DEVICE)
        else:
            full_head_idxs = torch.cat([a.full_head_idxs for a in assignments], dim=0)
            object_idxs = torch.cat([a.object_idxs for a in assignments], dim=0)
        return LabelAssignment(num_assignments, full_head_idxs, object_idxs)

