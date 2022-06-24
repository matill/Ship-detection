import torch

from yolo_lib.detectors.yolo_heads.annotation_encoding import PointAnnotationEncoding
from .label_assigner import LabelAssigner, LabelAssignment
from yolo_lib.cfg import DEVICE, SAFE_MODE


class TopK(LabelAssigner):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    @torch.no_grad()
    def assign(
        self,
        match_loss: torch.Tensor,
        yx_annotation_encoding: PointAnnotationEncoding,
    ) -> LabelAssignment:
        assert isinstance(match_loss, torch.Tensor)
        assert isinstance(yx_annotation_encoding, PointAnnotationEncoding)

        # Get the y, x and image indexes of the objects
        (y_idxs, x_idxs, img_idxs) = yx_annotation_encoding.get_annotation_idxs()
        num_objects = y_idxs.shape[0]

        # For each object, get the index of the top-scoring head
        _topk_losses, topk_idxs = torch.topk(
            match_loss,
            k=self.k,
            largest=False,
            sorted=False,
            dim=1
        )

        # topk_head_idxs[i] = [head_idx]
        topk_head_idxs = topk_idxs.flatten()

        # idx_grid_repeated[i] = [img_idx, y_idx, x_idx]
        # Same as idx_grid, but each row is repeated k times
        # TODO: replace unsqueeze
        idx_grid = torch.cat(
            [img_idxs.unsqueeze(1), y_idxs.unsqueeze(1), x_idxs.unsqueeze(1)],
            dim=1
        )
        idx_grid_repeated_u = idx_grid.unsqueeze(1).repeat(1, self.k, 1)
        idx_grid_repeated = idx_grid_repeated_u.reshape((num_objects * self.k, 3))
        if SAFE_MODE:
            assert list(topk_idxs.shape) == [num_objects, self.k]
            assert list(topk_head_idxs.shape) == [num_objects * self.k]
            assert list(idx_grid.shape) == [num_objects, 3]
            assert list(idx_grid_repeated_u.shape) == [num_objects, self.k, 3]
            assert list(idx_grid_repeated.shape) == [num_objects * self.k, 3]

        # full_head_idxs[i] = [img_idx, head_idx, grid_idx_y, grid_idx_x]
        full_head_idxs = torch.cat(
            [
                idx_grid_repeated[:, 0].unsqueeze(1),
                topk_head_idxs.unsqueeze(1),
                idx_grid_repeated[:, 1].unsqueeze(1),
                idx_grid_repeated[:, 2].unsqueeze(1)
            ],
            dim=1
        )

        object_idxs = torch.arange(0, num_objects, dtype=torch.int64, device=DEVICE)
        object_idxs = object_idxs.unsqueeze(1)
        object_idxs = object_idxs.repeat(1, self.k)
        object_idxs = object_idxs.flatten()

        return LabelAssignment(
            num_assignments=num_objects * self.k,
            full_head_idxs=full_head_idxs,
            object_idxs=object_idxs,
        )

