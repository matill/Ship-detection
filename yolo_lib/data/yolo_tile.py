from __future__ import annotations
import torch
from typing import List
from dataclasses import dataclass
from .annotation import AnnotationBlock


@dataclass
class YOLOTile:
    image: torch.Tensor
    annotations: AnnotationBlock

    def __post_init__(self):
        assert len(self.image.shape) == 4


@dataclass
class YOLOTileStack:
    images: torch.Tensor
    img_idxs: torch.Tensor
    annotations: AnnotationBlock

    @staticmethod
    @torch.no_grad()
    def stack_tiles(tiles: List[YOLOTile]) -> YOLOTileStack:
        annotation_blocks: List[AnnotationBlock] = [tile.annotations for tile in tiles]
        return YOLOTileStack(
            images=torch.cat([tile.image for tile in tiles]),
            img_idxs=YOLOTileStack.get_img_idxs(annotation_blocks),
            annotations=AnnotationBlock.stack_blocks(annotation_blocks)
        )

    @staticmethod
    @torch.no_grad()
    def get_img_idxs(annotation_blocks: List[AnnotationBlock]) -> torch.Tensor:
        return torch.cat([
            torch.tensor(img_idx).expand(annotation_block.size)
            for (img_idx, annotation_block) in enumerate(annotation_blocks)
        ])

    @torch.no_grad()
    def to_device(self, use_gpu: bool) -> YOLOTileStack:
        if use_gpu:
            return YOLOTileStack(
                self.images.cuda(),
                self.img_idxs.cuda(),
                self.annotations.to_device(use_gpu)
            )
        else:
            return self

    def __post_init__(self):
        assert isinstance(self.images, torch.Tensor)
        assert isinstance(self.img_idxs, torch.Tensor)
        assert isinstance(self.annotations, AnnotationBlock)
        assert self.img_idxs.shape == (self.annotations.size, )
        assert self.annotations.center_yx.shape == (self.annotations.size, 2)

    def extract_tile(self, idx: int) -> YOLOTile:
        assert isinstance(idx, int)
        image = self.images[idx][None]
        annotations = self.annotations.extract_bitmap(self.img_idxs == idx)
        return YOLOTile(image, annotations)

    def split_into_tiles(self) -> List[YOLOTile]:
        return [self.extract_tile(i) for i in range(self.images.shape[0])]
