import json
from typing import List
import torch
from torch.utils.data.dataset import Dataset
from torch import Tensor
from yolo_lib.data.string_dataset import StringDataset
from yolo_lib.data.yolo_tile import YOLOTile
from yolo_lib.data.annotation import AnnotationBlock


class YOLOTileDataset(Dataset):
    def get_annotations(self, idx: int) -> AnnotationBlock:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> YOLOTile:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class AnnotationDataset(Dataset):
    def __getitem__(self, idx: int) -> AnnotationBlock:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class ImageDataset(Dataset):
    def __getitem__(self, idx: int) -> Tensor:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class AnnotationDatasetString(AnnotationDataset):
    def __init__(self, annotations: List[AnnotationBlock]) -> None:
        self.string_dataset = StringDataset([a.to_str() for a in annotations])

    def __getitem__(self, idx: int) -> AnnotationBlock:
        return AnnotationBlock.from_str(self.string_dataset[idx])

    def __len__(self) -> int:
        return len(self.string_dataset)


class ImageDatasetStacked(ImageDataset):
    def __init__(self, image_list: List[Tensor], h: int, w: int, c: int) -> None:
        for image in image_list:
            assert image.shape == (1, c, h, w)

        self.image_stack = torch.cat(image_list, dim=0)
        self.h = h
        self.w = w
        self.c = c
        self.n = len(image_list)

    def __getitem__(self, idx: int) -> Tensor:
        return self.image_stack[idx][None]

    def __len__(self) -> int:
        return self.n


class YOLOTileDatasetImgPlusAnnotation(YOLOTileDataset):
    def __init__(self, image_dataset: ImageDataset, annotation_dataset: AnnotationDataset) -> None:
        assert isinstance(image_dataset, ImageDataset), type(image_dataset)
        assert isinstance(annotation_dataset, AnnotationDataset)
        assert len(image_dataset) == len(annotation_dataset)
        self.image_dataset = image_dataset
        self.annotation_dataset = annotation_dataset

    def get_annotations(self, idx: int) -> AnnotationBlock:
        return self.annotation_dataset[idx]

    def __getitem__(self, idx: int) -> YOLOTile:
        return YOLOTile(self.image_dataset[idx], self.annotation_dataset[idx])

    def __len__(self) -> int:
        return len(self.annotation_dataset)