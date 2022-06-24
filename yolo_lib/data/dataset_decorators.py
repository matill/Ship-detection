from __future__ import annotations
import numpy as np
import torch
import random
import json
from typing import Any, Dict, Iterator, List, Optional
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from yolo_lib.data.string_dataset import StringDataset
from yolo_lib.data.annotation import AnnotationBlock
from yolo_lib.data.yolo_tile import YOLOTile

__all__ = ["MultiProcessTileDataset"]


class Transformer:
    def apply(img: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MultiProcessTileDataset(Dataset):
    def __init__(
        self,
        string_dataset: StringDataset,
        tile_dtype: np.dtype,
        tile_size: int,
        transformer: Transformer
    ):
        self.string_dataset = string_dataset
        self.tile_dtype = tile_dtype
        self.tile_size = tile_size
        self.transformer = transformer

    def store(self, path: str):
        self.string_dataset.store(path)

    @staticmethod
    def try_load(
        path: str,
        tile_dtype: np.dtype,
        tile_size: int,
        transformer: Transformer
    ) -> Optional[MultiProcessTileDataset]:
        string_dataset = StringDataset.load(path)
        if string_dataset is None:
            return None
        else:
            return MultiProcessTileDataset(string_dataset, tile_dtype, tile_size, transformer)

    def __getitem__(self, idx: int) -> YOLOTile:
        # Get file path and annotation JSON list from string dataset
        as_dict = json.loads(self.string_dataset[idx])
        path = as_dict["path"]
        channels: List[str] = as_dict["channels"]
        annotations_dict_list = as_dict["annotations"]

        # Get annotation block
        annotation_block = AnnotationBlock.from_dict_list(annotations_dict_list)

        # Fetch the vv channel of the image file
        c = len(channels)
        shape = (self.tile_size, self.tile_size, c)
        img_memmap = np.memmap(path, dtype=self.tile_dtype, mode="c", shape=shape)
        vv = "vv"
        assert vv in channels
        vv_idx = channels.index(vv)
        img_raw_np = np.array(img_memmap[:, :, vv_idx])[None, :, :] # [1, h, w]
        assert img_raw_np.shape == (1, self.tile_size, self.tile_size) 

        # Apply transformer, and convert to torch tensor
        img_transformed_np = self.transformer.apply(img_raw_np.astype(np.float32))
        img_torch = torch.tensor(img_transformed_np)[None, :, :]
        assert img_torch.shape == (1, 1, self.tile_size, self.tile_size)

        # Return YOLOTile
        return YOLOTile(img_torch, annotation_block)

    def __len__(self) -> int:
        return len(self.string_dataset)


class CatDataset(Dataset):
    def __init__(self, datasets: List[Dataset]) -> None:
        super().__init__()
        self.datasets = datasets


        # Create a matrix:
        # [   0, n(1)]
        # [n(1), n(2)]
        # [n(2), n(3)]
        # where [n(1), n(2), n(3)] is the cumulative-lengths vector for all datasets
        # This way, we can easily check if 

        lengths = np.array([len(ds) for ds in datasets])
        self.idxs = np.cumsum(lengths)
        self.offset_sub = self.idxs - lengths
        self.len = int(self.idxs[len(datasets) - 1])

    def __getitem__(self, idx: int) -> YOLOTile:
        # Index into the self.datasets list
        ds_idx = np.sum(idx >= self.idxs)
        ds = self.datasets[ds_idx]
        
        # Offset into the sub-dataset
        ds_offset = idx - self.offset_sub[ds_idx]
        return ds[ds_offset]

    def __len__(self) -> int:
        return self.len
