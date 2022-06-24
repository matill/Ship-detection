from __future__ import annotations
import numpy as np
import torch
import random
import json
from typing import Any, Dict, Iterator, List, Optional
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from yolo_lib.data.map_datasets import YOLOTileDataset
from yolo_lib.data.string_dataset import StringDataset
from yolo_lib.data.annotation import AnnotationBlock
from yolo_lib.data.yolo_tile import YOLOTile

__all__ = ["RandomPartitionSampler", "MultiProcessTileDataset"]


class RandomPartitionSampler(Sampler):
    """
    usage:
    -----------------------------
    elements_per_epoch = 12_800
    train_dl = DataLoader(
        train_ds,
        sampler=RandomPartitionSampler(
            len(train_ds),
            elements_per_epoch
        )
    )
    ------------------------------
    Samples data points randomly in "epochs" of reduced size. This makes it easier
    to compute performance metrics within epochs, and adjust learning rate and other
    parameters after some number of iterations.
    By setting elements_per_epoch=12_800, you get 12_800 random tiles in each epoch.
    This corresponds to 400 batches of 32 tiles.
    These tiles do not repeat until all tiles in the dataset have been sampled once.
    """

    def __init__(self, dataset_size: int, elements_per_epoch: int) -> None:
        assert elements_per_epoch <= dataset_size
        self.elements_per_epoch = elements_per_epoch
        self.dataset_size = dataset_size
        self.num_partition = int(self.dataset_size / self.elements_per_epoch)
        self.order = self.get_order()

    def get_order(self) -> List[List[int]]:
        # Get a non-repeating list of all indices
        flat_shuffled_order = list(range(self.dataset_size))
        random.shuffle(flat_shuffled_order)

        # Group in lists of elements_per_epoch indices, and drop the remainder
        structured_order = []
        for i in range(self.num_partition):
            start_i = i * self.elements_per_epoch
            end_i = start_i + self.elements_per_epoch
            structured_order.append(flat_shuffled_order[start_i:end_i])

        return structured_order

    def __iter__(self) -> Iterator[int]:
        if len(self.order) == 0:
            self.order = self.get_order()
        return self.order.pop().__iter__()

    def __len__(self) -> int:
        return self.elements_per_epoch


class BalancedSampler(Sampler):
    """
    Similar to RandomPartitionSampler, but enforces a consistent balance between
    positive and negative tiles.
    """

    def __init__(
        self,
        negatives_per_positive: float,
        epoch_size: int,
        dataset: YOLOTileDataset
    ) -> None:
        self.negatives_per_positive = negatives_per_positive
        self.epoch_size = epoch_size

        # Index over positive and negative tiles in the dataset
        self.positives: List[int] = []
        self.negatives: List[int] = []
        for i in range(len(dataset)):
            if dataset.get_annotations(i).size > 0:
                self.positives.append(i)
            else:
                self.negatives.append(i)

        # Number of positive and negative samples per epoch
        self.positives_per_epoch = int(epoch_size / (1 + negatives_per_positive))
        self.negatives_per_epoch = int(epoch_size - self.positives_per_epoch)
        assert self.positives_per_epoch + self.negatives_per_epoch == epoch_size
        assert 0 < self.negatives_per_epoch <= len(self.negatives), f"Needs {self.negatives_per_epoch} negatives per epoch, but only {len(self.negatives)} negatives in dataset"
        assert 0 < self.positives_per_epoch <= len(self.positives), f"Needs {self.positives_per_epoch} positives per epoch, but only {len(self.positives)} positives in dataset"

        # Random samplers for positives and negatives
        self.positive_sampler = RandomPartitionSampler(len(self.positives), self.positives_per_epoch)
        self.negative_sampler = RandomPartitionSampler(len(self.negatives), self.negatives_per_epoch)
        print("positives_per_epoch", self.positives_per_epoch)
        print("negatives_per_epoch", self.negatives_per_epoch)

    def __iter__(self) -> Iterator[int]:
        # Get lists of positive and negative 
        positives = [self.positives[i] for i in self.positive_sampler.__iter__()]
        negatives = [self.negatives[i] for i in self.negative_sampler.__iter__()]
        assert len(positives) == self.positives_per_epoch
        assert len(negatives) == self.negatives_per_epoch

        # Merge positives and negatives, and shuffle
        all_idxs = positives + negatives
        random.shuffle(all_idxs)
        print("all_idxs", len(all_idxs))
        return all_idxs.__iter__()

    def __len__(self) -> int:
        return self.epoch_size