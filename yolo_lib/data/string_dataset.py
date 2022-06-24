from __future__ import annotations
from ctypes import Union
from typing import List, Tuple
import torch
import numpy as np
import json


__all__ = ["StringDataset"]


# --- UTILITY FUNCTIONS ---
def string_to_sequence(s: str, dtype) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=dtype)

def sequence_to_string(seq: np.ndarray) -> str:
    return ''.join([chr(c) for c in seq])

def pack_sequences(seqs: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
    values = np.concatenate(seqs, axis=0)
    offsets = np.cumsum([len(s) for s in seqs])
    return values, offsets

def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    off1 = offsets[index]
    if index > 0:
        off0 = offsets[index - 1]
    elif index == 0:
        off0 = 0
    else:
        raise ValueError(index)
    return values[off0:off1]


class StringDataset(torch.utils.data.Dataset):
    """
    A torch map-style dataset that corresponds to a list of strings.
    This implementation of a string dataset allows efficiently sharing
    memory between sub-processes without copying memory.
    This is copied from the following link, with trivial modifications:
    https://github.com/pytorch/pytorch/issues/13246#issuecomment-617140519
    For details on why this is so important, see:
    https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662

    Modifications:
    - Added store() and load() methods
    """
    def __init__(self, strings: List[str], dtype=np.int8):
        if len(strings) == 0:
            self.len = 0
            return

        self.len = len(strings)

        # Convert each string to sequence of codepoints (integer),
        # and then pack them into a numpy array.
        seqs = [string_to_sequence(s, dtype) for s in strings]
        self.strings_v, self.strings_o = pack_sequences(seqs)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, i) -> str:
        assert 0 <= i < self.len
        # Use indirect lookup to fetch the i-th sequence. This only uses integer numpy
        # array lookups, which avoids that the objects are subsequently replicated by
        # child processes.
        seq = unpack_sequence(self.strings_v, self.strings_o, i)
        string = sequence_to_string(seq)
        return string

    def store(self, path: str):
        with open(path, "w") as F:
            json.dump(list(self), F)

    @staticmethod
    def load(path: str) -> StringDataset:
        print("Loading StringDataset from", path)
        try:
            with open(path, "r") as F:
                as_list = json.load(F)
                return StringDataset(as_list)

        except FileNotFoundError:
            return None
