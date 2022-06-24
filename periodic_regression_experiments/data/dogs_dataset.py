from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
import json
from scipy.io import loadmat
from torchvision.transforms.functional import rotate, center_crop, resize


DATA_BASE_PATH = "../datasets/stanford_dogs"
TRAIN_FILE = os.path.join(DATA_BASE_PATH, "train_list.mat")
TEST_FILE = os.path.join(DATA_BASE_PATH, "test_list.mat")
IMG_FOLDER = os.path.join(DATA_BASE_PATH, "Images")


ANNOTATION_BASE_PATH = os.path.join(DATA_BASE_PATH, "annotations")
JPEG_BASE_PATH = os.path.join(DATA_BASE_PATH, "JPEGImages")
TILE_SIZE = 512
CIRCLE_RADIUS_PX = 150

TRAIN_RNG_SEED = 932851
TEST_RNG_SEED = 39052

# List of corrupted image files
IMAGE_BLACKLIST = [
    "n02105855_2933.jpg"
]

class DogDataset(Dataset):
    def __init__(self, file_list: str, true_rotations_degrees: np.ndarray, known_rotations_degrees: np.ndarray) -> None:
        super().__init__()
        self.file_list = file_list
        self.true_rotations_degrees = true_rotations_degrees
        self.known_rotations_01 = known_rotations_degrees / 360

        coordinate_arange = torch.arange(0, TILE_SIZE, 1)
        diffs = ((TILE_SIZE / 2) - coordinate_arange).square()
        diffs_2d = diffs[:, None] + diffs[None, :]
        assert diffs_2d.shape == (TILE_SIZE, TILE_SIZE)
        self.circle_mask = (diffs_2d < (CIRCLE_RADIUS_PX ** 2)).cuda()

    @staticmethod
    def get_train_ds(max_inaccuracy_degrees: Optional[float]=None) -> DogDataset:
        return DogDataset.get_ds(TRAIN_FILE, TRAIN_RNG_SEED, max_inaccuracy_degrees)

    @staticmethod
    def get_test_ds() -> DogDataset:
        return DogDataset.get_ds(TEST_FILE, TEST_RNG_SEED)

    @staticmethod
    def get_ds(list_file: str, rng_seed: int, max_inaccuracy_degrees: Optional[float]=None) -> DogDataset:

        # Read list of train files
        file = loadmat(list_file)
        file_list_np = file["file_list"]
        file_list = [
            "".join((str(x) for x in np_str))[2:-2]
            for np_str in file_list_np
        ]

        file_list = [fname for fname in file_list if fname not in IMAGE_BLACKLIST]
        n_files = len(file_list)

        # Get rotation for each image
        rng = np.random.default_rng(rng_seed)
        true_rotations_degrees = rng.random(size=len(file_list)) * 360

        # Get array of inaccuracies
        if max_inaccuracy_degrees is None:
            known_rotations_degrees = true_rotations_degrees
        else:
            rotation_errors_degrees = (rng.random(size=n_files) - 0.5) * max_inaccuracy_degrees
            known_rotations_degrees = (true_rotations_degrees + rotation_errors_degrees) % 360

        # Return DogDataset
        return DogDataset(file_list, true_rotations_degrees, known_rotations_degrees)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, float]:
        # Read image file
        path = os.path.join(IMG_FOLDER, self.file_list[idx])
        img_np = plt.imread(path, format="jpeg")
        img_torch = torch.tensor(img_np).permute(2, 0, 1).type(torch.float32).cuda() / 255
        c, h, w = img_torch.shape
        if c > 3:
            img_torch = img_torch[0:3]
            assert img_torch.shape == (3, h, w)

        # Crop to 512x512
        if min(h, w) >= TILE_SIZE:
            img_cropped = center_crop(img_torch, [TILE_SIZE, TILE_SIZE])
        else:
            scale = TILE_SIZE / min(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            img_scaled = resize(img_torch, (new_h, new_w))
            img_cropped = center_crop(img_scaled, (TILE_SIZE, TILE_SIZE))

        # Rotate, and combine rotated and un-rotated image
        assert img_cropped.shape == (3, TILE_SIZE, TILE_SIZE), f"{img_cropped.shape} {img_torch.shape} {path}"
        true_angle_degrees = self.true_rotations_degrees[idx]
        known_angle_01 = self.known_rotations_01[idx]
        img_rotated = rotate(img_cropped, true_angle_degrees, expand=False)
        img_out = torch.where(self.circle_mask[None, :, :], img_rotated, img_cropped)
        assert img_out.shape == (3, TILE_SIZE, TILE_SIZE)
        return (img_out, known_angle_01)

    def __len__(self) -> int:
        return len(self.file_list)


