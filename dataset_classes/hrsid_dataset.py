from __future__ import annotations
from typing import List, Tuple
import os
from torch.utils.data import Dataset
from dataset_classes.ls_ssdd_dataset import read_jpeg
from yolo_lib.data.annotation import AnnotationBlock
from yolo_lib.data.string_dataset import StringDataset
from yolo_lib.data.dataclasses import YOLOTile
import json


DATA_BASE_PATH = "./datasets/HRSID_JPG"
ANNOTATION_BASE_PATH = os.path.join(DATA_BASE_PATH, "annotations")
JPEG_BASE_PATH = os.path.join(DATA_BASE_PATH, "JPEGImages")
TILE_SIZE = 800


class HRSIDDataset(Dataset):
    def __init__(self, tile_strings: List[str]) -> None:
        super().__init__()
        self.string_dataset = StringDataset(tile_strings)

    @staticmethod
    def get_tile_strings(filename: str) -> List[str]:
        path = os.path.join(ANNOTATION_BASE_PATH, filename)
        with open(path, "r") as F:
            as_json = json.load(F)

        images_dict = {
            img["id"]: {"fn": img["file_name"], "an": []}
            for img in as_json["images"]
        }

        for annotation in as_json["annotations"]:
            (min_x, min_y, w, h) = annotation["bbox"]
            center_x = min_x + 0.5 * w
            center_y = min_y + 0.5 * h
            annotation_dict = {"yx": [center_y, center_x], "hw": [h, w]}
            images_dict[annotation["image_id"]]["an"].append(annotation_dict)

        return [json.dumps(val) for val in images_dict.values()]

    @staticmethod
    def get_split(num_displayed_tests: int) -> Tuple[HRSIDDataset, HRSIDDataset, HRSIDDataset]:
        train_ds = HRSIDDataset(HRSIDDataset.get_tile_strings("train2017.json"))
        test_tile_strings = HRSIDDataset.get_tile_strings("test2017.json")
        test_ds = HRSIDDataset(test_tile_strings[num_displayed_tests:])
        displayed_test_ds = HRSIDDataset(test_tile_strings[:num_displayed_tests])
        return (train_ds, test_ds, displayed_test_ds)

    def __getitem__(self, idx: int) -> YOLOTile:
        as_dict = json.loads(self.string_dataset[idx])
        img_fname = as_dict["fn"]
        img_path = os.path.join(JPEG_BASE_PATH, img_fname)
        img = read_jpeg(img_path, TILE_SIZE)
        annotations = AnnotationBlock.from_dict_list(as_dict["an"])
        return YOLOTile(img[None, None], annotations)

    def __len__(self) -> int:
        return len(self.string_dataset)


