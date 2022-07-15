from __future__ import annotations
from typing import List, Tuple
import torch
import os
from torch.utils.data import Dataset
from dataset_classes.read_jpeg import read_jpeg
from yolo_lib.data.annotation import AnnotationBlock
from yolo_lib.data.string_dataset import StringDataset
from yolo_lib.data.dataclasses import YOLOTile
import xml.etree.ElementTree as ET

DATA_BASE_PATH = "./datasets/LS-SSDD-v1.0-OPEN"
XML_BASE_PATH = os.path.join(DATA_BASE_PATH, "Annotations_sub")
JPEG_BASE_PATH = os.path.join(DATA_BASE_PATH, "JPEGImages_sub")
IMAGE_SETS_BASE_PATH = os.path.join(DATA_BASE_PATH, "ImageSets/Main")
TILE_SIZE = 800


def read_xml(path: str) -> AnnotationBlock:
    tree = ET.parse(path)
    root = tree.getroot()
    objects = [child for child in root if child.tag == "object"]
    num_objects = len(objects)
    if num_objects == 0:
        return AnnotationBlock.empty()

    # Extract xmin, xmax, ymin, and ymax from all objects
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []
    for obj in objects:
        # Dict representation of object
        as_dict = {attr.tag: attr for attr in obj}

        # Name should be "ship"
        name = as_dict["name"].text
        assert name == "ship", f"got name = {name} (expected ship)"

        # bndbox should have xmin, ymin, xmax, ymax
        bndbox_dict = {attr.tag: attr for attr in as_dict["bndbox"]}
        xmin_list.append(float(bndbox_dict["xmin"].text))
        ymin_list.append(float(bndbox_dict["ymin"].text))
        xmax_list.append(float(bndbox_dict["xmax"].text))
        ymax_list.append(float(bndbox_dict["ymax"].text))

    # Create torch tensor from list of attributes, and compute centere and size
    xmin = torch.tensor(xmin_list)
    ymin = torch.tensor(ymin_list)
    xmax = torch.tensor(xmax_list)
    ymax = torch.tensor(ymax_list)

    center_y = (ymax + ymin) / 2
    center_x = (xmax + xmin) / 2
    size_h = ymax - ymin
    size_w = xmax - xmin
    center_yx = torch.stack([center_y, center_x]).T
    size_hw = torch.stack([size_h, size_w]).T
    assert center_yx.shape == (num_objects, 2)
    assert size_hw.shape == (num_objects, 2)

    # Create AnnotationBlock
    return AnnotationBlock(
        num_objects,
        center_yx,
        torch.tensor(True)[None].expand(num_objects),
        size_hw,
        torch.tensor(True)[None].expand(num_objects),
        torch.tensor(0.0)[None].expand(num_objects),
        torch.tensor(False)[None].expand(num_objects),
        torch.tensor(False)[None].expand(num_objects),
    )


class LSSDDataset(Dataset):
    def __init__(self, grid_names: List[str]) -> None:
        super().__init__()
        # Store grid names as string-dataset
        self.name_string_ds = StringDataset(grid_names)

        # Read each xml file and create a compact json representation
        self.annotation_string_ds = StringDataset([
            read_xml(os.path.join(XML_BASE_PATH, f"{grid_name}.xml")).to_str()
            for grid_name in grid_names
        ])

    @staticmethod
    def get_subset_filenames(filename: str) -> LSSDDataset:
        path = os.path.join(IMAGE_SETS_BASE_PATH, filename)
        print("path", path)
        with open(path, "r") as F:
            lines = F.read().split("\n")
            return [line for line in lines if len(line)]

    @staticmethod
    def get_split(num_displayed_tests: int) -> Tuple[LSSDDataset, LSSDDataset, LSSDDataset]:
        train_ds = LSSDDataset.get_image_set("train.txt")
        test_filenames = LSSDDataset.get_subset_filenames("test.txt")
        test_ds = LSSDDataset(test_filenames[num_displayed_tests:])
        displayed_test_ds = LSSDDataset(test_filenames[:num_displayed_tests])
        return (train_ds, test_ds, displayed_test_ds)

    @staticmethod
    def get_image_set(filename: str) -> LSSDDataset:
        lines = LSSDDataset.get_subset_filenames(filename)
        return LSSDDataset(lines)

    def __getitem__(self, index) -> YOLOTile:
        # Get image
        grid_name = self.name_string_ds.__getitem__(index)
        jpeg_name = os.path.join(JPEG_BASE_PATH, f"{grid_name}.jpg")
        img = read_jpeg(jpeg_name, TILE_SIZE)

        # Get json representation of annotation, and create AnnotationBlock
        annotation_str = self.annotation_string_ds[index]
        annotations = AnnotationBlock.from_str(annotation_str)
        return YOLOTile(img[None, None], annotations)

    def __len__(self) -> int:
        return len(self.name_string_ds)

