from __future__ import annotations
import json
from typing import Callable, Dict, List, Tuple
from matplotlib import pyplot as plt
import torch
import os
import sys
from torch.utils.data import DataLoader
from dataset_classes.hrsid_dataset import HRSIDDataset
from scripts.script import Script
from yolo_lib.data.dataclasses import YOLOTile, YOLOTileStack
from yolo_lib.data_augmentation.data_augmentation import DataAugmentation
from yolo_lib.data_augmentation.flat_batch import FlatBatch
from yolo_lib.data_augmentation.mosaic import Mosaic
from yolo_lib.data_augmentation.random_crop import RandomCrop
from yolo_lib.data_augmentation.sat import SAT
from yolo_lib.models.blocks.attention import AttentionCfg, MultilayerAttentionCfg, NoAttentionCfg, YOLOv4AttentionCfg
from yolo_lib.models.blocks.dilated_encoder import EncoderConfig
from yolo_lib.detectors.managed_architectures.attention_yolof import AttentionYOLOF, AttentionYOLOFCfg
from yolo_lib.detectors.managed_architectures.base_detector import DetectorCfg
from yolo_lib.detectors.yolo_heads.label_assignment.label_assignment_cfg import DIoUBasedOverlappingAssignmentLossCfg, DistanceBasedOverlappingAssignmentLossCfg, IoUBasedOverlappingAssignmentLossCfg
from yolo_lib.models.backbones import BackboneCfg
from yolo_lib.detectors.yolo_heads.losses.objectness_loss import FocalLossCfg
from yolo_lib.detectors.yolo_heads.yolo_head import YOLOHeadCfg
from yolo_lib.detectors.yolo_heads.losses.center_yx_losses import CenterYXSmoothL1
from yolo_lib.detectors.yolo_heads.losses.siou_box_loss import SIoUBoxLoss
from yolo_lib.detectors.yolo_heads.losses.ciou_loss import DIoUBoxLoss
from yolo_lib.detectors.yolo_heads.losses.adv_loss import ADVLoss
from yolo_lib.util.model_storage import ModelStorage
from dataset_classes.ls_ssdd_dataset import LSSDDataset
from yolo_lib.performance_metrics import get_default_performance_metrics
from yolo_lib.training_loop import TrainingLoop
from yolo_lib.optimization_criteria import OptimizationCriteria
from yolo_lib.util.display_detections import display_yolo_tile


OUTPUT_BASE_DIR = "./out/box_vs_point"
LOG_FILE_DIR = os.path.join(OUTPUT_BASE_DIR, "log_files")
MODEL_STORAGE_DIR = os.path.join(OUTPUT_BASE_DIR, "model_storage")

# Training configurations
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 1
NUM_WORKERS = 0
MAX_EPOCHS = 100
PRERFETCH_FACTOR = 2

# Architecture hyperparameter constants (shared across all models)
YX_MATCH_THRESH = 2.0
YX_MULTIPLIER = 2.3
NUM_ANCHORS = 4
LOSS_OBJECTNESS_WEIGHT = 0.5
LOSS_BOX_WEIGHT = 0.5
LOSS_ADV_WEIGHT = 0.0

RESNET_VARIATION = 34
IMAGE_CHANNELS = 1

LONG_ATTENTION_NUM_LAYERS = 4
ATTENTION_NUM_LAYERS = 2
ATTENTION_NUM_CHANNELS = 64

def get_script():
    return Script(
        {
            "train": train,
            "plot_training_log": plot_training_log,
            "display_val_set": display_val_set,
        }
    )

def identity_collate(tiles: List[YOLOTile]) -> List[YOLOTile]:
    return tiles

def get_model_cfg(model_type_name: str) -> DetectorCfg:

    # Choose assignment loss function
    if model_type_name == "Box(IoU)":
        assignment_loss_cfg = IoUBasedOverlappingAssignmentLossCfg(YX_MATCH_THRESH, False, 0.5, 0.5)
    elif model_type_name == "Box(DIoU)":
        assignment_loss_cfg = DIoUBasedOverlappingAssignmentLossCfg(YX_MATCH_THRESH, False, 0.5, 0.5)
    elif model_type_name == "Point":
        assignment_loss_cfg = DistanceBasedOverlappingAssignmentLossCfg(YX_MATCH_THRESH, False, 0.5, 0.5)

    # Choose box / point regression loss
    if model_type_name in ["Box(IoU)", "Box(DIoU)"]:
        box_loss = DIoUBoxLoss(do_detach=False)
    elif model_type_name == "Point":
        box_loss = SIoUBoxLoss(1.0, 0.0, CenterYXSmoothL1())

    # Create YOLOHeadCfg
    head_cfg = YOLOHeadCfg(
        assignment_loss_cfg,
        YX_MULTIPLIER,
        NUM_ANCHORS,
        ADVLoss(),
        box_loss,
        FocalLossCfg(neg_weight=0.3, pos_weight=1.0, gamma=2).build(),
        LOSS_OBJECTNESS_WEIGHT,
        LOSS_BOX_WEIGHT,
        LOSS_ADV_WEIGHT,
    )

    return AttentionYOLOFCfg(
        head_cfg,
        BackboneCfg(IMAGE_CHANNELS, RESNET_VARIATION),
        EncoderConfig.default(),
        MultilayerAttentionCfg(ATTENTION_NUM_CHANNELS, ATTENTION_NUM_LAYERS),
    )

def get_model_configs() -> List[Dict[str, str]]:
    return [
        {"dataset_name": "HRSID", "model_type_name": "Point"},
        {"dataset_name": "HRSID", "model_type_name": "Box(DIoU)"},

        {"dataset_name": "LS-SSDD", "model_type_name": "Box(DIoU)"},
        {"dataset_name": "LS-SSDD", "model_type_name": "Point"},

        {"dataset_name": "LS-SSDD", "model_type_name": "Box(IoU)"},
        {"dataset_name": "HRSID", "model_type_name": "Box(IoU)"},

        # # Comparison of MultiLayerAttention, NoAttention, and 3DMF
        # {"dataset_name": "LS-SSDD", "assignment_loss_name": "L2", "box_loss_name": "L1+SIoU", "attention_name": "LongMultiLayerAttention"},
        # {"dataset_name": "LS-SSDD", "assignment_loss_name": "L2", "box_loss_name": "L1+SIoU", "attention_name": "MultiLayerAttention"},
        # {"dataset_name": "LS-SSDD", "assignment_loss_name": "L2", "box_loss_name": "L1+SIoU", "attention_name": "SAM"},
        # {"dataset_name": "LS-SSDD", "assignment_loss_name": "L2", "box_loss_name": "L1+SIoU", "attention_name": "NoAttention"},
    ]

def get_model_name(dataset_name: str, model_type_name: str) -> str:
    return f"{dataset_name}__{model_type_name}"

def get_data_augmentations() -> List[DataAugmentation]:
    return [
        Mosaic(0.7, 80, 10, 0.1),
        RandomCrop(list(range(576, 800, 32)), 0.4, 1),
        SAT(0.2, 10, 0.01, 0.1)
    ]

def get_training_loop(dataset_name: str, model_type_name: str) -> TrainingLoop:
    # Build model
    model_cfg = get_model_cfg(model_type_name)
    model = model_cfg.build().cuda()

    # Get model name
    name = get_model_name(dataset_name, model_type_name)
    print(name)

    # Get dataset
    train_dl, test_dl = get_dataloaders(dataset_name)

    N = 10
    P = 0.95
    lr_scheduler_lambda = lambda i: ((i+1) / N) if i < N else P ** (i + 1 - N)
    return TrainingLoop(
        name,
        model,
        torch.optim.Adam(model.parameters(), 5e-5),
        get_data_augmentations(),
        get_default_performance_metrics(),
        lr_scheduler_lambda,
        MAX_EPOCHS,
        train_dl,
        test_dl,
        get_optimization_criterias(),
    )

def get_optimization_criterias() -> List[OptimizationCriteria]:
    return [
        OptimizationCriteria(
            "dAP",
            ["performance_metrics", "Distance-AP", "AP"],
        ),
        OptimizationCriteria(
            "F2",
            ["performance_metrics", "Distance-AP", "F2", "F2"],
        ),
        OptimizationCriteria(
            "loss",
            ["epoch_loss_sum"],
            minimize=True,
        ),
    ]


def get_dataloaders(dataset_name: str) -> Tuple[DataLoader, DataLoader]:
    if dataset_name == "LS-SSDD":
        train_ds, test_ds = LSSDDataset.get_split()
    elif dataset_name == "HRSID":
        train_ds, test_ds = HRSIDDataset.get_split()

    train_dl = DataLoader(
        train_ds,
        TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=YOLOTileStack.stack_tiles,
        prefetch_factor=PRERFETCH_FACTOR,
        persistent_workers=False,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=identity_collate,
        prefetch_factor=PRERFETCH_FACTOR,
        persistent_workers=False,
    )

    return train_dl, test_dl

def train():
    model_storage = ModelStorage(MODEL_STORAGE_DIR)
    config_dicts = get_model_configs()
    for config_dict in config_dicts:
        dataset_name = config_dict["dataset_name"]
        model_type_name = config_dict["model_type_name"]

        # Create model and training loop
        training_loop = get_training_loop(dataset_name, model_type_name)

        # Train model
        training_loop.run(model_storage, LOG_FILE_DIR)

def plot_training_log():
    model_cfgs = [
        {"dataset_name": "LS-SSDD", "model_type_name": "Box(IoU)"},
        {"dataset_name": "LS-SSDD", "model_type_name": "Box(DIoU)"},
        {"dataset_name": "LS-SSDD", "model_type_name": "Point"},
    ]

    for config_dict in model_cfgs:
        dataset_name = config_dict["dataset_name"]
        model_type_name = config_dict["model_type_name"]

        model_name = get_model_name(dataset_name, model_type_name)

        path = os.path.join(LOG_FILE_DIR, model_name, "training_log.json")
        with open(path, "r") as F:
            log = json.load(F)
        
        d_ap_vals = [x["performance_metrics"]["Distance-AP"]["AP"] for x in log]
        epochs = [x["epoch"] for x in log]
        plt.plot(epochs, d_ap_vals, label=model_type_name)
        print(f"{model_type_name} : {max(d_ap_vals)}")
    
    plt.legend()
    plt.show()

def display_val_set():
    model_cfgs = [
        {"dataset_name": "HRSID", "model_type_name": "Point"},
    ]

    model_storage = ModelStorage(MODEL_STORAGE_DIR)

    for config_dict in model_cfgs:
        dataset_name = config_dict["dataset_name"]
        model_type_name = config_dict["model_type_name"]

        model_name = get_model_name(dataset_name, model_type_name)
        model_cfg = get_model_cfg(model_type_name)
        model = model_cfg.build().cuda()
        model = TrainingLoop.load_trained_model(model_name, "dAP", model, model_storage)

        # Get dataset
        train_dl, test_dl = get_dataloaders(dataset_name)
        for i, tiles in enumerate(test_dl):
            tiles: List[YOLOTile] = tiles
            tile = tiles[0]
            detections = model.detect_objects(tile.image.cuda()).as_detection_block().filter_min_positivity(0.6)
            display_yolo_tile(
                tile,
                f"./out/box_vs_point/val/file{i}.png",
                detections,
                # grid_spacing=None,
                display_mode="WIDE_SQUARE",
            )

            if i == 10:
                break

