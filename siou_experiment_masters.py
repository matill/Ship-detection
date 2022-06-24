from __future__ import annotations
import json
from random import shuffle
from typing import Dict, Iterator, List
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.data import DataLoader, Dataset
from yolo_lib.data.annotation import AnnotationBlock
from yolo_lib.data.string_dataset import StringDataset
from yolo_lib.data.dataset_decorators import CatDataset
from yolo_lib.data.samplers import RandomPartitionSampler
from yolo_lib.data.dataclasses import YOLOTile, YOLOTileStack
from yolo_lib.cfg import DEVICE, USE_GPU
from real_experiment_cfg import get_anchor_priors
from yolo_lib.data_augmentation.flat_batch import FlatBatch
from yolo_lib.data_augmentation.mosaic import Mosaic
from yolo_lib.data_augmentation.random_crop import RandomCrop
from yolo_lib.data_augmentation.sat import SAT
from yolo_lib.models.blocks.attention import MultilayerAttentionCfg
from yolo_lib.detectors.cfg_types.dilated_encoder_cfg import EncoderConfig
from yolo_lib.models.backbones import BackboneCfg
from yolo_lib.detectors.cfg_types.detector_cfg import AttentionYOLOFCfg, AuxiliaryHeadYOLOFCfg
from yolo_lib.detectors.cfg_types.head_cfg import LocalMatchingYOLOHeadCfg, OverlappingCellYOLOHeadCfg
from yolo_lib.detectors.cfg_types.loss_cfg import FocalLossCfg
from yolo_lib.detectors.yolo_heads.label_assigner.topk import TopK
from yolo_lib.detectors.yolo_heads.yolo_head import IoUPotoMatchlossCfg, OverlappingCellYOLOHead, PointPotoMatchlossCfg, PotoMatchlossCfg
from yolo_lib.detectors.yolo_heads.losses.center_yx_losses import CenterYXSmoothL1
from yolo_lib.detectors.yolo_heads.losses.siou_box_loss import SIoUBoxLoss
from yolo_lib.detectors.yolo_heads.losses.ciou_loss import CIoUBoxLoss, DIoUBoxLoss
from yolo_lib.detectors.yolo_heads.losses.sincos_losses import SinCosLoss
from yolo_lib.model_storage import ModelStorage

from yolo_lib.display_detections import display_yolo_tile
from tqdm import tqdm

from ls_ssdd_dataset import LSSDDataset
from hrsid_dataset import HRSIDDataset
from yolo_lib.performance_metrics import BasePerformanceMetric, get_default_performance_metrics
from yolo_lib.training_loop import TrainingLoop
from yolo_lib.detectors.managed_architectures.auxiliary_head_yolof import AuxiliaryHeadYOLOF

DATA_BASE_PATH = "./LS-SSDD-v1.0-OPEN"
XML_BASE_PATH = os.path.join(DATA_BASE_PATH, "Annotations_sub")
JPEG_BASE_PATH = os.path.join(DATA_BASE_PATH, "JPEGImages_sub")

OUTPUT_DIR = "./out/train_with_ls_ssd/log_files"
MODEL_STORAGE_DIR = "./out/train_with_ls_ssd/model_storage"

# Train set size: 6000
TILE_SIZE = 800 
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 1
NUM_WORKERS = 0
MAX_EPOCHS = 100
PRERFETCH_FACTOR = 2

NUM_DISPLAYED_TESTS = 50

EPOCHS_PER_DISPLAY = 10
EPOCHS_PER_CHECKPOINT = 5


def identity_collate(tiles: List[YOLOTile]) -> List[YOLOTile]:
    return tiles

def get_experiment_name(e: Dict[str, str]) -> str:
    dataset = e["dataset"]
    matchloss_name = e["matchloss_name"]
    loss_name = e["loss_name"]
    return f"{dataset}+POTO({matchloss_name})+{loss_name}"

def get_training_loop(dataset: str, matchloss_name: str, loss_name: str, loss_weight_name: str) -> TrainingLoop:
    name = f"{dataset}+POTO({matchloss_name})+{loss_name}"
    print(name)

    # Matchloss type
    matchloss_cfg = {
        "IoU": IoUPotoMatchlossCfg(0.8),
        "CenterDistance": PointPotoMatchlossCfg(0.5, 0.5, None),
        "CenteredIoU+CenterDistance": PointPotoMatchlossCfg(0.334, 0.333, 0.333),
    }[matchloss_name]

    # Regression loss type
    box_loss = {
        "CIoU": CIoUBoxLoss(stable=False),
        "UnstableCIoU": CIoUBoxLoss(stable=False),
        "StableCIoU": CIoUBoxLoss(stable=True),
        "DIoU": DIoUBoxLoss(do_detach=True),
        "DIoU(NoDetach)": DIoUBoxLoss(do_detach=False),
        "CenteredIoU+CenterDistance": SIoUBoxLoss(0.5, 0.5, CenterYXSmoothL1()),
    }[loss_name]

    model = AuxiliaryHeadYOLOF(
        BackboneCfg(1, 34).build(),
        EncoderConfig.default(),
        MultilayerAttentionCfg(64,2),
        OverlappingCellYOLOHead(
            512,
            4,
            get_anchor_priors(4),
            2.3,
            2.0,
            matchloss_cfg,
            SinCosLoss(True, False),
            box_loss,
            FocalLossCfg(neg_weight=0.3, pos_weight=1.0, gamma=2).build(),
            0.5,
            0.30,
            0.2,
        ),
        LocalMatchingYOLOHeadCfg(
            512,
            4,
            get_anchor_priors(4),
            0.25,
            0.25,
            0.25,
            0.25,
            0.05,
            0.10,
            0.425,
            0.425
        ).build(),
        0.1
    ).cuda()

    N = 10
    P = 0.95
    lr_scheduler_lambda = lambda i: ((i+1) / N) if i < N else P ** (i + 1 - N)
    return TrainingLoop(
        name,
        model,
        torch.optim.Adam(model.parameters(), 5e-5),
        [
            Mosaic(0.7, 80, 10, 0.1),
            RandomCrop(list(range(576, 800, 32)), 0.4, 1),
            FlatBatch(1.0),
            SAT(0.2, 10, 0.01, 0.1)
        ],
        get_default_performance_metrics(),
        [],
        lr_scheduler_lambda,
        MAX_EPOCHS
    )

def get_experiment_specs() -> List[Dict[str, str]]:
    return [

        # LS-SSDD


        {"dataset": "LS-SSDD", "matchloss_name": "CenterDistance", "loss_name": "CenteredIoU+CenterDistance"},
        {"dataset": "LS-SSDD", "matchloss_name": "CenterDistance", "loss_name": "DIoU(NoDetach)"},

        {"dataset": "LS-SSDD", "matchloss_name": "CenteredIoU+CenterDistance", "loss_name": "CenteredIoU+CenterDistance"},
        {"dataset": "LS-SSDD", "matchloss_name": "CenteredIoU+CenterDistance", "loss_name": "DIoU(NoDetach)"},

        {"dataset": "LS-SSDD", "matchloss_name": "IoU", "loss_name": "CenteredIoU+CenterDistance"},
        {"dataset": "LS-SSDD", "matchloss_name": "IoU", "loss_name": "DIoU(NoDetach)"},


        # {"dataset": "LS-SSDD", "matchloss_name": "CenterDistance", "loss_name": "DIoU"},
        # {"dataset": "LS-SSDD", "matchloss_name": "IoU", "loss_name": "DIoU"},

        # {"dataset": "LS-SSDD", "matchloss_name": "CenteredIoU+CenterDistance", "loss_name": "CIoU"},
        # {"dataset": "LS-SSDD", "matchloss_name": "CenteredIoU+CenterDistance", "loss_name": "StableCIoU"},
        # {"dataset": "LS-SSDD", "matchloss_name": "CenteredIoU+CenterDistance", "loss_name": "UnstableCIoU"},
        # {"dataset": "LS-SSDD", "matchloss_name": "CenteredIoU+CenterDistance", "loss_name": "DIoU"},
        # {"dataset": "LS-SSDD", "matchloss_name": "CenterDistance", "loss_name": "CIoU"},

        # {"dataset": "LS-SSDD", "matchloss_name": "IoU", "loss_name": "CIoU"},
        # {"dataset": "LS-SSDD", "matchloss_name": "IoU", "loss_name": "StableCIoU"},


        # LS-SSDD+HRSID
        # {"dataset": "LS-SSDD+HRSID", "matchloss_name": "IoU", "loss_name": "CIoU"},
        # {"dataset": "LS-SSDD+HRSID", "matchloss_name": "CenterDistance", "loss_name": "CIoU"},
        # {"dataset": "LS-SSDD+HRSID", "matchloss_name": "CenterDistance", "loss_name": "CenteredIoU+CenterDistance"},
    ]

def train():
    model_storage = ModelStorage(MODEL_STORAGE_DIR)
    experiment_specs = get_experiment_specs()

    # Create datasets
    ls_ssdd_split = LSSDDataset.get_split(NUM_DISPLAYED_TESTS)
    hrsid_split = HRSIDDataset.get_split(NUM_DISPLAYED_TESTS)
    (ls_ssdd_train_ds, ls_ssdd_test_ds, ls_ssdd_displayed_test_ds) = ls_ssdd_split
    (hrsid_train_ds, hrsid_test_ds, hrsid_displayed_test_ds) = hrsid_split


    for experiment_spec in experiment_specs:
        ds_cfg = experiment_spec["dataset"]
        matchloss_name = experiment_spec["matchloss_name"]
        loss_name = experiment_spec["loss_name"]

        # Decide which datasets to use
        if ds_cfg == "LS-SSDD":
            print(ds_cfg, "Using LS-SSDD only")
            train_ds = ls_ssdd_train_ds
            test_ds = ls_ssdd_test_ds
            displayed_test_ds = ls_ssdd_displayed_test_ds
        elif ds_cfg == "LS-SSDD+HRSID":
            print(ds_cfg, "Using LS-SSDD and HRSID")
            train_ds = CatDataset([hrsid_train_ds, ls_ssdd_train_ds])
            test_ds = CatDataset([hrsid_test_ds, ls_ssdd_test_ds])
            displayed_test_ds = CatDataset([hrsid_displayed_test_ds, ls_ssdd_displayed_test_ds])
        else:
            assert False

        # Create data loaders
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

        displayed_test_dl = DataLoader(
            displayed_test_ds,
            batch_size=TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=identity_collate,
        )

        # for experiment in experiment_specs:
        training_loop = get_training_loop(ds_cfg, matchloss_name, loss_name)
        # model_storage.load_model(training_loop, training_loop.name)
        # training_loop.evaluate(test_dl, displayed_test_dl, f"tmp/{training_loop.name}", False)
        # performance_metrics = training_loop.epoch_log_object["performance_metrics"]
        # print(json.dumps(performance_metrics, indent=2))
        training_loop.run(
            EPOCHS_PER_DISPLAY,
            EPOCHS_PER_CHECKPOINT,
            model_storage,
            train_dl,
            test_dl,
            displayed_test_dl,
            OUTPUT_DIR
        )

def examples():
    model_storage = ModelStorage(MODEL_STORAGE_DIR)
    experiment_specs = get_experiment_specs()

    # Create datasets
    ls_ssdd_split = LSSDDataset.get_split(NUM_DISPLAYED_TESTS)
    (train_ds, test_ds, displayed_test_ds) = ls_ssdd_split
    for experiment_spec in experiment_specs:
        ds_cfg = experiment_spec["dataset"]
        matchloss_name = experiment_spec["matchloss_name"]
        loss_name = experiment_spec["loss_name"]
        training_loop = get_training_loop(ds_cfg, matchloss_name, loss_name)
        model_storage.load_model(training_loop, training_loop.name, not_exists_ok=False)
        model = training_loop.model.cuda()
        log = training_loop.read_log(OUTPUT_DIR)
        threshold = log[-1]["performance_metrics"]["Distance-AP"]["F2"]["positivity"]
        print("threshold", threshold)
        count = 0

        tile = test_ds[28]
        # for i, tile in enumerate(test_ds):
            # if tile.annotations.size > 0:
                # count += tile.annotations.size
        detections = model.detect_objects(tile.image.cuda()).as_detection_block().filter_min_positivity(threshold) 
        fname = f"out/train_with_ls_ssd/examples/{training_loop.name}-final.png"
        display_yolo_tile(tile, fname, detections, display_mode="BOX")
        if count > 50:
            break

def plot():
    raise NotImplementedError

def table():
    table = []
    for e in get_experiment_specs():
        name = get_experiment_name(e)
        path = os.path.join(OUTPUT_DIR, name, "training_log.json")
        with open(path, "r") as F:
            log = json.load(F)

        # Search for the best performance for each model
        best_ap_score = -1
        best_ap_log_obj = None
        best_iou_ap_score = -1
        best_iou_ap_log_obj = None
        best_iou50_ap_score = -1
        best_iou50_ap_log_obj = None
        best_f2_score = -1
        best_f2_log_obj = None
        best_ciou_score = -1
        best_ciou_log_obj = None
        for log_obj in log:
            ap_score = log_obj["performance_metrics"]["Distance-AP"]["AP"]
            iou_ap_score = log_obj["performance_metrics"]["IoU-AP"]["AP"]
            iou50_ap_score = log_obj["performance_metrics"]["IoU-AP"]["AP_0.5"]
            f2_score = log_obj["performance_metrics"]["Distance-AP"]["F2"]["F2"]
            ciou_score = log_obj["performance_metrics"]["Distance-AP"]["avg_iou"]
            if best_ap_score < ap_score:
                best_ap_score = ap_score
                best_ap_log_obj = log_obj
            if best_iou_ap_score < iou_ap_score:
                best_iou_ap_score = iou_ap_score
                best_iou_ap_log_obj = log_obj
            if best_iou50_ap_score < iou50_ap_score:
                best_iou50_ap_score = iou50_ap_score
                best_iou50_ap_log_obj = log_obj
            if best_f2_score < f2_score:
                best_f2_score = f2_score
                best_f2_log_obj = log_obj
            if best_ciou_score < ciou_score:
                best_ciou_score = ciou_score
                best_ciou_log_obj = log_obj

        # Add to table list
        table.append({
            "name": name,
            "matchloss_name": e["matchloss_name"],
            "loss_name": e["loss_name"],
            "distance_ap": best_ap_log_obj["performance_metrics"]["Distance-AP"]["AP"],
            "iou_ap": best_iou_ap_log_obj["performance_metrics"]["IoU-AP"]["AP"],
            "iou50_ap": best_iou50_ap_log_obj["performance_metrics"]["IoU-AP"]["AP_0.5"],
            "f2": best_f2_log_obj["performance_metrics"]["Distance-AP"]["F2"]["F2"],
            "precision": best_f2_log_obj["performance_metrics"]["Distance-AP"]["F2"]["precision"],
            "recall": best_f2_log_obj["performance_metrics"]["Distance-AP"]["F2"]["recall"],
            "SIoU": best_ciou_score,
        })

    # Print table
    lines = [
        r"\begin{tabular}{| c c | c c c c c c c |}",
        r"\hline",
        r"\multicolumn{2}{|c|}{\textbf{Configuration}} & \multicolumn{7}{c|}{\textbf{Performance}}",
        r"\\",
        r"Matching & Loss & d-AP & AP & AP50 & F2 & p & r & SIoU \\",
        r"\hline",
    ]

    for e in table:
        loss_name = {
            "CenteredIoU+CenterDistance": "$SIoU + L_1$",
            "DIoU(NoDetach)": "$DIoU$",
        }[e["loss_name"]]

        matchloss_name = {
            "CenterDistance": "$L_2$",
            "IoU": "$IoU$",
            "CenteredIoU+CenterDistance": "$L_2 + SIoU$",
        }[e["matchloss_name"]]

        distance_ap = f'{e["distance_ap"]:.3f}'
        iou_ap = f'{e["iou_ap"]:.3f}'
        iou50_ap = f'{e["iou50_ap"]:.3f}'
        f2 = f'{e["f2"]:.3f}'
        precision = f'{e["precision"]:.3f}'
        recall = f'{e["recall"]:.3f}'
        siou = f'{e["SIoU"]:.3f}'
        lines.append(f"{matchloss_name} & {loss_name} & {distance_ap} & {iou_ap} & {iou50_ap} & {f2} & {precision} & {recall} & {siou} \\\\")

        # $IoU$ & $DIoU$ & xxx & xxx & xxx & xxx & xxx & xxx
        # \\
        # $IoU$ & $SIoU + L_2$ & xxx & xxx & xxx & xxx & xxx & xxx
        # \\
        # $SIoU + L_2$ & $DIoU$ & xxx & xxx & xxx & xxx & xxx & xxx
        # \\
        # $SIoU + L_2$ & $SIoU + L_2$ & xxx & xxx & xxx & xxx & xxx & xxx
        # \\
        # $L_2$ & $DIoU$ & xxx & xxx & xxx & xxx & xxx & xxx
        # \\
        # $L_2$ & $SIoU + L_2$ & xxx & xxx & xxx & xxx & xxx & xxx
        # \\

    lines += [
        r"\hline",
        r"\end{tabular}"
    ]

    # table = [
    #     {
    #         "distance_ap": e["performance_metrics"]["best_ap_log_obj"]["Distance-AP"]["AP"],
    #         "ap": e["performance_metrics"]["best_iou_ap_log_obj"]["IoU-AP"]["AP"],
    #         "ap50": e["performance_metrics"]["best_iou50_ap_log_obj"]["IoU-AP"]["AP_0.5"],
    #         "f2": e["performance_metrics"]["best_f2_log_obj"]["Distance-AP"]["F2"],
    #     }
    #     for e in table
    # ]

    print("\n".join(lines))

    print("\n\n")
    print(json.dumps(table, indent=2))


    # # print(json.dumps(table, indent=2))
    # print(json.dumps([
    #     {
    #         "name": e["name"],
    #         # "epochs": list({
    #         #     e["best_ap_log_obj"]["epoch"],
    #         #     e["best_iou_ap_log_obj"]["epoch"],
    #         #     e["best_iou50_ap_log_obj"]["epoch"],
    #         #     e["best_f2_log_obj"]["epoch"],
    #         # }),
    #         "best_ap": e["best_ap_log_obj"]["performance_metrics"]["Distance-AP"]["AP"],
    #         # "best_f2": e["best_f2_log_obj"]["performance_metrics"]["Distance-AP"]["F2"]["F2"],
    #         # "best_iou_ap": e["best_iou_ap_log_obj"]["performance_metrics"]["IoU-AP"]["AP"],
    #         # "best_iou50_ap": e["best_iou50_ap_log_obj"]["performance_metrics"]["IoU-AP"]["AP_0.5"],
    #     }
    #     for e in table
    # ], indent=2))



if __name__ == "__main__":
    print("Args", sys.argv)
    assert len(sys.argv) in [1, 2]
    arg = "train" if len(sys.argv) == 1 else sys.argv[1]
    assert arg in ["train", "plot", "table", "examples"]
    if arg == "train":
        train()
    if arg == "plot":
        plot()
    if arg == "table":
        table()
    if arg == "examples":
        examples()
