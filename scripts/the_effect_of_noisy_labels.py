import json
import os
import sys
from typing import Dict, List, Tuple
import torch
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from yolo_lib.data.yolo_tile import YOLOTileStack
from yolo_lib.data.fake_data import DsWeaknessCfg, ImgShapeCfg, SyntheticDs, VesselShapeCfg, FakeDataCfg
from yolo_lib.data_augmentation.sat import SAT
from yolo_lib.models.blocks.attention import MultilayerAttentionCfg
from yolo_lib.detectors.cfg_types.dilated_encoder_cfg import EncoderConfig
from yolo_lib.models.backbones import BackboneCfg
# from yolo_lib.detectors.cfg_types.head_cfg import LocalMatchingYOLOHeadCfg
from yolo_lib.detectors.cfg_types.loss_cfg import FocalLossCfg
from yolo_lib.detectors.managed_architectures.auxiliary_head_yolof import AuxiliaryHeadYOLOF
from yolo_lib.detectors.managed_architectures.base_detector import BaseDetector
# from yolo_lib.detectors.yolo_heads.heads.managed_yolo_head import ManagedYOLOHead
from yolo_lib.detectors.yolo_heads.yolo_head import YOLOHead
# from yolo_lib.detectors.yolo_heads.yolo_head import OverlappingCellYOLOHead, PointPotoMatchlossCfg
from yolo_lib.detectors.yolo_heads.losses.center_yx_losses import CenterYXSmoothL1
from yolo_lib.detectors.yolo_heads.losses.siou_box_loss import SIoUBoxLoss
from yolo_lib.detectors.yolo_heads.losses.sincos_losses import SinCosLoss
from yolo_lib.display_detections import display_yolo_tile
from yolo_lib.performance_metrics import get_default_performance_metrics
from yolo_lib.training_loop import TrainingLoop
from yolo_lib.model_storage import ModelStorage


OUTPUT_DIR = "./out/the_effect_of_noisy_labels/log_files"
MODEL_STORAGE_DIR = "./out/the_effect_of_noisy_labels/model_storage"

# Image size
TILE_SIZE = 512
IMG_H = TILE_SIZE
IMG_W = TILE_SIZE


# Epochs
MAX_EPOCHS = 100
EPOCHS_PER_DISPLAY = 150
EPOCHS_PER_CHECKPOINT = 50


# Train set size
IMGS_PER_EPOCH = 1600
BATCH_SIZE = 8


# Test set size
TEST_SET_SIZE = 400
NUM_DISPLAYED_TESTS = 0
NUM_SILENT_TESTS = TEST_SET_SIZE - NUM_DISPLAYED_TESTS


INACCURACIES = [0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 50, 60]
# INACCURACIES = [8, 15, 25, 35]
# INACCURACIES = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50, 60]
# INACCURACIES = [0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35]
# INACCURACIES = [40, 50, 60]
# INACCURACIES = [0]


OVERLAPPING = "Overlapping"
OVERLAPPING_WIDE = "OverlappingWide"
NON_OVERLAPPING = "NonOverlapping"
HEAD_VARIATIONS = [
    OVERLAPPING,
    OVERLAPPING_WIDE,
    NON_OVERLAPPING,
]

HEAD_VARIATION_PRETTY_NAMES = {
    OVERLAPPING: "Overlapping64",
    OVERLAPPING_WIDE: "Overlapping128",
    NON_OVERLAPPING: "NonOverlapping",
}


vessel_shape_cfg = VesselShapeCfg(
    vessel_length_low=28,
    vessel_length_high=36,
    vessel_width_low=8,
    vessel_width_high=14,
    rear_end_width_multiplier=0.6,
    middle_width_multiplier=1.4,
)

img_shape_cfg = ImgShapeCfg(
    img_h=TILE_SIZE,
    img_w=TILE_SIZE,
    max_fake_vessels=1,
    max_real_vessels=3,
)

def identity_collate(x):
    return x

def get_head(variation: str) -> YOLOHead:
    assert variation in HEAD_VARIATIONS
    if variation in [OVERLAPPING, OVERLAPPING_WIDE]:
        if variation == OVERLAPPING:
            multiplier = 2.3
            threshold = 2.0
            match_yx_weight = 0.5
            match_objectness_weight = 0.5
        else:
            multiplier = 4.3
            threshold = 4.0
            match_yx_weight = 0.3
            match_objectness_weight = 0.7


        print("variation", variation, "multiplier", multiplier, "threshold", threshold)
        return OverlappingCellYOLOHead(
            512,
            4,
            get_anchor_priors(4),
            multiplier,
            threshold,
            PointPotoMatchlossCfg(match_objectness_weight, match_yx_weight, None),
            SinCosLoss(False, False),
            SIoUBoxLoss(0.5, 0.5, CenterYXSmoothL1()),
            FocalLossCfg(neg_weight=0.3, pos_weight=1.0, gamma=2).build(),
            0.5,
            0.30,
            0.2,
        )
    else:
        return LocalMatchingYOLOHeadCfg(
            512,
            4,
            get_anchor_priors(4),

            0.5,
            0.5,
            0.0,
            0.0,

            0.5,
            0.15,
            0.15,
            0.2,

            allow_180=False
        ).build()

def get_model(variation: str) -> BaseDetector:
    return AuxiliaryHeadYOLOF(
        BackboneCfg(1, 34).build(),
        EncoderConfig.default(),
        MultilayerAttentionCfg(64,2),
        get_head(variation),
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


def get_train_ds(max_yx_error_px: int) -> SyntheticDs:
    train_ds_weakness_cfg = DsWeaknessCfg(
        rotation_known_probability=0.1,
        hw_known_probability=0.4,
        max_yx_error_px=float(max_yx_error_px),
    )
    train_ds_cfg = FakeDataCfg.make(vessel_shape_cfg, img_shape_cfg, train_ds_weakness_cfg)
    return SyntheticDs(IMGS_PER_EPOCH, train_ds_cfg, repeat=False)


def run_experiment(
    variation: str,
    max_yx_error_px: int,
    test_dl: DataLoader,
    displayed_test_dl: DataLoader,
) -> None:
    assert variation in HEAD_VARIATIONS

    # Load model storage
    model_storage = ModelStorage(MODEL_STORAGE_DIR)

    # Experiment name
    name = f"{variation}+MaxError({max_yx_error_px})"
    print("name", name)

    # Create training set data loader
    train_ds = get_train_ds(max_yx_error_px)
    train_dl = DataLoader(train_ds, BATCH_SIZE, collate_fn=YOLOTileStack.stack_tiles)

    # Create model
    model = get_model(variation)

    # Create optimizer
    N = 10
    P = 0.95
    lr_scheduler_lambda = lambda i: ((i+1) / N) if i < N else P ** (i + 1 - N)
    optimizer = Adam(model.parameters(), 5e-5)

    # Create training loop
    training_loop = TrainingLoop(
        name,
        model,
        optimizer,
        [SAT(0.2, 10, 0.01, 0.1)],
        get_default_performance_metrics(),
        [],
        lr_scheduler_lambda,
        MAX_EPOCHS,
    )

    # Run training loop
    training_loop.run(
        EPOCHS_PER_DISPLAY,
        EPOCHS_PER_CHECKPOINT,
        model_storage,
        train_dl,
        test_dl,
        displayed_test_dl,
        OUTPUT_DIR
    )

def postprocess():
    pass

def read_f2(epoch_log_obj) -> float:
    return epoch_log_obj["performance_metrics"]["Distance-AP"]["F2"]["F2"]

def read_ap(epoch_log_obj) -> float:
    return epoch_log_obj["performance_metrics"]["Distance-AP"]["AP"]


def read_best(variation: str, max_yx_error_px: int, metric: str) -> float:
    """Returns the best Distance-AP score for a given configuration"""
    name = f"{variation}+MaxError({max_yx_error_px})"
    path = os.path.join(OUTPUT_DIR, name, "training_log.json")
    with open(path, "r") as F:
        as_json = json.load(F)

    if len(as_json) == 0:
        return None
    else:
        print("len", len(as_json))
        reader = {"F2": read_f2, "AP": read_ap}[metric]
        return max([
            reader(epoch_log_obj)
            for epoch_log_obj in as_json
        ])

def plot():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    # fig, axs = plt.subplots(1, 2, constrained_layout=True, gridspec_kw={'wspace': 0.1, 'hspace': 0.1})

    # Figure "architecture"
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=2, ncols=2)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[:, 1])

    # Labels
    fig.suptitle('The effect of inaccurate position labels', fontsize=16)
    ax0.set(ylabel="Distance-AP", xlabel="Max label inaccuracy (px)")
    ax1.set(ylabel="F2", xlabel="Max label inaccuracy (px)")

    # Plot lines
    for variation in [OVERLAPPING, OVERLAPPING_WIDE, NON_OVERLAPPING]:
        print("variation", variation)
        pretty_name = HEAD_VARIATION_PRETTY_NAMES[variation]
        ap_measures = [read_best(variation, max_yx_error_px, "AP") for max_yx_error_px in INACCURACIES]
        f2_measures = [read_best(variation, max_yx_error_px, "F2") for max_yx_error_px in INACCURACIES]
        ax0.plot(INACCURACIES, ap_measures, label=pretty_name)
        ax1.plot(INACCURACIES, f2_measures, label=pretty_name)

    ax0.legend()
    ax1.legend()

    fig.savefig("out/the_effect_of_noisy_labels/figs/the_effect_of_noisy_labels.pdf")
    fig.savefig("out/the_effect_of_noisy_labels/figs/the_effect_of_noisy_labels.png")


def plot_paper():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Figure "architecture"
    fig = plt.figure(figsize=(5, 5))
    gs = GridSpec(nrows=1, ncols=1)
    ax0 = fig.add_subplot(gs[:, 0])
    # ax1 = fig.add_subplot(gs[:, 1])

    # Labels
    # fig.suptitle('The effect of inaccurate position labels on dAP', fontsize=14)
    ax0.set(ylabel="dAP (%)", xlabel="Max label inaccuracy (px)")
    # ax1.set(ylabel="Predicted positions' inaccuracy (px)", xlabel="Max label inaccuracy (px)")

    # Plot lines
    for variation in [OVERLAPPING, OVERLAPPING_WIDE, NON_OVERLAPPING]:
        print("variation", variation)
        pretty_name = HEAD_VARIATION_PRETTY_NAMES[variation]

        best_epoch_objs = []
        for max_yx_error_px in INACCURACIES:
            name = f"{variation}+MaxError({max_yx_error_px})"
            path = os.path.join(OUTPUT_DIR, name, "training_log.json")
            with open(path, "r") as F:
                as_json = json.load(F)

            best_epoch_objs.append(max(as_json, key=lambda log_obj: log_obj["performance_metrics"]["Distance-AP"]["AP"]))


        print(json.dumps(best_epoch_objs, indent=2))
        dap_measures = [best_epoch_obj["performance_metrics"]["Distance-AP"]["AP"] * 100 for best_epoch_obj in best_epoch_objs]
        dist_measures = [best_epoch_obj["performance_metrics"]["Distance-AP"]["mean_center_distance"] for best_epoch_obj in best_epoch_objs]
        siou_measures = [best_epoch_obj["performance_metrics"]["Distance-AP"]["avg_iou"] for best_epoch_obj in best_epoch_objs]
        epochs = [best_epoch_obj["epoch"] for best_epoch_obj in best_epoch_objs]

        # ap_measures = [read_best(variation, max_yx_error_px, "AP") for max_yx_error_px in INACCURACIES]
        # f2_measures = [read_best(variation, max_yx_error_px, "F2") for max_yx_error_px in INACCURACIES]
        ax0.plot(INACCURACIES, dap_measures, label=pretty_name)
        # ax1.plot(INACCURACIES, dist_measures, label=pretty_name)

    ax0.legend()
    # ax1.legend()

    fig.savefig("out/the_effect_of_noisy_labels/figs/the-effect-of-noisy-labels-paper.pdf")
    fig.savefig("out/the_effect_of_noisy_labels/figs/the-effect-of-noisy-labels-paper.png")

def train_img_samples():
    folder = "out/the_effect_of_noisy_labels/train_img_samples/"
    for max_yx_error_px in INACCURACIES:
        ds = get_train_ds(max_yx_error_px)
        for tile, idx in zip(ds, range(10)):
            path = os.path.join(folder, f"error({max_yx_error_px})-idx({idx})")
            display_yolo_tile(tile, path)

def main():

    # Create test dataset
    test_ds_weakness_cfg = DsWeaknessCfg(
        rotation_known_probability=1.0,
        hw_known_probability=1.0,
        max_yx_error_px=0.0,
    )
    test_ds_cfg = FakeDataCfg.make(vessel_shape_cfg, img_shape_cfg, test_ds_weakness_cfg)
    test_ds = SyntheticDs(NUM_SILENT_TESTS, test_ds_cfg, repeat=True)
    displayed_test_ds = SyntheticDs(NUM_DISPLAYED_TESTS, test_ds_cfg, repeat=True)
    displayed_test_dl = DataLoader(displayed_test_ds, BATCH_SIZE, collate_fn=identity_collate)
    test_dl = DataLoader(test_ds, BATCH_SIZE, collate_fn=identity_collate)

    # Run experiments
    for max_yx_error_px in INACCURACIES:
        for variation in HEAD_VARIATIONS:
            torch.cuda.empty_cache()
            run_experiment(variation, max_yx_error_px, test_dl, displayed_test_dl)

if __name__ == "__main__":
    print("Args", sys.argv)
    assert len(sys.argv) in [1, 2]
    arg = "train" if len(sys.argv) == 1 else sys.argv[1]
    assert arg in ["train", "plot", "train_img_samples", "plot_paper"]
    if arg == "train":
        main()
    if arg == "plot":
        plot()
    if arg == "train_img_samples":
        train_img_samples()
    if arg == "plot_paper":
        plot_paper()
