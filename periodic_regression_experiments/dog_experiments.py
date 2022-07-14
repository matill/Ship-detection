from dataclasses import dataclass
import json
import os
import sys
from typing import List, Tuple
import torch
from torch import nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from models.dv_regression import PDV, NormalizePDV, ProjectedPDV, ProjectedPDV_175
from models.grey_code import GreyCode3PlusOffset, GreyCode5PlusOffset, GreyCode7PlusOffset
from models.logistic_regression import LogisticRegression
from models.circular_smooth_label import CSL_128Bit_01Gaussian, CSL_128Bit_01WindowWidth_PlusOffset, CSL_128Bit_03Gaussian, CSL_128Bit_03WindowWidth_PlusOffset, CSL_256Bit_01Gaussian, CSL_256Bit_03Gaussian, CSL_32Bit_01Gaussian, CSL_32Bit_01WindowWidth_PlusOffset, CSL_32Bit_03Gaussian, CSL_32Bit_03WindowWidth_PlusOffset, CSL_256Bit_01WindowWidth_PlusOffset, CSL_256Bit_03WindowWidth_PlusOffset
from data.dogs_dataset import DogDataset
from data.line_dataset import LineDataset
from training_loop import training_loop
from plot_configs import PlotConfig, PLOT_CONFIGS

USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
BATCH_SIZE = 16
EPOCH_SIZE = 400
OUT_FOLDER = "../out/periodic_regression/dog_experiments/"
FIG_FOLDER = os.path.join(OUT_FOLDER, "figs")


INACCURACY_LEVELS = [
    0,
    10,
    50,
    100,
]


EVALUATION_RANGES = [
    # Small ranges
    {"min": 0, "max": 30},
    {"min": 30, "max": 60},
    {"min": 60, "max": 90},
    {"min": 90, "max": 120},
    {"min": 120, "max": 150},
    {"min": 150, "max": 180},
    {"min": 180, "max": 210},
    {"min": 210, "max": 240},
    {"min": 240, "max": 270},
    {"min": 270, "max": 300},
    {"min": 300, "max": 330},
    {"min": 330, "max": 360},

    # Full range
    {"min": 0, "max": 360},
]

MODEL_CLASSES = [
    # Gaussian window function
    CSL_32Bit_01Gaussian,
    CSL_128Bit_01Gaussian,
    CSL_256Bit_01Gaussian,
    CSL_32Bit_03Gaussian,
    CSL_128Bit_03Gaussian,
    CSL_256Bit_03Gaussian,

    ProjectedPDV,
    PDV,
    NormalizePDV,
    ProjectedPDV_175,

    GreyCode3PlusOffset,
    GreyCode5PlusOffset,
    GreyCode7PlusOffset,

    CSL_32Bit_01WindowWidth_PlusOffset,
    CSL_32Bit_03WindowWidth_PlusOffset,
    CSL_128Bit_01WindowWidth_PlusOffset,
    CSL_128Bit_03WindowWidth_PlusOffset,
    CSL_256Bit_01WindowWidth_PlusOffset,
    CSL_256Bit_03WindowWidth_PlusOffset,

    LogisticRegression,
]


def run_experiment(experiment_name: str, train_ds: Dataset, test_ds: Dataset, angle_range: int):
    for ModelClass in MODEL_CLASSES:
        model = ModelClass()
        training_loop(model, train_ds, test_ds, experiment_name, angle_range)

def line_experiment():
    train_ds = LineDataset.get_train_ds(EPOCH_SIZE)
    test_ds = LineDataset.get_test_ds(0.1)
    run_experiment("line_experiment", train_ds, test_ds, 180)

def dog_experiment(max_inaccuracy: int):
    train_ds = DogDataset.get_train_ds(float(max_inaccuracy))
    test_ds = DogDataset.get_test_ds()
    run_experiment(f"dog_experiment(max_inaccuracy={max_inaccuracy})", train_ds, test_ds, 360)

def train():
    # line_experiment()
    for max_inaccuracy in INACCURACY_LEVELS:
        dog_experiment(max_inaccuracy)

def display_subsample():
    train_ds = DogDataset.get_train_ds()

    image, rotation = train_ds[10]
    # image, rotation = train_ds[10]

    fig, axs = plt.subplots(1, 3)

    for i in range(3):
        image, rotation = train_ds[10 + i]
        axs[i].imshow(image.permute(1, 2, 0).cpu())
        axs[i].tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False
        )
        axs[i].set_title(f"Rotation: {rotation:.1f}°")
    plt.show()
    exit()

def read_median(epoch_log_obj) -> float:
    return epoch_log_obj["performance_metrics"]["median_diff_0_360"]

def read_mean(epoch_log_obj) -> float:
    return epoch_log_obj["performance_metrics"]["mean_diff_0_360"]

def read_best(ModelClass, max_inaccuracy: int, metric: str) -> float:
    """Returns the best Distance-AP score for a given configuration"""
    experiment_name = f"dog_experiment(max_inaccuracy={max_inaccuracy})"
    path = os.path.join(OUT_FOLDER, experiment_name, "log_files", f"{ModelClass.__name__}.json")
    print("path", path)
    with open(path, "r") as F:
        as_json = json.load(F)

    if len(as_json) == 0:
        return None
    else:
        print("len", len(as_json))
        reader = {"median": read_median, "mean": read_mean}[metric]
        return min([
            reader(epoch_log_obj)
            for epoch_log_obj in as_json
        ])

# def plot_helper(plot_name: str, model_classes: List):
def plot_helper(plot_cfg: PlotConfig):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Figure "architecture"
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=1, ncols=2)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[:, 1])
    fig.tight_layout(pad=2.0)

    # Labels
    # fig.suptitle(plot_title, fontsize=16)
    xlabel = "Noise distribution length (degrees)"
    ax0.set(ylabel="Median abs. test error (degrees)", xlabel=xlabel)
    ax1.set(ylabel="Avgerage abs. test error (degrees)", xlabel=xlabel)

    # Plot lines
    # for variation in [OVERLAPPING, OVERLAPPING_WIDE, NON_OVERLAPPING]:
    for ModelClass, pretty_name in plot_cfg.members:
        # model = ModelClass()
        formal_name = ModelClass.__name__
        print("(formal_name, pretty_name)", (formal_name, pretty_name))
        median_measures = [read_best(ModelClass, max_inaccuracy, "median") for max_inaccuracy in INACCURACY_LEVELS]
        mean_measures = [read_best(ModelClass, max_inaccuracy, "mean") for max_inaccuracy in INACCURACY_LEVELS]
        ax0.plot(INACCURACY_LEVELS, median_measures, label=pretty_name)
        ax1.plot(INACCURACY_LEVELS, mean_measures, label=pretty_name)

    ax0.legend()
    ax1.legend()
    fig.savefig(os.path.join(FIG_FOLDER, f"{plot_cfg.name}.pdf"))
    fig.savefig(os.path.join(FIG_FOLDER, f"{plot_cfg.name}.png"))


def plot():
    for plot_cfg in PLOT_CONFIGS:
        plot_helper(plot_cfg)
    # plot_helper("rotation_all", "Rotation regression benchmark (all models)", MODEL_CLASSES)
    # plot_helper("rotation_pdv", "Comparing ADV loss functions", PDV_CLASSES)
    # plot_helper("rotation_gcl", "Rotation regression benchmark (GCL)", GCL_CLASSES)
    # plot_helper("rotation_csl", "Rotation regression benchmark (CSL)", CSL_CLASSES)
    # plot_helper("rotation_clean", "Rotation regression benchmark (summary)", CLEAN_MODEL_CLASSES)

if __name__ == "__main__":
    print("Args", sys.argv)
    assert len(sys.argv) in [1, 2]
    arg = "train" if len(sys.argv) == 1 else sys.argv[1]
    assert arg in ["train", "plot", "table"]
    if arg == "train":
        train()
    if arg == "plot":
        plot()


