from dataclasses import dataclass
import json
import os
from typing import Callable, Dict, List, Optional
import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from data.data_generator import TrainDs, generate_image
from models.base_model import PeriodicRegression
from models.dv_regression import PDV
from models.grey_code import GreyCode3, GreyCode5
from models.logistic_regression import LogisticRegression
from models.circular_smooth_label import CSL_32_01
from tqdm import tqdm


USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
BATCH_SIZE = 16
MAX_EPOCHS = 5
EPOCH_SIZE = 400
OUT_FOLDER = "../out/periodic_regression/"


def run_experiment(
    max_angle_error_degrees: float,
    train_set_size_limit: Optional[int],
    experiment_name: str,
    models: List[Callable[[], PeriodicRegression]],
    max_epochs: int=MAX_EPOCHS
):
    train_ds = TrainDs(EPOCH_SIZE, max_angle_error_degrees, train_set_size_limit)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=False)
    for model_constuctor in models:
        model = model_constuctor()
        model = model.cuda() if USE_GPU else model
        model_name = model.__class__.__name__
        fname = os.path.join(OUT_FOLDER, f"{experiment_name}_{model_name}.json")
        train_loop(model, train_dl, max_epochs, model_name, fname)

def train_loop(
    model: PeriodicRegression,
    train_dl: DataLoader,
    epochs: int,
    name: str,
    out_fname: str
) -> None:
        optimizer = torch.optim.Adam(model.parameters())
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        parameters = list(model.parameters())
        log = []
        for epoch in range(epochs):

            # Train
            model.train(True)
            loss_sum = 0
            for (images, labels) in tqdm(train_dl, f"Epoch {epoch} (training) {name}"):
                labels = labels.cuda() if USE_GPU else labels 
                loss = model.loss(images, labels)
                loss.backward(inputs=parameters)
                loss_sum += loss.detach()
                optimizer.step()
                optimizer.zero_grad()

            # Evaluate
            model.train(False)
            log_obj = {
                "epoch": epoch,
                "name": name,
                "loss_sum": float(loss_sum),
                "performance_metrics": performance_metrics(model),
                "min_prediction": float(model.min_prediction),
                "max_prediction": float(model.max_prediction),
                "min_label": float(model.min_label),
                "max_label": float(model.max_label),
            }
            print(json.dumps(log_obj, indent=2))
            log.append(log_obj)
            store_log(out_fname, log)

            # Update learning rate
            lr_scheduler.step()

@torch.inference_mode()
def evaluate(model: PeriodicRegression) -> torch.Tensor:
    """Returns an absolute error at 0, 1, ..., 179 degrees"""
    diffs = []
    for true_angle_degrees in range(180):
        image = generate_image(true_angle_degrees)[None]
        predicted_angle_01 = model.infer(image)
        assert 0 <= predicted_angle_01 <= 1
        predicted_angle_degrees = predicted_angle_01 * 180
        max_angle = max(true_angle_degrees, predicted_angle_degrees)
        min_angle = min(true_angle_degrees, predicted_angle_degrees)
        abs_diff = min(max_angle - min_angle, 180 - max_angle + min_angle)
        assert 0.0 <= abs_diff <= 90.0 
        diffs.append(abs_diff)

    return torch.tensor(diffs)

@torch.inference_mode()
def performance_metrics(model: PeriodicRegression) -> Dict[str, float]:
    errors = evaluate(model)
    assert errors.shape == (180,)
    avg_error = errors.mean()
    return {
        "avg_error": float(avg_error),
        "avg_error_000_030": float(errors[000: 30].mean()),
        "avg_error_030_060": float(errors[ 30: 60].mean()),
        "avg_error_060_090": float(errors[ 60: 90].mean()),
        "avg_error_090_120": float(errors[ 90:120].mean()),
        "avg_error_120_150": float(errors[120:150].mean()),
        "avg_error_150_180": float(errors[150:180].mean()),

        "max_error_000_030": float(errors[000: 30].amax()),
        "max_error_030_060": float(errors[ 30: 60].amax()),
        "max_error_060_090": float(errors[ 60: 90].amax()),
        "max_error_090_120": float(errors[ 90:120].amax()),
        "max_error_120_150": float(errors[120:150].amax()),
        "max_error_150_180": float(errors[150:180].amax()),
    }

def store_log(fname: str, log: List):
    with open(fname, "w") as F:
        json.dump(log, F, indent=2)

def experiment_1():
    """
    No noise in training data
    Entire range is covered
    Should be the easiest test possible, and show if the different models work at all
    """
    run_experiment(
        0.0,
        None,
        "experiment1",
        [LogisticRegression, PDV, GreyCode3, GreyCode5, CSL_32_01],
        # [GreyCode5],
    )


def experiment_2():
    """
    Experiment 2
    Some noise in training data, but otherwise identical to experiment 1:
    Logistic regression should get big problems on the wrapping p
    oint here
    """
    run_experiment(
        5.0,
        25,
        "experiment2_25",
        [PDV, GreyCode3, GreyCode5],
        10,
    )
    run_experiment(
        5.0,
        50,
        "experiment2_50",
        [PDV, GreyCode3, GreyCode5],
        10,
    )
    run_experiment(
        5.0,
        100,
        "experiment2_100",
        [PDV, GreyCode3, GreyCode5],
        10,
    )
    # run_experiment(
    #     5.0,
    #     200,
    #     "experiment2_200",
    #     [PDV, GreyCode],
    # )
    # run_experiment(
    #     5.0,
    #     400,
    #     "experiment2_400",
    #     [PDV, GreyCode],
    # )


def experiment_3():
    run_experiment(
        5.0,
        None,
        "experiment3_15_inf",
        [LogisticRegression],
        50,
    )


if __name__ == "__main__":
    experiment_1()
    experiment_2()
    experiment_3()

