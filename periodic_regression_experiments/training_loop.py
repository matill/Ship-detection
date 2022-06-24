import json
import os
from typing import Any, Dict
import torch
from torch import nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.base_model import PeriodicRegression
from data.line_dataset import LineDataset

USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
BATCH_SIZE = 16
MAX_EPOCHS = 30
EPOCH_SIZE = 400
OUT_FOLDER = "../out/periodic_regression/dog_experiments/"
MODEL_STORAGE_FOLDER = os.path.join(OUT_FOLDER, "model_storage")
LOG_FILE_FOLDER = os.path.join(OUT_FOLDER, "log_files")

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

@torch.inference_mode()
def evaluate(model: PeriodicRegression, test_dl: DataLoader, epoch: int, angle_range: int) -> Dict[str, Any]:
    # Get lists of all predictions and all labels
    name = model.__class__.__name__
    labels_all = []
    predictions_all = []
    for (images, labels_01) in tqdm(test_dl, f"Epoch {epoch} (evaluation) {name}"):
        labels_degrees = labels_01 * angle_range
        predictions_01 = model.infer(images)
        predictions_degrees = predictions_01 * angle_range
        labels_all.append(labels_degrees)
        predictions_all.append(predictions_degrees.cpu())

    # Flatten batched lists into two large tensors
    label_tensor = torch.cat(labels_all)
    prediction_tensor = torch.cat(predictions_all)
    (n, ) = label_tensor.shape
    assert prediction_tensor.shape == (n, )

    # Absolute diff for each label-truth
    diffs_1 = torch.abs(label_tensor - prediction_tensor)
    del prediction_tensor
    diffs = torch.min(diffs_1, angle_range - diffs_1)
    assert diffs.shape == (n, )

    # For each evaluation range, get mean, median and max error
    metric_dict = {}
    for range in EVALUATION_RANGES:
        # Get absolute differences from the current range
        min_degrees = range["min"]
        max_degrees = range["max"]
        range_bitmap = ((min_degrees <= label_tensor) & (label_tensor < max_degrees))
        diffs_in_range = diffs[range_bitmap]

        # If bitmap is empty, continue
        if range_bitmap.logical_not().all():
            continue

        # Get mean, median and max in range. Add to dict
        range_mean = diffs_in_range.mean()
        range_max = diffs_in_range.amax()
        range_median = diffs_in_range.median()
        metric_dict[f"mean_diff_{min_degrees}_{max_degrees}"] = float(range_mean)
        metric_dict[f"max_diff_{min_degrees}_{max_degrees}"] = float(range_max)
        metric_dict[f"median_diff_{min_degrees}_{max_degrees}"] = float(range_median)

    return metric_dict


class TrainingLoop:
    def __init__(
        self,
        model: PeriodicRegression,
        experiment_name: str,
        angle_range: int,
    ) -> None:
        self.model = model.cuda()
        self.name = model.__class__.__name__
        self.experiment_name = experiment_name
        self.angle_range = angle_range
        self.optimizer = torch.optim.Adam(model.parameters())
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)
        self.log = []
        self.epoch = 0

    def get_model_storage_fname(self) -> str:
        folder = os.path.join(OUT_FOLDER, self.experiment_name, "model_storage")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{self.name}.pt")

    def get_log_fname(self) -> str:
        folder = os.path.join(OUT_FOLDER, self.experiment_name, "log_files")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{self.name}.json")

    def load(self):
        try:
            state_dict = torch.load(self.get_model_storage_fname())
            self.model.load_state_dict(state_dict["model"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
            self.log = state_dict["log"]
            self.epoch = state_dict["epoch"]
        except FileNotFoundError:
            pass

    def store(self):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "log": self.log,
                "epoch": self.epoch,
            },
            self.get_model_storage_fname()
        )

    def run(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
    ):
        # Create data loaders
        train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=(not isinstance(train_ds, LineDataset)))
        test_dl = DataLoader(test_ds, BATCH_SIZE, shuffle=False)
        parameters = list(self.model.parameters())
        while self.epoch < MAX_EPOCHS:
            # Train
            self.model.train(True)
            loss_sum = 0
            for (images, labels_01) in tqdm(train_dl, f"Epoch {self.epoch} (training) {self.name}"):
                labels_01 = labels_01.cuda() if USE_GPU else labels_01
                loss = self.model.loss(images, labels_01)
                loss.backward(inputs=parameters)
                loss_sum += loss.detach()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Evaluate
            self.model.train(False)
            log_obj = {
                "epoch": self.epoch,
                "name": self.name,
                "loss_sum": float(loss_sum),
                "performance_metrics": evaluate(self.model, test_dl, self.epoch, self.angle_range),
                "min_prediction": float(self.model.min_prediction),
                "max_prediction": float(self.model.max_prediction),
                "min_label": float(self.model.min_label),
                "max_label": float(self.model.max_label),
            }
            print(json.dumps(log_obj, indent=2))
            self.log.append(log_obj)
            with open(self.get_log_fname(), "w") as F:
                json.dump(self.log, F)

            # Update learning rate and checkpoint
            self.lr_scheduler.step()
            self.epoch += 1
            self.store()


def training_loop(
    model: PeriodicRegression,
    train_ds: Dataset,
    test_ds: Dataset,
    experiment_name: str,
    angle_range: int,
):
    training_loop = TrainingLoop(model, experiment_name, angle_range)
    training_loop.load()
    training_loop.run(train_ds, test_ds)
