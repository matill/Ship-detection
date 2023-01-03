from __future__ import annotations
import json
import torch
import os
from typing import Any, Callable, Dict, List, Optional
from torch import optim
from torch.utils.data.dataloader import DataLoader
from yolo_lib.data.yolo_tile import YOLOTile
from yolo_lib.data_augmentation.data_augmentation import DataAugmentation
from yolo_lib.detectors.managed_architectures.base_detector import BaseDetector
from yolo_lib.cfg import USE_GPU
from yolo_lib.util.model_storage import ModelStorage
from yolo_lib.performance_metrics.base_performance_metric import BasePerformanceMetric
from yolo_lib.util.timer import Timer
from yolo_lib.data.dataclasses import DetectionBlock, YOLOTileStack
from tqdm import tqdm
from yolo_lib.optimization_criteria import OptimizationCriteria


class TrainingLoop(torch.nn.Module):
    def __init__(
        self,
        name: str,
        model: BaseDetector,
        optimizer: optim.Optimizer,
        data_augmentations: List[DataAugmentation],
        performance_metrics: BasePerformanceMetric,
        lr_scheduler_lambda: Callable[[int], float],
        max_epochs: int,
        train_dl: DataLoader,
        test_dl: DataLoader,
        optimization_criterias: List[OptimizationCriteria],
    ) -> None:
        super().__init__()
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.data_augmentations = data_augmentations
        self.performance_metrics = performance_metrics
        self.lr_scheduler_lambda = lr_scheduler_lambda
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_scheduler_lambda)
        self.max_epochs = max_epochs
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.optimization_criterias = optimization_criterias

        # Loop state that should be stored at checkpoints
        self.epoch = 0
        self.tasks = []
        self.epoch_log_object = {}

    def state_dict(self) -> Dict[str, Any]:
        """Overload Module.state_dict() to actually work"""
        return {
            # Sub-modules
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),

            # Loop state
            "epoch": self.epoch,
            "tasks": self.tasks,
            "epoch_log_object": self.epoch_log_object,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Overload Module.load_state_dict() to actually work"""
        # Sub-modules
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

        # Loop state
        self.epoch = state_dict["epoch"]
        self.tasks = state_dict["tasks"]
        self.epoch_log_object = state_dict["epoch_log_object"]

    def get_next_task(self) -> Optional[str]:
        if len(self.tasks) == 0:
            print("resetting tasks")
            self.tasks = [
                "RESET_CRT_LOG_OBJECT",
                "TRAIN_EPOCH",
                "CHECKPOINT",
                "EVALUATE",
                "CHECKPOINT",
                "CHECKPOINT_BEST_METRICS",
                "INCREMENT_LR_SCHEDULER",
                "LOG_STORE",
                "INCREMENT_EPOCH",
            ]

        if self.max_epochs <= self.epoch:
            return None
        else:
            return self.tasks.pop(0)

    def read_log(self, output_directory_base_path):
        experiment_output_folder = os.path.join(output_directory_base_path, self.name)
        training_log_fname = os.path.join(experiment_output_folder, "training_log.json")
        display_detection_folder = os.path.join(experiment_output_folder, "display_detections")
        print("display_detection_folder", display_detection_folder)
        os.makedirs(display_detection_folder, exist_ok=True)
        if self.epoch > 0:
            with open(training_log_fname, "r") as F:
                training_log = json.load(F)
                return [x for x in training_log if x["epoch"] < self.epoch]
        else:
            return None

    def run(
        self,
        # epochs_per_display: int,
        # epochs_per_checkpoint: Optional[int],
        model_storage: ModelStorage,

        # train_dl: DataLoader,
        # test_dl: DataLoader,
        # displayed_test_dl: DataLoader,

        output_directory_base_path: str,
    ):
        # Load latest stored version of model
        model_storage.load_model(self, self.name, not_exists_ok=True)

        # Output folders and files + training log
        experiment_output_folder = os.path.join(output_directory_base_path, self.name)
        training_log_fname = os.path.join(experiment_output_folder, "training_log.json")
        display_detection_folder = os.path.join(experiment_output_folder, "display_detections")
        print("display_detection_folder", display_detection_folder)
        os.makedirs(display_detection_folder, exist_ok=True)
        if self.epoch > 0:
            with open(training_log_fname, "r") as F:
                training_log = json.load(F)
                training_log = [x for x in training_log if x["epoch"] < self.epoch]
        else:
            training_log = []

        while True:
            task = self.get_next_task()
            print("\ntask", task, "epoch", int(self.epoch))
            if task is None:
                break
            elif task == "RESET_CRT_LOG_OBJECT":
                self.reset_crt_log_object()
            elif task == "TRAIN_EPOCH":
                self.train_epoch()
            elif task == "CHECKPOINT":
                self.checkpoint(model_storage)
            elif task == "EVALUATE":
                self.evaluate()
            elif task == "INCREMENT_LR_SCHEDULER":
                self.increment_lr_scheduler()
            elif task == "LOG_STORE":
                self.log_store(training_log, training_log_fname)
            elif task == "INCREMENT_EPOCH":
                self.increment_epoch()
            elif task == "CHECKPOINT_BEST_METRICS":
                self.checkpoint_best_metrics(model_storage, training_log)
            else:
                raise ValueError(f"Unexpected task {task}")

        self.checkpoint(model_storage)

    def checkpoint(self, model_storage: ModelStorage):
        model_storage.store_model(self, self.name)

    def checkpoint_best_metrics(self, model_storage: ModelStorage, training_log: List[Any]):
        for optimization_criteria in self.optimization_criterias:
            is_crt_best = optimization_criteria.is_crt_best(training_log, self.epoch_log_object)
            short_name = optimization_criteria.short_name
            print(f"Optimization criteria {short_name} {'broke' if is_crt_best else 'did not break'} record this epoch ({self.epoch})")
            if is_crt_best:
                model_storage.store_model(self, self.name, tag=optimization_criteria.short_name)

    def increment_epoch(self):
        self.epoch += 1

    def increment_lr_scheduler(self):
        self.lr_scheduler.step()

    def log_store(self, training_log: List, training_log_fname: str):
        print(self.name)
        self.epoch_log_object["epoch"] = int(self.epoch_log_object["epoch"])
        print(self.epoch_log_object)
        print(json.dumps(self.epoch_log_object, indent=2))
        training_log.append(self.epoch_log_object)
        with open(training_log_fname, "w") as F:
            json.dump(training_log, F)

    def reset_crt_log_object(self):
        self.epoch_log_object = {
            "epoch": self.epoch
        }

    def train_epoch(self) -> None:
        # Start timers
        start_timer = Timer()
        end_timer = Timer()
        start_timer.record()

        self.model.train(True)
        epoch_loss_sum = 0.0
        epoch_loss_subterm_sums = {}
        desc = f"Epoch ({self.epoch}) training"
        tqdm_dataloader = tqdm(self.train_dl, desc=desc, smoothing=0.01)
        num_augmentation_sum = 0
        for yolo_tiles in tqdm_dataloader:
            yolo_tiles: YOLOTileStack = yolo_tiles.to_device(USE_GPU)

            # Apply data augmentations
            # tiles_before = yolo_tiles
            yolo_tiles, num_augmentations = DataAugmentation.apply_list(self.data_augmentations, yolo_tiles, self.model, int(self.epoch))
            num_augmentation_sum += num_augmentations

            # Compute loss and do SGD step
            loss, loss_subterms = self.model.compute_loss(yolo_tiles)
            if str(float(loss)) in ["nan", "inf", "-inf"]:
                print("loss is nan or inf: ", loss)
                print(loss_subterms)
                assert False

            loss.backward(inputs=list(self.model.parameters()))
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Increment loss for entire epoch
            epoch_loss_sum += float(loss)
            for (subterm_key, subterm_loss) in loss_subterms.items():
                float_val = float(subterm_loss)
                if subterm_key in epoch_loss_subterm_sums:
                    epoch_loss_subterm_sums[subterm_key] += float_val
                else:
                    epoch_loss_subterm_sums[subterm_key] = float_val

            # Update tqdm postfix (loss terms)
            tqdm_postfix = {"loss": epoch_loss_sum, "augments": num_augmentation_sum}
            tqdm_dataloader.set_postfix(tqdm_postfix)

        # Add metrics to log
        end_timer.record()
        Timer.sync()
        self.epoch_log_object["epoch_loss_sum"] = epoch_loss_sum
        self.epoch_log_object["epoch_loss_subterm_sums"] = epoch_loss_subterm_sums
        self.epoch_log_object["train_elapsed_seconds"] = Timer.elapsed_seconds(start_timer, end_timer)

    @torch.inference_mode()
    def evaluate(self,) -> None:
        # Start timers
        start_timer = Timer()
        end_timer = Timer()
        start_timer.record()

        # Enter evaluation mode. Reset performance metric counter
        self.model.train(False)
        self.performance_metrics.reset()

        # Compute performance for test set
        desc = f"Epoch ({self.epoch}) evaluation"
        for tiles in tqdm(self.test_dl, desc=desc, smoothing=0.01):
            # Check that the tile list has the right type
            tiles: List[YOLOTile] = tiles
            assert isinstance(tiles, list) and all((isinstance(tile, YOLOTile) for tile in tiles))

            # Create batch of images, and get a DetectionBlock for each image
            images = torch.cat([tile.image for tile in tiles])
            device_images = images.cuda() if USE_GPU else images
            detection_grid = self.model.detect_objects(device_images)
            detection_blocks = [
                detection_grid_i.as_detection_block()
                for detection_grid_i in detection_grid.split_by_image()
            ]

            # Increment performance metrics for all images
            annotation_blocks = [tile.annotations.to_device(USE_GPU) for tile in tiles]
            for (annotations, detections) in zip(annotation_blocks, detection_blocks):
                self.performance_metrics.increment(detections, annotations)

        # Finalize performance metrics
        end_timer.record()
        Timer.sync()
        self.epoch_log_object["performance_metrics"] = self.performance_metrics.finalize()
        self.epoch_log_object["test_elapsed_seconds"] = Timer.elapsed_seconds(start_timer, end_timer)

