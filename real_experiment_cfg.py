from dataclasses import dataclass
from typing import List
import torch

from yolo_lib.detectors.cfg_types.dilated_encoder_cfg import EncoderConfig
from yolo_lib.data_augmentation.data_augmentation import DataAugmentation
from yolo_lib.cfg import USE_GPU
from yolo_lib.detectors.cfg_types.dilated_encoder_cfg import EncoderConfig
from yolo_lib.detectors.cfg_types.detector_cfg import DetectorCfg
from yolo_lib.performance_metrics import get_default_performance_metrics
from yolo_lib.performance_metrics.base_performance_metric import BasePerformanceMetric
from yolo_lib.training_loop import TrainCallback, TrainingLoop


@dataclass
class Exp:
    name: str
    detector_cfg: DetectorCfg
    data_augmentations: List[DataAugmentation] = None
    performance_metrics: BasePerformanceMetric = None
    callbacks: List[TrainCallback] = None
    lr_warmup_epochs: int = 10
    lr_decay_factor: float = 0.95
    base_lr: float = 0.0001

    def default(self, value, default):
        if value is None:
            return default
        else:
            return value

    def get_training_loop(self, max_epochs: int) -> TrainingLoop:
        model = self.detector_cfg.build()
        if USE_GPU:
            model = model.cuda()

        optimizer = torch.optim.Adam(list(model.parameters()), lr=self.base_lr)
        N = self.lr_warmup_epochs
        P = self.lr_decay_factor
        lr_scheduler_lambda = lambda i: ((i+1) / N) if i < N else P ** (i + 1 - N)
        return TrainingLoop(
            self.name,
            model,
            optimizer,
            self.default(self.data_augmentations, []),
            self.default(self.performance_metrics, get_default_performance_metrics()),
            self.default(self.callbacks, []),
            lr_scheduler_lambda,
            max_epochs,
        )


def get_anchor_priors(num_anchors: int):
    return torch.nn.Parameter(torch.rand(num_anchors, 2, dtype=torch.float64))


