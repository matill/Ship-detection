from dataclasses import dataclass
import json
from types import LambdaType
from typing import Callable, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrainingLogEntry:
    epoch: int
    training_loss_sum: float
    performance_metrics: Union[Dict, None]
    training_loss_subterms: Dict
    avg_inference_time_seconds: float
    train_elapsed_seconds: float
    test_elapsed_seconds: float
    epoch_elapsed_seconds: float

    def get_performance_metric(self, metric_name) -> Union[None, float]:
        if self.performance_metrics is None:
            return None
        elif metric_name not in self.performance_metrics:
            return None
        else:
            return self.performance_metrics[metric_name]


PerformanceMetric = Tuple[
    Callable[[TrainingLogEntry], Optional[float]],
    str
]

def read_training_log(path: str) -> List[TrainingLogEntry]:
    """Given the name of an experiment, reads the training_log.json file and returns a list of TrainingLogEntry objects"""
    with open(path, "r") as F:
        training_log_json = json.load(F)
    return [TrainingLogEntry(**dictionary) for dictionary in training_log_json]


def plot_training_log_performance(
    log: List[TrainingLogEntry],
    performance_metrics: List[PerformanceMetric],
    ax
):
    # fig, ax = plt.subplots()

    # For each performance metric, get a list of epoch numbers and corresponding performance
    # metric value, and make a line plot
    for metric_function, metric_name in performance_metrics:
        epoch_performance_tuples = [
            (log_entry.epoch, metric_function(log_entry))
            for log_entry in log
            if metric_function(log_entry) is not None
        ]
        assert len(epoch_performance_tuples) > 0
        epochs = [epoch for (epoch, _performance) in epoch_performance_tuples]
        performances = [performance for (_epoch, performance) in epoch_performance_tuples]
        ax.plot(epochs, performances, label=metric_name)

    print("show")
    ax.legend()
    # plt.savefig(out_file)
    # plt.close("all")


def get_best_epoch(log: List[TrainingLogEntry], mapping) -> TrainingLogEntry:
    scores = np.array([mapping(log_entry) for log_entry in log])
    scores_argmax = np.argmax(scores)
    return log[scores_argmax]


def get_best_epochs(log: List[TrainingLogEntry], mappings) -> List[TrainingLogEntry]:
    return [get_best_epoch(log, mapping) for mapping in mappings]


