from logging import log
from typing import Optional
from training_log_interpretation import read_training_log, plot_training_log_performance, TrainingLogEntry, get_best_epoch
import os
import matplotlib.pyplot as plt

COMPARING_NEG_WEIGHTS_PLOT = "../real_experiment_results/comparing_neg_weights/plots/combined.pdf"
COMPARING_BATCH_SIZE_PLOT = "../real_experiment_results/comparing_batch_sizes/plots/combined_batch_size.pdf"
COMPARING_BACKBONE_PLOT = "../real_experiment_results/comparing_batch_sizes/plots/combined_backbone.pdf"

def get_recall(log_entry: TrainingLogEntry) -> Optional[float]:
    recall = log_entry.get_performance_metric("recall")
    return 0.0 if recall is None else recall

def get_precision(log_entry: TrainingLogEntry) -> Optional[float]:
    precision = log_entry.get_performance_metric("precision")
    return 0.0 if precision is None else precision

def get_relaxed_recall(log_entry: TrainingLogEntry) -> Optional[float]:
    relaxed_recall = log_entry.get_performance_metric("relaxed_recall")
    return 0.0 if relaxed_recall is None else relaxed_recall

def get_relaxed_precision(log_entry: TrainingLogEntry) -> Optional[float]:
    relaxed_precision = log_entry.get_performance_metric("relaxed_precision")
    return 0.0 if relaxed_precision is None else relaxed_precision



class FBetaBase:
    def __init__(self, beta):
        self.beta = beta
        self.beta_sqrd = beta * beta

    def __call__(self, log_entry) -> float:
        precision = log_entry.get_performance_metric(self.PRECISION)
        recall = log_entry.get_performance_metric(self.RECALL)
        if precision is None or recall is None:
            return 0.0
        else:
            return ((1 + self.beta_sqrd) * precision * recall) / (self.beta_sqrd * precision + recall)

class FBeta(FBetaBase):
    PRECISION = "precision"
    RECALL = "recall"

class RelaxedFBeta(FBetaBase):
    PRECISION = "relaxed_precision"
    RECALL = "relaxed_recall"

class FBetaSum:
    def __init__(self, beta, strict_weight):
        assert 0 < strict_weight < 1
        self.f_beta = FBeta(beta)
        self.relaxed_f_beta = RelaxedFBeta(beta)
        self.strict_weight = strict_weight
        self.relaxed_weight = 1 - strict_weight

    def __call__(self, log_entry) -> float:
        strict = self.f_beta(log_entry) * self.strict_weight
        relaxed = self.relaxed_f_beta(log_entry) * self.relaxed_weight
        return strict + relaxed



# F2: More weight on recall than precision
# F01: More weight on precision than recall
f2 = FBeta(2)
f1 = FBeta(1)
f05 = FBeta(0.5)

relaxed_f2 = RelaxedFBeta(2)
relaxed_f1 = RelaxedFBeta(1)
relaxed_f05 = RelaxedFBeta(0.5)

f2_prod = FBetaSum(2, 0.2)
f1_prod = FBetaSum(1, 0.2)
f05_prod = FBetaSum(0.5, 0.2)



USED_METRICS = {

    "precision": (get_precision, "Precision"),
    "recall": (get_recall, "Recall"),
    "relaxed_recall": (get_relaxed_recall, "Relaxed recall"),
    "relaxed_precision": (get_relaxed_precision, "Relaxed precision"),
    "f2": (f2, "F2"),
}





def plot_neg_weight_comparison():
    EXPERIMENTS = [
        {
            "path": "../real_experiment_results/comparing_neg_weights/logs/PointYOLOF-ResNet18-ShallowEncoder-top1-4_head-FocalLoss(gamma=3, neg-weight=0.3, pos-weight=1.0).json",
            "title": "Negative weight = 0.3"
        },
        {
            "path": "../real_experiment_results/comparing_neg_weights/logs/PointYOLOF-ResNet18-ShallowEncoder-top1-4_head-FocalLoss(gamma=3, neg-weight=0.1, pos-weight=1.0).json",
            "title": "Negative weight = 0.1"
        },
        {
            "path": "../real_experiment_results/comparing_neg_weights/logs/PointYOLOF-ResNet18-ShallowEncoder-top1-4_head-FocalLoss(gamma=3, neg-weight=0.01, pos-weight=1.0).json",
            "title": "Negative weight = 0.01"
        },
    ]

    # fig, axs = plt.subplots(1, 3)
    fig, axs = plt.subplots(3, sharex=True)
    # fig, axs = plt.subplots(2, 2)
    for i, e in enumerate(EXPERIMENTS):
        log_path = e["path"]
        title = e["title"]

        # Read log
        training_log = read_training_log(log_path)

        # Plot log
        # plot_path = os.path.join(PLOT_FOLDER, f"{name}.png")
        ax = axs[i]
        ax.set_title(title)
        xlabel = "Epoch number" if i == 2 else None
        ax.set(ylabel="Performance", xlabel=xlabel)
        plot_training_log_performance(
            training_log,
            [
                USED_METRICS["precision"],
                USED_METRICS["recall"],
            ],
            ax
        )

    # fig.show()
    fig.tight_layout(pad=1.0)
    fig.savefig(COMPARING_NEG_WEIGHTS_PLOT)
    plt.close("all")


def plot_batch_size_comparison():
    EXPERIMENTS = [
        {
            "path": "../real_experiment_results/comparing_batch_sizes/logs/PointYOLOF-ResNet18-ShallowEncoder-top1-4_head-FocalLoss(gamma=3, neg-weight=0.3, pos-weight=1.0)-BatchSize16.json",
            "title": "Batch size: 16"
        },
        {
            "path": "../real_experiment_results/comparing_batch_sizes/logs/PointYOLOF-ResNet18-ShallowEncoder-top1-4_head-FocalLoss(gamma=3, neg-weight=0.3, pos-weight=1.0)-BatchSize8.json",
            "title": "Batch size: 8"
        },
        {
            "path": "../real_experiment_results/comparing_neg_weights/logs/PointYOLOF-ResNet18-ShallowEncoder-top1-4_head-FocalLoss(gamma=3, neg-weight=0.3, pos-weight=1.0).json",
            "title": "Batch size: 4"
        },
    ]

    fig, axs = plt.subplots(3, sharex=True)
    for i, e in enumerate(EXPERIMENTS):
        log_path = e["path"]
        title = e["title"]

        # Read log
        training_log = read_training_log(log_path)

        # Plot log
        # plot_path = os.path.join(PLOT_FOLDER, f"{name}.png")
        ax = axs[i]
        ax.set_title(title)
        xlabel = "Epoch number" if i == 2 else None
        ax.set(ylabel="Performance", xlabel=xlabel)
        plot_training_log_performance(
            training_log,
            [
                USED_METRICS["precision"],
                USED_METRICS["recall"],
            ],
            ax
        )

        # Plot dotted recall and precision lines corresponding to best precision
        # produced by BS-16 configuration
        precision = 0.6805111821086262
        recall = 0.7123745819397993

        x = [0, 110]
        yp = [precision, precision]
        yr = [recall, recall]
        ax.plot(x, yp, linestyle="dashed", linewidth=0.7, color="black")
        ax.plot(x, yr, linestyle="dashed", linewidth=0.7, color="black")

    # fig.show()
    fig.tight_layout(pad=1.0)
    fig.savefig(COMPARING_BATCH_SIZE_PLOT)
    plt.close("all")


def plot_backbone_comparison():
    EXPERIMENTS = [
        {
            "path": "../real_experiment_results/comparing_batch_sizes/logs/PointYOLOF-ResNet18-ShallowEncoder-top1-4_head-FocalLoss(gamma=3, neg-weight=0.3, pos-weight=1.0)-BatchSize16.json",
            "title": "ResNet18 + ShallowEncoder"
        },
        {
            "path": "../real_experiment_results/comparing_batch_sizes/logs/PointYOLOF-ResNet34-ShallowEncoder-top1-4_head-FocalLoss(gamma=3, neg-weight=0.3, pos-weight=1.0).json",
            "title": "ResNet34 + ShallowEncoder"
        },
        {
            "path": "../real_experiment_results/comparing_batch_sizes/logs/PointYOLOF-ResNet18-DeepEncoder-top1-4_head-FocalLoss(gamma=3, neg-weight=0.3, pos-weight=1.0).json",
            "title": "ResNet18 + DeepEncoder"
        },
    ]

    fig, axs = plt.subplots(3, sharex=True)
    for i, e in enumerate(EXPERIMENTS):
        log_path = e["path"]
        title = e["title"]

        # Read log
        training_log = read_training_log(log_path)

        # Plot log
        # plot_path = os.path.join(PLOT_FOLDER, f"{name}.png")
        ax = axs[i]
        ax.set_title(title)
        xlabel = "Epoch number" if i == 2 else None
        ax.set(ylabel="Performance", xlabel=xlabel)
        plot_training_log_performance(
            training_log,
            [
                USED_METRICS["precision"],
                USED_METRICS["recall"],
            ],
            ax
        )

        # Plot dotted recall and precision lines corresponding to best precision
        # produced by BS-16 configuration
        precision = 0.6805111821086262
        recall = 0.7123745819397993

        x = [0, 110]
        yp = [precision, precision]
        yr = [recall, recall]
        ax.plot(x, yp, linestyle="dashed", linewidth=0.7, color="black")
        ax.plot(x, yr, linestyle="dashed", linewidth=0.7, color="black")

    # fig.show()
    fig.tight_layout(pad=1.0)
    fig.savefig(COMPARING_BACKBONE_PLOT)
    plt.close("all")



plot_neg_weight_comparison()
plot_batch_size_comparison()
plot_backbone_comparison()




C_BATCHES = "../real_experiment_results/comparing_batch_sizes/logs/"
C_WEIGHTS = "../real_experiment_results/comparing_neg_weights/logs/"
table_experiments = [
    {
        "folder": C_BATCHES,
        "name": "PointYOLOF-ResNet18-ShallowEncoder-top1-4_head-FocalLoss(gamma=3, neg-weight=0.3, pos-weight=1.0)-BatchSize16"
    },
    {
        "folder": C_BATCHES,
        "name": "PointYOLOF-ResNet18-DeepEncoder-top1-4_head-FocalLoss(gamma=3, neg-weight=0.3, pos-weight=1.0)"
    },
    {
        "folder": C_BATCHES,
        "name": "PointYOLOF-ResNet34-ShallowEncoder-top1-4_head-FocalLoss(gamma=3, neg-weight=0.3, pos-weight=1.0)"
    },
]

import numpy as np



results = []
for e in table_experiments:
    folder = e["folder"]
    name = e["name"]
    path = folder + name + ".json"

    print("\n")
    print(e)
    training_log = read_training_log(path)

    epoch_elapsed_seconds = np.array([x.epoch_elapsed_seconds for x in training_log])
    avg_inference_time_seconds = np.array([x.avg_inference_time_seconds for x in training_log])

    median_epoch_elapsed_seconds = np.median(epoch_elapsed_seconds)
    median_inference_time_seconds = np.median(avg_inference_time_seconds)
    print("median_epoch_elapsed_seconds", median_epoch_elapsed_seconds)
    print("median_inference_time_seconds", median_inference_time_seconds)

    best_epoch = get_best_epoch(
        training_log,
        f2,
    )
    print(best_epoch.performance_metrics)


    results.append({
        "model": name,
        "median_epoch_elapsed_seconds": float(median_epoch_elapsed_seconds),
        "median_inference_time_seconds": float(median_inference_time_seconds),

        "precision": best_epoch.performance_metrics["precision"],
        "recall": best_epoch.performance_metrics["recall"],
        "relaxed_precision": best_epoch.performance_metrics["relaxed_precision"],
        "relaxed_recall": best_epoch.performance_metrics["relaxed_recall"],
        "f2": f2(best_epoch),
    })


# ResNet18 + ShallowEncoder, only the first 66 epochs
e = table_experiments[0]
folder = e["folder"]
name = e["name"]
path = folder + name + ".json"

print("\n")
print(e)
training_log = read_training_log(path)
training_log = training_log[0:66]

epoch_elapsed_seconds = np.array([x.epoch_elapsed_seconds for x in training_log])
avg_inference_time_seconds = np.array([x.avg_inference_time_seconds for x in training_log])

median_epoch_elapsed_seconds = np.median(epoch_elapsed_seconds)
median_inference_time_seconds = np.median(avg_inference_time_seconds)

best_epoch = get_best_epoch(
    training_log,
    f2,
)
results.append({
    "model": name,
    "median_epoch_elapsed_seconds": float(median_epoch_elapsed_seconds),
    "median_inference_time_seconds": float(median_inference_time_seconds),

    "precision": best_epoch.performance_metrics["precision"],
    "recall": best_epoch.performance_metrics["recall"],
    "relaxed_precision": best_epoch.performance_metrics["relaxed_precision"],
    "relaxed_recall": best_epoch.performance_metrics["relaxed_recall"],
    "f2": f2(best_epoch),
})

import json
print(json.dumps(results, indent=2))






    ResNet18 + ShallowEncoder & 81.22 & 0.005152 & 0.57971 & 0.86956 & 0.6220 & 0.8795 & 0.790513 \\
    ResNet18 + DeepEncoder & 87.47 & 0.007635 & 0.51955 & 0.86622 & 0.5711 & 0.8729 & 0.764237 \\
    ResNet34 + ShallowEncoder & 92.47 & 0.006979 & 0.49485 & 0.88461 & 0.5787 & 0.8963 & 0.764229 \\






\begin{table}[h!]
\centering
\begin{tabular}{||p{2.5cm}|p{1.1cm} p{1.1cm} p{1.1cm} p{1.1cm} p{1.1cm} p{1.1cm} p{1.1cm}||} 
 \hline
 Model & Epoch \newline time (s) & Inference \newline time (s) & Precision & Relaxed \newline precision & Recall & Relaxed \newline recall & F2 \\ [0.5ex] 
 \hline\hline
    ResNet18 + \newline ShallowEncoder & 81.22 & 0.005152 & 0.57971 & 0.86956 & 0.6220 & 0.8795 & 0.790513 \\
    ResNet18 + \newline DeepEncoder & 87.47 & 0.007635 & 0.51955 & 0.86622 & 0.5711 & 0.8729 & 0.764237 \\
    ResNet34 + \newline ShallowEncoder & 92.47 & 0.006979 & 0.49485 & 0.88461 & 0.5787 & 0.8963 & 0.764229 \\
%  5 & 88 & 788 & 6344 \\ [1ex] 
 \hline
\end{tabular}
\caption{Table to test captions and labels.}
\label{table:performance}
\end{table}



    ResNet18 + ShallowEncoder 
    81.22
    0.005152
    0.57971
    0.86956
    0.6220
    0.8795
    0.790513


    ResNet18 + \newline DeepEncoder &
    87.47 &
    0.007635 &
    0.51955 &
    0.86622 &
    0.5711 &
    0.8729 &
    0.764237 &

    ResNet34 + \newline ShallowEncoder &
    92.47 &
    0.006979 &
    0.49485 &
    0.88461 &
    0.5787 &
    0.8963 &
    0.764229 &




\begin{table}[h!]
\centering
\begin{tabular}{||w{2.5cm}|w{2.5cm} w{2.5cm} w{2.5cm}||} 
 \hline
%  What & ResNet18 + \newline ShallowEncoder & ResNet18 + \newline DeepEncoder & ResNet34 + \newline ShallowEncoder\\ [0.5ex] 
 What & ResNet18 + \newline ShallowEncoder & ResNet18 + \newline DeepEncoder & ResNet34 + \newline ShallowEncoder \\ [0.5ex] 
 \hline\hline
 
    Epoch \newline time (s)     & 81.22       & 87.47       & 92.47 \\
    Inference \newline time (s) & 0.005152    & 0.007635    & 0.006979 \\
    Precision                   & 0.57971     & 0.51955     & 0.49485 \\
    Relaxed \newline precision  & 0.86956     & 0.86622     & 0.88461 \\
    Recall                      & 0.6220      & 0.5711      & 0.5787 \\
    Relaxed \newline recall     & 0.8795      & 0.8729      & 0.8963 \\
    F2                          & 0.790513    & 0.764237    & 0.764229 \\

 \hline
\end{tabular}
\caption{Table to test captions and labels.}
\label{table:performance}
\end{table}