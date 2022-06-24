from dataclasses import dataclass
from logging import log
from typing import List, Optional
from training_log_interpretation import read_training_log, plot_training_log_performance, TrainingLogEntry, get_best_epoch
import os
import matplotlib.pyplot as plt

# OUT_FILE_NAME_F1_PDF = "./f1_comparison.pdf"
# OUT_FILE_NAME_F1_PNG = "./f1_comparison.png"

TITLE = "F2 scores"
DATA_FOLDER_NAME = "../out/experiments"
PLOT_FOLDER_NAME = "../out/plots"



@dataclass
class ExpName:
    name: str
    displayed_name: str

EXPERIMENT_NAMES = [
    ExpName("PartiallyOrderedMatching", "PartialOrdering"),
    ExpName("NaiveMultiAttributeMatching-EqualWeight", "NaiveOrdering(0.25)"),
    ExpName("NaiveMultiAttributeMatching-PositivityWeight=04", "NaiveOrdering(0.4)"),
    ExpName("NaiveMultiAttributeMatching-PositivityWeight=06", "NaiveOrdering(0.6)"),
    ExpName("PositivityAndPositionMatching", "PositionAndPositivity"),
    ExpName("GlobalMatching", "GlobalMatching"),
    ExpName("GlobalMatching+MultilayerAttentionCfg(64,2)", "GlobalMatching+Att"),
    ExpName("GlobalMatching+Attention(64,2)+SAT(0.1,0.01,0.1)", "GlobalMatching+Att+SAT"),
    ExpName("GlobalMatching+Attention(64,2)+SAT(0.1,0.01,0.1)+AuxiliaryLoss", "GlobalMatching+Att+SAT+Aux"),
]

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
        elif precision == 0.0 and recall == 0.0:
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



# USED_METRICS = {
    # "precision": (get_precision, "Precision"),
    # "recall": (get_recall, "Recall"),
    # "relaxed_recall": (get_relaxed_recall, "Relaxed recall"),
    # "relaxed_precision": (get_relaxed_precision, "Relaxed precision"),
    # "f2": (f2, "F2"),
# }


def save_fig(fig, name: str):
    fig.savefig(f"{PLOT_FOLDER_NAME}/{name}.pdf")
    fig.savefig(f"{PLOT_FOLDER_NAME}/{name}.png")


def plot_f1_and_f2_comparison(fname: str, title: str, experiment_names: List[ExpName]):
    fig, axs = plt.subplots(2, 1)
    # fig, axs = plt.subplots(4, 1)
    # ax = axs[0]
    axs[0].set_title("F1 scores")
    axs[0].set(ylabel="F1", xlabel="Epoch number")
    axs[1].set_title("F2 scores")
    axs[1].set(ylabel="F2", xlabel="Epoch number")
    # axs[2].set_title("Precision")
    # axs[2].set(ylabel="Precision", xlabel="Epoch number")
    # axs[3].set_title("Recall")
    # axs[3].set(ylabel="Recall", xlabel="Epoch number")
    for i, e in enumerate(experiment_names):
        log_path = f"{DATA_FOLDER_NAME}/{e.name}/training_log.json"

        # Read log
        training_log = read_training_log(log_path)

        # Plot f1 and f2 logs
        plot_training_log_performance(
            training_log,
            [(f1, e.displayed_name)],
            axs[0]
        )
        plot_training_log_performance(
            training_log,
            [(f2, e.displayed_name)],
            axs[1]
        )
        # plot_training_log_performance(
        #     training_log,
        #     [(get_precision, e.displayed_name)],
        #     axs[2]
        # )
        # plot_training_log_performance(
        #     training_log,
        #     [(get_recall, e.displayed_name)],
        #     axs[3]
        # )

    # fig.show()
    fig.tight_layout(pad=1.0)
    save_fig(fig, fname)
    plt.close("all")




def plot_precision_recall_comparison(fname: str, title: str, experiment_names: List[ExpName]):
    fig, axs = plt.subplots(2, 1)
    # fig, axs = plt.subplots(4, 1)
    # ax = axs[0]
    # axs[0].set_title("Precision")
    # axs[0].set(ylabel="F1", xlabel="Epoch number")
    # axs[1].set_title("F2 scores")
    # axs[1].set(ylabel="F2", xlabel="Epoch number")
    axs[0].set_title("Precision")
    axs[0].set(ylabel="Precision", xlabel="Epoch number")
    axs[1].set_title("Recall")
    axs[1].set(ylabel="Recall", xlabel="Epoch number")
    for i, e in enumerate(experiment_names):
        log_path = f"{DATA_FOLDER_NAME}/{e.name}/training_log.json"

        # Read log
        training_log = read_training_log(log_path)

        # Plot f1 and f2 logs
        plot_training_log_performance(
            training_log,
            [(get_precision, e.displayed_name)],
            axs[0]
        )
        plot_training_log_performance(
            training_log,
            [(get_recall, e.displayed_name)],
            axs[1]
        )
        # plot_training_log_performance(
        #     training_log,
        #     [(get_precision, e.displayed_name)],
        #     axs[2]
        # )
        # plot_training_log_performance(
        #     training_log,
        #     [(get_recall, e.displayed_name)],
        #     axs[3]
        # )

    # fig.show()
    fig.tight_layout(pad=1.0)
    save_fig(fig, fname)
    plt.close("all")


EXP_1 = [
    ExpName("PartiallyOrderedMatching", "PartialOrdering"),
    ExpName("NaiveMultiAttributeMatching-EqualWeight", "NaiveOrdering(0.25)"),
    ExpName("NaiveMultiAttributeMatching-PositivityWeight=04", "NaiveOrdering(0.4)"),
    ExpName("NaiveMultiAttributeMatching-PositivityWeight=06", "NaiveOrdering(0.6)"),
    ExpName("PositivityAndPositionMatching", "PositionAndPositivity"),
    # ExpName("GlobalMatching", "GlobalMatching"),
    # ExpName("GlobalMatching+MultilayerAttentionCfg(64,2)", "GlobalMatching+Att"),
    # ExpName("GlobalMatching+Attention(64,2)+SAT(0.1,0.01,0.1)", "GlobalMatching+Att+SAT"),
    # ExpName("GlobalMatching+Attention(64,2)+SAT(0.1,0.01,0.1)+AuxiliaryLoss", "GlobalMatching+Att+SAT+Aux"),
]
EXP_2 = [
    # ExpName("PartiallyOrderedMatching", "PartialOrdering"),
    # ExpName("NaiveMultiAttributeMatching-EqualWeight", "NaiveOrdering(0.25)"),
    # ExpName("NaiveMultiAttributeMatching-PositivityWeight=04", "NaiveOrdering(0.4)"),
    # ExpName("NaiveMultiAttributeMatching-PositivityWeight=06", "NaiveOrdering(0.6)"),
    ExpName("PositivityAndPositionMatching", "PositionAndPositivity"),
    ExpName("GlobalMatching", "GlobalMatching"),
    # ExpName("GlobalMatching+MultilayerAttentionCfg(64,2)", "GlobalMatching+Att"),
    # ExpName("GlobalMatching+Attention(64,2)+SAT(0.1,0.01,0.1)", "GlobalMatching+Att+SAT"),
    # ExpName("GlobalMatching+Attention(64,2)+SAT(0.1,0.01,0.1)+AuxiliaryLoss", "GlobalMatching+Att+SAT+Aux"),
]
EXP_3 = [
    # ExpName("PartiallyOrderedMatching", "PartialOrdering"),
    # ExpName("NaiveMultiAttributeMatching-EqualWeight", "NaiveOrdering(0.25)"),
    # ExpName("NaiveMultiAttributeMatching-PositivityWeight=04", "NaiveOrdering(0.4)"),
    # ExpName("NaiveMultiAttributeMatching-PositivityWeight=06", "NaiveOrdering(0.6)"),
    # ExpName("PositivityAndPositionMatching", "PositionAndPositivity"),
    ExpName("GlobalMatching", "GlobalMatching"),
    ExpName("GlobalMatching+MultilayerAttentionCfg(64,2)", "GlobalMatching+Att"),
    ExpName("GlobalMatching+Attention(64,2)+SAT(0.1,0.01,0.1)", "GlobalMatching+Att+SAT"),
    ExpName("GlobalMatching+Attention(64,2)+SAT(0.1,0.01,0.1)+AuxiliaryLoss", "GlobalMatching+Att+SAT+Aux"),
]
EXP_3_ATT = [
    # ExpName("PartiallyOrderedMatching", "PartialOrdering"),
    # ExpName("NaiveMultiAttributeMatching-EqualWeight", "NaiveOrdering(0.25)"),
    # ExpName("NaiveMultiAttributeMatching-PositivityWeight=04", "NaiveOrdering(0.4)"),
    # ExpName("NaiveMultiAttributeMatching-PositivityWeight=06", "NaiveOrdering(0.6)"),
    # ExpName("PositivityAndPositionMatching", "PositionAndPositivity"),
    ExpName("GlobalMatching", "GlobalMatching"),
    ExpName("GlobalMatching+MultilayerAttentionCfg(64,2)", "GlobalMatching+Att"),
    # ExpName("GlobalMatching+Attention(64,2)+SAT(0.1,0.01,0.1)", "GlobalMatching+Att+SAT"),
    # ExpName("GlobalMatching+Attention(64,2)+SAT(0.1,0.01,0.1)+AuxiliaryLoss", "GlobalMatching+Att+SAT+Aux"),
]

plot_f1_and_f2_comparison(
    "f1_f2_graph_exp1",
    "TITLE NOT USED",
    EXP_1,
)
plot_precision_recall_comparison(
    "precision_recall_graph_exp1",
    "TITLE NOT USED",
    EXP_1,
)

plot_f1_and_f2_comparison(
    "f1_f2_graph_exp2",
    "TITLE NOT USED",
    EXP_2,
)

plot_precision_recall_comparison(
    "precision_recall_graph_exp2",
    "TITLE NOT USED",
    EXP_2,
)

plot_precision_recall_comparison(
    "precision_recall_graph_exp3(freebies-and-specials)",
    "TITLE NOT USED",
    EXP_3,
)

plot_f1_and_f2_comparison(
    "f1_f2_graph_exp3(freebies-and-specials)",
    "TITLE NOT USED",
    EXP_3,
)

plot_precision_recall_comparison(
    "precision_recall_graph_exp3_att",
    "TITLE NOT USED",
    EXP_3_ATT,
)

# def plot_f1_comparison():
#     fig, ax = plt.subplots(1, 1)
#     # ax = axs[0]
#     ax.set_title(TITLE)
#     ax.set(ylabel="F1", xlabel="Epoch number")
#     for i, e in enumerate(EXPERIMENT_NAMES):
#         log_path = f"{DATA_FOLDER_NAME}/{e.name}/training_log.json"

#         # Read log
#         training_log = read_training_log(log_path)

#         # Plot log
#         plot_training_log_performance(
#             training_log,
#             [(f1, e.displayed_name)],
#             ax
#         )

#     # fig.show()
#     fig.tight_layout(pad=1.0)
#     fig.savefig(OUT_FILE_NAME_F1_PDF)
#     fig.savefig(OUT_FILE_NAME_F1_PNG)
#     plt.close("all")



# plot_f2_comparison()
# plot_f1_comparison()


# C_BATCHES = "../real_experiment_results/comparing_batch_sizes/logs/"
# C_WEIGHTS = "../real_experiment_results/comparing_neg_weights/logs/"
# table_experiments = [
#     {
#         "folder": C_BATCHES,
#         "name": "PointYOLOF-ResNet18-ShallowEncoder-top1-4_head-FocalLoss(gamma=3, neg-weight=0.3, pos-weight=1.0)-BatchSize16"
#     },
#     {
#         "folder": C_BATCHES,
#         "name": "PointYOLOF-ResNet18-DeepEncoder-top1-4_head-FocalLoss(gamma=3, neg-weight=0.3, pos-weight=1.0)"
#     },
#     {
#         "folder": C_BATCHES,
#         "name": "PointYOLOF-ResNet34-ShallowEncoder-top1-4_head-FocalLoss(gamma=3, neg-weight=0.3, pos-weight=1.0)"
#     },
# ]



# import numpy as np



# results = []
# for e in table_experiments:
#     folder = e["folder"]
#     name = e["name"]
#     path = folder + name + ".json"

#     print("\n")
#     print(e)
#     training_log = read_training_log(path)

#     epoch_elapsed_seconds = np.array([x.epoch_elapsed_seconds for x in training_log])
#     avg_inference_time_seconds = np.array([x.avg_inference_time_seconds for x in training_log])

#     median_epoch_elapsed_seconds = np.median(epoch_elapsed_seconds)
#     median_inference_time_seconds = np.median(avg_inference_time_seconds)
#     print("median_epoch_elapsed_seconds", median_epoch_elapsed_seconds)
#     print("median_inference_time_seconds", median_inference_time_seconds)

#     best_epoch = get_best_epoch(
#         training_log,
#         f2,
#     )
#     print(best_epoch.performance_metrics)


#     results.append({
#         "model": name,
#         "median_epoch_elapsed_seconds": float(median_epoch_elapsed_seconds),
#         "median_inference_time_seconds": float(median_inference_time_seconds),

#         "precision": best_epoch.performance_metrics["precision"],
#         "recall": best_epoch.performance_metrics["recall"],
#         "relaxed_precision": best_epoch.performance_metrics["relaxed_precision"],
#         "relaxed_recall": best_epoch.performance_metrics["relaxed_recall"],
#         "f2": f2(best_epoch),
#     })


# # ResNet18 + ShallowEncoder, only the first 66 epochs
# e = table_experiments[0]
# folder = e["folder"]
# name = e["name"]
# path = folder + name + ".json"

# print("\n")
# print(e)
# training_log = read_training_log(path)
# training_log = training_log[0:66]

# epoch_elapsed_seconds = np.array([x.epoch_elapsed_seconds for x in training_log])
# avg_inference_time_seconds = np.array([x.avg_inference_time_seconds for x in training_log])

# median_epoch_elapsed_seconds = np.median(epoch_elapsed_seconds)
# median_inference_time_seconds = np.median(avg_inference_time_seconds)

# best_epoch = get_best_epoch(
#     training_log,
#     f2,
# )
# results.append({
#     "model": name,
#     "median_epoch_elapsed_seconds": float(median_epoch_elapsed_seconds),
#     "median_inference_time_seconds": float(median_inference_time_seconds),

#     "precision": best_epoch.performance_metrics["precision"],
#     "recall": best_epoch.performance_metrics["recall"],
#     "relaxed_precision": best_epoch.performance_metrics["relaxed_precision"],
#     "relaxed_recall": best_epoch.performance_metrics["relaxed_recall"],
#     "f2": f2(best_epoch),
# })

# import json
# print(json.dumps(results, indent=2))


