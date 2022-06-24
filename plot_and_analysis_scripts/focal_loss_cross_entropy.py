import matplotlib.pyplot as plt
import numpy as np


PLOT_FOCAL_LOSS_FROM_PROB = True
PLOT_FOCAL_LOSS_FROM_LOGITS = True
FOCAL_LOSS_GAMMAS = [
    (0, "γ=0 (Standard CE-loss)"),
    (1, "γ=1"),
    (2, "γ=2"),
    (3, "γ=3"),
    (4, "γ=4"),
    (5, "γ=5"),
]

# TODO: Currently working on a per-anchor focal loss solution to the low/high confidence positive problem
# TODO: Notes on discussion in the book
# TODO: Negatives have one gamma
# TODO: High confidence positives have another gamma
# TODO: Low confidence positives have yet another gamma


if PLOT_FOCAL_LOSS_FROM_PROB:
    # Compute sigmoid and "standard" cross entropy
    logits = np.linspace(start=-4, stop=4, num=200)
    prob = 1 / (1 + np.exp(-logits))
    cross_entropy = -np.log(prob)
    plt.grid(True)
    # plt.plot(logits, prob)

    for gamma, caption in FOCAL_LOSS_GAMMAS:
        focal_loss = cross_entropy * (1 - prob) ** gamma
        plt.plot(prob, focal_loss)

    plt.legend(labels=[x[1] for x in FOCAL_LOSS_GAMMAS])
    plt.show()

if PLOT_FOCAL_LOSS_FROM_PROB:
    # Compute sigmoid and "standard" cross entropy
    logits = np.linspace(start=-3, stop=3, num=200)
    prob = 1 / (1 + np.exp(-logits))
    cross_entropy = -np.log(prob)
    plt.grid(True)
    # plt.plot(logits, prob)

    for gamma, caption in FOCAL_LOSS_GAMMAS:
        focal_loss = cross_entropy * (1 - prob) ** gamma
        plt.plot(logits, focal_loss)

    plt.legend(labels=[x[1] for x in FOCAL_LOSS_GAMMAS])
    plt.show()