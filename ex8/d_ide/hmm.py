"""HMM."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def forward(outputs, PI, A, B):
    """Calculate forward algorithm.

    Args:
        outputs (ndarray): output sequence
        PI (ndarray): Initial probability
        A (ndarray): state transition probability matrix
        B (ndarray): Output probability

    Returns:
        ndarray: model
    """
    size, t = outputs.shape
    models = np.zeros(size)

    for i in range(size):
        alpha = PI[:, :, 0] * B[:, :, outputs[i, 0]]

        for j in range(1, t):
            alpha = B[:, :, outputs[i, j]] * np.sum(A.T * alpha.T, axis=1).T

        models[i] = np.argmax(np.sum(alpha, axis=1))

    return models


def viterbi(outputs, PI, A, B):
    """Calculate viterbi algorithm.

    Args:
        outputs (ndarray): output sequence
        PI (ndarray): Initial probability
        A (ndarray): state transition probability matrix
        B (ndarray): Output probability

    Returns:
        ndarray: model
    """
    size, t = outputs.shape
    models = np.zeros(size)

    for i in range(size):
        alpha = PI[:, :, 0] * B[:, :, outputs[i, 0]]

        for j in range(1, t):
            alpha = B[:, :, outputs[i, j]] * np.max(A.T * alpha.T, axis=1).T

        models[i] = np.argmax(np.max(alpha, axis=1))

    return models


def plot_model(answer, models, conf_mat_model, title, acc):
    """Plot model.

    Args:
        answer (ndarray): correct label
        models (ndarray): each model
        conf_mat_model (ndarray): each model mat
        title (str): graph title
        acc (int): accuracy
    """
    for i in range(answer.shape[0]):
        conf_mat_model[int(models[i]), answer[i]] += 1

    sns.heatmap(conf_mat_model, cmap="PuRd", annot=True, cbar=True)
    plt.title(title + "\n (Acc. {}%)".format(acc))
    plt.ylabel("Actual Model")
    plt.xlabel("Predict Model")
    plt.savefig("result/")
    plt.show()
