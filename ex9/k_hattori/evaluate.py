import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy import stats


def cm_plot(predicted_label, truth_label, title=None, cmap=plt.cm.Reds):
    cm = confusion_matrix(predicted_label, truth_label)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.ylabel("Predicted")
    plt.xlabel("Ground truth")
    plt.tight_layout()


def print_accinterval(acc, size, alpha):
    z = stats.norm.interval(alpha)[1]
    Z = z * np.sqrt(acc * (1 - acc) / size)
    print("{}%信頼区間: {:.3f} < acc < {:.3f}".format(alpha*100, acc - Z, acc + Z))


def main():
    pred_path = "result.csv"
    truth_path = "answer.csv"
    pred = pd.read_csv(pred_path)
    truth = pd.read_csv(truth_path)
    cm_plot(pred["label"], truth["label"])
    plt.show()
    acc = accuracy_score(truth["label"], pred["label"])
    print("Accuracy: ", acc)
    print_accinterval(acc, len(pred), 0.95)


if __name__ == "__main__":
    main()
