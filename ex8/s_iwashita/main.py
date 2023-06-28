"""Predict models using HMM."""
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


class HMM:
    def __init__(self, fname):
        """Define each data.

        Args:
            fname (str): File name for loading data.
        """
        self.data = self.load_pickle(fname)
        self.fname = fname
        self.output = np.array(self.data["output"])
        self.A = np.array(self.data["models"]["A"])
        self.PI = np.array(self.data["models"]["PI"])
        self.B = np.array(self.data["models"]["B"])
        self.answer = np.array(self.data["answer_models"])

    def load_pickle(self, fname):
        """Load data from a pickle file.

        Args:
            fname (str): File name for loading data.

        Returns:
            dict: Data from a pickle file.
        """
        data = pickle.load(open("../" + fname, "rb"))
        return data

    def forward_algorithm(self):
        """Run forward algorithm."""
        t_sta = time.perf_counter()
        output = self.output[:, :, np.newaxis]
        alpha = self.PI[:, :, np.newaxis] * self.B[:, :, output[:, 0]]

        for i in range(1, self.output.shape[1]):
            alpha = (
                np.sum(alpha * self.A[:, :, np.newaxis], axis=1).transpose(0, 2, 1)[
                    :, :, :, np.newaxis
                ]
                * self.B[:, :, output[:, i]]
            )

        prob = np.sum(alpha, axis=1)
        self.pre_forward = np.argmax(prob, axis=0).reshape(-1)
        t_end = time.perf_counter()
        self.t_forward = t_end - t_sta

    def viterbi_algorithm(self):
        """Run viterbi algorithm."""
        t_sta = time.perf_counter()
        output = self.output[:, :, np.newaxis]
        alpha = self.PI[:, :, np.newaxis] * self.B[:, :, output[:, 0]]

        for i in range(1, self.output.shape[1]):
            alpha = (
                np.max(alpha * self.A[:, :, np.newaxis], axis=1).transpose(0, 2, 1)[
                    :, :, :, np.newaxis
                ]
                * self.B[:, :, output[:, i]]
            )

        prob = np.max(alpha, axis=1)
        self.pre_viterbi = np.argmax(prob, axis=0).reshape(-1)
        t_end = time.perf_counter()
        self.t_viterbi = t_end - t_sta

    def display_confusion_matrix(self):
        """Display confusion matrix."""
        labels = list(set(self.answer))

        fig = plt.figure()
        plt.subplots_adjust(wspace=0.3)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        cm_forward = confusion_matrix(self.answer, self.pre_forward, labels=labels)
        cm_viterbi = confusion_matrix(self.answer, self.pre_viterbi, labels=labels)
        sns.heatmap(
            cm_forward, annot=True, cbar=False, square=True, cmap="Blues", ax=ax1
        )
        sns.heatmap(
            cm_viterbi, annot=True, cbar=False, square=True, cmap="Blues", ax=ax2
        )
        acc_forward = int(accuracy_score(self.answer, self.pre_forward) * 100)
        acc_viterbi = int(accuracy_score(self.answer, self.pre_viterbi) * 100)
        ax1.set_title("Forward algorithm\n(Acc. {}%)".format(acc_forward))
        ax2.set_title("Viterbi algorithm\n(Acc. {}%)".format(acc_viterbi))
        ax1.set(xlabel="Predicted model", ylabel="Actual model")
        ax2.set(xlabel="Predicted model", ylabel="Actual model")
        plt.savefig("result/{}_result".format(self.fname[:5]))

    def measure_time(self):
        """Measure algorithm runtime."""
        print("============================================================")
        print("Data                 : {}".format(self.fname[:5]))
        print("Forward algorithm    : {}    (second)".format(self.t_forward))
        print("Viterbi algorithm    : {}    (second)".format(self.t_viterbi))
        print("============================================================")


def main():
    """Predict models using two algorithms."""
    fname = sys.argv[1]
    hmm = HMM(fname)

    hmm.forward_algorithm()
    hmm.viterbi_algorithm()
    hmm.display_confusion_matrix()
    hmm.measure_time()


if __name__ == "__main__":
    main()
