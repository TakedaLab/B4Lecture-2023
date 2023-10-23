import os
import sys

import pandas as pd
import numpy as np
from util import files_to_spectrogram

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score


def load_test_data():
    test_csv = pd.read_csv("./test.csv")
    files_to_spectrogram(test_csv["path"].values, save_dir="spectrograms/test")

    files = test_csv["path"].values
    labels = test_csv["label"].values

    X_test = [img_to_array(load_img(f)) for f in files]
    y_test = labels
    # y_test = to_categorical(labels)

    X_test = np.asanyarray(X_test)
    X_test /= 255

    return X_test, y_test


def plot_confusion_matrix(true, pred, labels, save="output.png"):
    cm = confusion_matrix(true, pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig(save)


def main():

    X_test, y_test = load_test_data()
    model = tf.saved_model.load("CNN")

    print("*** Evaluation ***")
    pred = np.argmax(model(X_test), axis=1)

    acc = accuracy_score(y_test, pred)
    recall = recall_score(y_test, pred, labels=range(10), average="macro")
    precision = precision_score(
        y_test, pred, labels=range(10), average="macro")

    print("Accuracy:", acc)
    print("Recall: ", recall)
    print("Precision:", precision)

    plot_confusion_matrix(y_test, pred, labels=range(
        10), save="./result/CNN/confusion_matrix.png")


if __name__ == "__main__":
    main()
