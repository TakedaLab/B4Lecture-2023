import os
import sys

import pandas as pd
import numpy as np
from util import files_to_spectrogram

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from keras.models import load_model
from sklearn.metrics import plot_confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def load_test_data():
    test_csv = pd.read_csv("../test_truth.csv")
    files_to_spectrogram(test_csv["path"].values, save_dir="spectrograms/test")

    test_csv["path"] = test_csv.apply(
        lambda row: os.path.join("./spectrograms/test", os.path.basename(row["path"])).replace(".wav", ".png"), axis=1)

    files = test_csv["path"].values
    labels = test_csv["label"].values

    X_test = [img_to_array(load_img(f)) for f in files]
    y_test = labels
    # y_test = to_categorical(labels)

    X_test = np.asanyarray(X_test)
    X_test /= 255

    return X_test, y_test



def main():

    X_test, y_test = load_test_data()

    # model = load_model("CNN")
    model = tf.saved_model.load("CNN")
    # plot_confusion_matrix(model, X_test, y_test, cmap="Blues")
    # plt.savefig(f"./result/{model.name}/eval_cm.png")

    pred = np.argmax(model(X_test), axis=1)
    print(pred)
    print()
    acc = accuracy_score(pred, y_test)

    print("*** Evaluation ***")
    print("Accuracy:", acc)
    



if __name__ == "__main__":
    main()







