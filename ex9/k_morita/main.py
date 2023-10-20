import os
import pandas as pd
import numpy as np

import scipy.io.wavfile as wav

from keras.utils.np_utils import to_categorical
# from tensorflow.keras.utils import load_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from util import files_to_spectrogram



def my_MLP_1(input_dim, output_dim): # CNN MLP

    model = Sequential(name="CNN")

    model.add(Conv2D(48, kernel_size=(2,2), activation="relu", input_shape=input_dim))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(2,2), activation="relu"))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=(2,2), activation="relu"))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(output_dim, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="Adadelta", metrics=["accuracy"])

    return model


def plot_history(history, dirname):

    base_dir = "./result"
    result_dir = os.path.join(base_dir, dirname)
    os.makedirs(result_dir, exist_ok=True)

    # 学習過程をグラフで出力
    # print(history.history.keys())
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc, label="train")
    plt.plot(epochs, val_acc, label="validation")
    plt.grid()
    plt.title("Model accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(result_dir, "acc.png"))
    # plt.show()

    plt.figure()
    plt.plot(epochs, loss, label="train")
    plt.plot(epochs, val_loss, label="validation")
    plt.grid()
    plt.title("Model loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(result_dir, "loss.png"))
    # plt.show()



def main():

    # 前処理
    train_csv = pd.read_csv("../training.csv")
    test_csv = pd.read_csv("../test_truth.csv") 
    files_to_spectrogram(train_csv["path"].values, save_dir="spectrograms/train")
    files_to_spectrogram(test_csv["path"].values, save_dir="spectrograms/test")

    return 
    train_spec_csv = pd.read_csv("train.csv")
    files = train_spec_csv["path"].values
    labels = train_spec_csv["label"].values

    X_train, X_val, y_train, y_val = train_test_split(files, labels, test_size=0.2)

    X_train = [img_to_array(load_img(f)) for f in X_train]
    X_val = [img_to_array(load_img(f)) for f in X_val]
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    X_train = np.asanyarray(X_train)
    X_val = np.asanyarray(X_val)

    X_train /= 255
    X_val /= 255

    input_dim = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    output_dim = 10  # dim(y_train)

    model = my_MLP_1(input_dim, output_dim)
    model.summary()

    history = model.fit(X_train, y_train, batch_size=50, validation_data=(X_val, y_val), epochs=200, verbose=2)

    # print(history.history.keys())

    # Result Plot
    plot_history(history, dirname=model.name)

    # Evaluation
    files = test_csv["path"].values
    labels = test_csv["label"].values
    X_test = [img_to_array(load_img(f)) for f in files]
    y_test = labels
    # y_test = to_categorical(labels)
    X_test = np.asanyarray(X_test)
    X_test /= 255

    pred = np.argmax(model.predict(X_test), axis=1)
    print(pred)
    acc = accuracy_score(pred, y_test)
    print(acc)
    


if __name__ == "__main__":
    main()