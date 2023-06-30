import os

import pandas as pd
import numpy as np

import scipy.io.wavfile as wav

from keras.utils.np_utils import to_categorical
from keras.utils import load_img
from keras.utils import img_to_array

from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import BatchNormalization

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def wavfile_to_spectrogram(
    audio_path,
    save_path,
    spectrogram_dimensions=(64, 64),
    noverlap=16,
    cmap="gray_r"
):
    if os.path.exists(save_path):
        return

    smaple_rate, samples = wav.read(audio_path)
    fig = plt.figure()
    fig.set_size_inches((
        spectrogram_dimensions[0]/fig.get_dpi(), 
        spectrogram_dimensions[1]/fig.get_dpi()
        ))
    ax = plt.Axes(fig, [0., 0., 1., 1.,])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.specgram(samples, cmap=cmap, Fs=2, noverlap=noverlap)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)


def files_to_spectrogram(
    files,
    save_dir,
    spectrogram_dimensions=(64, 64),
    noverlap=12,
    cmap="gray_r"
):
    for file_name in files:
        audio_path = os.path.join(os.path.pardir, file_name)
        spectrogram_path = os.path.join(save_dir, os.path.basename(file_name).replace(".wav", ".png"))
        wavfile_to_spectrogram(
            audio_path,
            spectrogram_path,
            spectrogram_dimensions=spectrogram_dimensions,
            noverlap=noverlap,
            cmap=cmap
        )


def my_MLP_1(input_dim, output_dim):

    model = Sequential(name="model_1")

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


def main():

    train_csv = pd.read_csv("train.csv")
    files = train_csv["path"].values
    labels = train_csv["label"].values

    X_train, X_test, y_train, y_test = train_test_split(files, labels, test_size=0.3)

    X_train = [img_to_array(load_img(f)) for f in X_train]
    X_test = [img_to_array(load_img(f)) for f in X_test]
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    X_train = np.asanyarray(X_train)
    X_test = np.asanyarray(X_test)

    X_train /= 255
    X_test /= 255

    input_dim = (X_train.shape[1], X_train.shaep[2], X_train.shape[3])
    output_dim = 10  # dim(y_train)

    model = my_MLP_1(input_dim, output_dim)
    # model.summary()

    model.fit(X_train, y_train, batch_size=50, validation_split=0.2, epochs=100, verbose=1)



if __name__ == "__main__":
    main()