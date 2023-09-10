import librosa
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import argparse


def get_feat(path_list):
    feature_mfcc = np.zeros((len(path_list), 13))
    feature_delta = np.zeros_like(feature_mfcc)
    feature_delta2 = np.zeros_like(feature_mfcc)
    for i, path in enumerate(path_list):
        # load wav file
        data, fs = librosa.load(path)
        # mfcc
        mfcc = librosa.feature.mfcc(y=data, n_mfcc=13)
        feature_mfcc[i] = np.mean(mfcc, axis=1)
        feature_delta[i] = np.mean(librosa.feature.delta(mfcc, width=7),
                                   axis=1)
        feature_delta2[i] = np.mean(librosa.feature.delta(mfcc, width=7,
                                                          order=2), axis=1)
    features = np.concatenate([feature_mfcc, feature_delta,
                               feature_delta2], axis=1)
    return features


def recog_nn(input_dim, output_dim):
    he_uni = tf.keras.initializers.HeUniform()
    input_layer = tf.keras.Input(shape=(input_dim,), name="input")
    hidden1 = tf.keras.layers.Dense(256, activation="relu", name="hidden1",
                                    kernel_initializer=he_uni,
                                    bias_initializer="zeros")(input_layer)
    hidden1 = tf.keras.layers.Dropout(rate=0.2, name="dropout1")(hidden1)
    hidden2 = tf.keras.layers.Dense(256, activation="relu", name="hidden2",
                                    kernel_initializer=he_uni,
                                    bias_initializer="zeros")(hidden1)
    hidden2 = tf.keras.layers.Dropout(rate=0.2, name="dropout2")(hidden2)
    out_layer = tf.keras.layers.Dense(output_dim, activation="softmax",
                                      name="output",
                                      kernel_initializer='zeros',
                                      bias_initializer="zeros")(hidden2)
    model = tf.keras.Model(inputs=input_layer, outputs=out_layer)
    return model


def save_result(path_list, labels, fname):
    with open(fname, "w") as f:
        f.write("path,label\n")
        assert len(path_list) == len(labels)
        for path, label in zip(path_list, labels):
            f.write("{path},{label}\n".format(path=path, label=label))


def main():
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", type=str,
        help="Name of result file",
        default="result.csv"
    )
    args = parser.parse_args()
    fname = args.o

    # load csv files
    training = pd.read_csv("training.csv")
    test = pd.read_csv("test.csv")
    # Extract features
    x_train = get_feat(training["path"].values)
    x_test = get_feat(test["path"].values)
    print(x_train.shape)
    print(x_test.shape)
    # set target labels
    y_train = training["label"]
    # split data
    X_train, X_val, Y_train, Y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=20)
    # create MLP model
    print(tf.config.list_physical_devices('GPU'))
    model = recog_nn(input_dim=X_train.shape[1], output_dim=10)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    # fit model
    early_stopping = EarlyStopping(monitor="loss", min_delta=0.001,
                                   patience=10)
    tensorboard_callback = TensorBoard(log_dir="./log", histogram_freq=1)
    model.fit(x_train, y_train, batch_size=32, epochs=500,
              validation_data=(X_val, Y_val),
              verbose=2, callbacks=[early_stopping, tensorboard_callback])
    # evaluate and predict
    score = model.evaluate(X_val, Y_val, verbose=0)
    print("Accuracy: ", score[1])
    predict = model.predict(x_test)
    predicted_label = np.argmax(predict, axis=1)
    # save model and label
    model.save("./keras_model/learned_model.h5")
    save_result(test["path"].values, predicted_label, fname)


if __name__ == "__main__":
    main()
