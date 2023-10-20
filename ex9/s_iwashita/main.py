#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def my_MLP(input_shape, output_dim):
    """
    MLPモデルの構築
    Args:
        input_shape: 入力の形
        output_dim: 出力次元
    Returns:
        model: 定義済みモデル
    """

    model = Sequential()

    model.add(Dense(256, input_dim=input_shape))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim))
    model.add(Activation("softmax"))

    # モデル構成の表示
    model.summary()

    return model


def compress_sound(x, rate=0.9):
    input_len = len(x)
    x = librosa.effects.time_stretch(x, rate)
    return x[:input_len]


def stretch_sound(x, rate=1.1):
    input_len = len(x)
    x = librosa.effects.time_stretch(x, rate)
    pad = input_len - len(x)
    return np.concatenate((x, np.zeros(pad, dtype=np.float32)))


def add_white_noise(x):
    wn = np.random.randn(len(x))
    return x + 0.005 * wn


def pitch_shift_up(x):
    return librosa.effects.pitch_shift(x, 22050, +1)


def pitch_shift_down(x):
    return librosa.effects.pitch_shift(x, 22050, -1)


def feature_extraction(path_list):
    """
    wavファイルのリストから特徴抽出を行い，リストで返す
    扱う特徴量はMFCC13次元の平均（0次は含めない）
    Args:
        path_list: 特徴抽出するファイルのパスリスト
    Returns:
        features: 特徴量
    """

    load_data = lambda path: librosa.load(path)[0]

    data = list(map(load_data, path_list))
    data_conpress = list(map(compress_sound, data))
    data_stretch = list(map(stretch_sound, data))
    data_add_white_noise = list(map(add_white_noise, data))
    data_pitch_shift_up = list(map(pitch_shift_up, data))
    data_pitch_shift_down = list(map(pitch_shift_down, data))

    features = np.array(
        [np.mean(librosa.feature.mfcc(y=y, n_mfcc=13), axis=1) for y in data]
    )
    features_conpress = np.array(
        [np.mean(librosa.feature.mfcc(y=y, n_mfcc=13), axis=1) for y in data_conpress]
    )
    features_stretch = np.array(
        [np.mean(librosa.feature.mfcc(y=y, n_mfcc=13), axis=1) for y in data_stretch]
    )
    features_add_white_noise = np.array(
        [
            np.mean(librosa.feature.mfcc(y=y, n_mfcc=13), axis=1)
            for y in data_add_white_noise
        ]
    )
    features_pitch_shift_up = np.array(
        [
            np.mean(librosa.feature.mfcc(y=y, n_mfcc=13), axis=1)
            for y in data_pitch_shift_up
        ]
    )
    features_pitch_shift_down = np.array(
        [
            np.mean(librosa.feature.mfcc(y=y, n_mfcc=13), axis=1)
            for y in data_pitch_shift_down
        ]
    )

    features = np.concatenate(
        (
            features,
            features_conpress,
            features_stretch,
            features_add_white_noise,
            features_pitch_shift_up,
            features_pitch_shift_down,
        )
    )

    return features


def feature_extraction_test(path_list):
    """
    wavファイルのリストから特徴抽出を行い，リストで返す
    扱う特徴量はMFCC13次元の平均（0次は含めない）
    Args:
        path_list: 特徴抽出するファイルのパスリスト
    Returns:
        features: 特徴量
    """

    load_data = lambda path: librosa.load(path)[0]

    data = list(map(load_data, path_list))
    features = np.array(
        [np.mean(librosa.feature.mfcc(y=y, n_mfcc=13), axis=1) for y in data]
    )

    return features


def show_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title("loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[0].set_title("categorical_accuracy")
    ax[0].plot(history.epoch, history.history["val_acc"], label="Validation accuracy")
    ax[0].legend()
    ax[1].legend()
    plt.savefig("result/history.png")


def plot_confusion_matrix(predict, ground_truth, title=None, cmap=plt.cm.Blues):
    """
    予測結果の混合行列をプロット
    Args:
        predict: 予測結果
        ground_truth: 正解ラベル
        title: グラフタイトル
        cmap: 混合行列の色
    Returns:
        Nothing
    """

    cm = confusion_matrix(predict, ground_truth)
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel("Predicted")
    plt.xlabel("Ground truth")
    plt.show()
    plt.savefig("s_iwashita/result.png")


def write_result(paths, outputs):
    """
    結果をcsvファイルで保存する
    Args:
        paths: テストする音声ファイルリスト
        outputs:
    Returns:
        Nothing
    """

    with open("s_iwashita/result.csv", "w") as f:
        f.write("path,output\n")
        assert len(paths) == len(outputs)
        for path, output in zip(paths, outputs):
            f.write("{path},{output}\n".format(path=path, output=output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_truth", type=str, help="テストデータの正解ファイルCSVのパス")
    args = parser.parse_args()

    # データの読み込み
    training = pd.read_csv("training.csv")
    test = pd.read_csv("test.csv")

    # 学習データの特徴抽出
    X_train = feature_extraction(training["path"].values)
    X_test = feature_extraction_test(test["path"].values)

    # 正解ラベルをone-hotベクトルに変換 ex. 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    Y_train = np_utils.to_categorical(y=training["label"], num_classes=10)
    Y_train = np.concatenate((Y_train, Y_train, Y_train, Y_train, Y_train, Y_train))

    # 学習データを学習データとバリデーションデータに分割 (バリデーションセットを20%とした例)
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train,
        Y_train,
        test_size=0.2,
        random_state=20200616,
    )
    print(X_train.shape[1:])

    # モデルの構築
    model = my_MLP(input_shape=X_train.shape[1], output_dim=10)

    # モデルの学習基準の設定
    model.compile(
        loss="categorical_crossentropy", optimizer=SGD(lr=0.002), metrics=["accuracy"]
    )

    # モデルの学習
    history = model.fit(X_train, Y_train, batch_size=32, epochs=100, verbose=1)

    # show_history(history)

    # モデル構成，学習した重みの保存
    model.save("s_iwashita/keras_model/my_model.h5")

    # バリデーションセットによるモデルの評価
    # モデルをいろいろ試すときはテストデータを使ってしまうとリークになる可能性があるため、このバリデーションセットによる指標を用いてください
    score = model.evaluate(X_validation, Y_validation, verbose=0)
    print("Validation accuracy: ", score[1])

    # 予測結果
    predict = model.predict(X_test)
    predicted_values = np.argmax(predict, axis=1)

    # テストデータに対して推論した結果の保存
    write_result(test["path"].values, predicted_values)

    # テストデータに対する正解ファイルが指定されていれば評価を行う（accuracyと混同行列）
    if args.path_to_truth:
        test_truth = pd.read_csv(args.path_to_truth)
        truth_values = test_truth["label"].values
        plot_confusion_matrix(predicted_values, truth_values)
        print("Test accuracy: ", accuracy_score(truth_values, predicted_values))


if __name__ == "__main__":
    main()
