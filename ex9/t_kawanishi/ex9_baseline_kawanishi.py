#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
B4輪講最終課題 パターン認識に挑戦してみよう
作成したスクリプト
特徴量；MFCCのをPCAで次元削減したもの
識別器；MLP
"""


from __future__ import division
from __future__ import print_function

import argparse

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from hmmlearn import hmm
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from keras.callbacks import LearningRateScheduler


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

    model.add(Dense(64))
    model.add(Activation("relu"))

    model.add(Dense(output_dim))
    model.add(Activation("softmax"))

    # モデル構成の表示
    model.summary()

    return model


def feature_extraction(path_list,desired_dimension,noise_strength):
    """
    wavファイルのリストから特徴抽出を行い,リストで返す
    扱う特徴量はMFCC13次元(0次は含めない)をPCAで次元削減したもの
    Args:
        path_list: 特徴抽出するファイルのパスリスト
        desired_dimension: ほしい特徴量の次元数
        noise_strength: ノイズの強さ
    Returns:
        mfcc_features_reduced: 特徴量
    """


    mfcc_features = [] # MFCC特徴量を格納するためのリスト
    max_length_of_all_audio_files = 0  # 最大の長さを格納する変数

    for audio_file in path_list:
        y, sr = librosa.load(audio_file, sr=None)  # 音声データの読み込み
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # MFCCの抽出

        # 音声データの長さを取得
        audio_length = len(mfcc[0])

        if audio_length > max_length_of_all_audio_files:
            max_length_of_all_audio_files = audio_length

    if noise_strength == 0:
        for audio_file in path_list:
            y, sr = librosa.load(audio_file, sr=None)  # 音声データの読み込み

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # MFCCの抽出

            # すべてのMFCC特徴量を同じ長さに整形
            mfcc = np.pad(mfcc,((0, 0), (0, max_length_of_all_audio_files - len(mfcc[0]))),
                          mode='constant')


            mfcc_features.append(mfcc)

    else:
        mfcc_features_noisy = []
        for audio_file in path_list:
            y, sr = librosa.load(audio_file, sr=None)  # 音声データの読み込み
            noisy_y = y + noise_strength * np.random.randn(len(y)) #ノイズ付きデータの生成

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # MFCCの抽出
            mfcc_noisy = librosa.feature.mfcc(y=noisy_y, sr=sr, n_mfcc=13)

            # すべてのMFCC特徴量を同じ長さに整形
            mfcc = np.pad(mfcc, ((0, 0), (0, max_length_of_all_audio_files - len(mfcc[0]))), mode='constant')
            mfcc_noisy = np.pad(mfcc_noisy, ((0, 0), (0, max_length_of_all_audio_files - len(mfcc_noisy[0]))), mode='constant')


            mfcc_features.append(mfcc)
            mfcc_features_noisy.append(mfcc_noisy)

        mfcc_features.extend(mfcc_features_noisy)

    pca = PCA(n_components=desired_dimension)
    mfcc_features_reduced = []

    for mfcc in mfcc_features:
        mfcc_reduced = pca.fit_transform(mfcc)
        mfcc_features_reduced.append(mfcc_reduced)

    return np.array(mfcc_features_reduced)


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
    plt.savefig("t_kawanishi/confusion_matrix.png")
    plt.show()


def write_result(paths, outputs):
    """
    結果をcsvファイルで保存する
    Args:
        paths: テストする音声ファイルリスト
        outputs:
    Returns:
        Nothing
    """

    with open("t_kawanishi/result.csv", "w") as f:
        f.write("path,output\n")
        assert len(paths) == len(outputs)
        for path, output in zip(paths, outputs):
            f.write("{path},{output}\n".format(path=path, output=output))


def plot_history(history):
    # 学習過程をグラフで出力
    # print(history.history.keys())
    acc = history.history["accuracy"]
    loss = history.history["loss"]
    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc)
    plt.grid()
    plt.title("Model accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("t_kawanishi/history_baseline_kawanishi_acc.png", transparent=True)
    plt.show()

    plt.figure()
    plt.plot(epochs, loss)
    plt.grid()
    plt.title("Model loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("t_kawanishi/history_baseline_kawanishi_loss.png", transparent=True)
    plt.show()


def lr_schedule(epoch):
    """学習率のスケジュール関数を定義

    Args:
        epoch: epochの回数

    Returns:
        learning_rate: 学習率
    """

    learning_rate = 0.001  # 初期学習率
    if epoch >= 75:
        learning_rate = 0.0001
    if epoch >= 150:
        learning_rate = 0.00001

    return learning_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_truth", type=str, help='テストデータの正解ファイルCSVのパス')
    args = parser.parse_args()

    # データの読み込み
    training = pd.read_csv("training.csv")
    test = pd.read_csv("test.csv")

    # 学習データの特徴抽出
    X_train = feature_extraction(training["path"].values,2,0.1)
    X_test = feature_extraction(test["path"].values,2,0)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    Y_train = np_utils.to_categorical(y=training["label"], num_classes=10)
    Y_train = np.concatenate((Y_train,Y_train),axis=0)

    # 学習データを学習データとバリデーションデータに分割 (バリデーションセットを20%とした例)
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train, Y_train,
        test_size=0.2,
        random_state=20200616,
    )

    # モデルの構築
    model1 = my_MLP(input_shape=X_train.shape[1], output_dim=10)

    # モデルの学習基準の設定
    model1.compile(loss="categorical_crossentropy",
                    optimizer=Adam(lr=0.001),
                    metrics=["accuracy"])

    lr_scheduler = LearningRateScheduler(lr_schedule)

    # モデルの学習
    history1 = model1.fit(X_train,
                        Y_train,
                        batch_size=32,
                        epochs=200,
                        verbose=1,
                        callbacks=[lr_scheduler])


    plot_history(history1)


    # モデル構成，学習した重みの保存
    model1.save("t_kawanishi/my_model.h5")

    # バリデーションセットによるモデルの評価
    # モデルをいろいろ試すときはテストデータを使ってしまうとリークになる可能性があるため、このバリデーションセットによる指標を用いてください
    score = model1.evaluate(X_validation, Y_validation, verbose=0)
    print("Validation accuracy: ", score[1])

    # 予測結果
    predict = model1.predict(X_test)
    predicted_values = np.argmax(predict, axis=1)

    # テストデータに対して推論した結果の保存
    write_result(test["path"].values, predicted_values)
    # テストデータに対する正解ファイルが指定されていれば評価を行う（accuracyと混同行列）
    if args.path_to_truth:
        test_truth = pd.read_csv(args.path_to_truth)
        truth_values = test_truth["label"].values
        test_accuracy = accuracy_score(truth_values, predicted_values)
        plot_confusion_matrix(
            predicted_values,
            truth_values,
            title=f"Acc. {round(test_accuracy*100,2)}%",
        )
        print("Test accuracy: ", test_accuracy)


if __name__ == "__main__":
    main()
