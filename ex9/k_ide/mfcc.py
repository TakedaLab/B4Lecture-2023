import os
import numpy as np
import torch
import torchaudio
import torchmetrics
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

training = pd.read_csv(os.path.join(root, "training.csv"))
train_path = training["path"].values
train_lebel = training['label'].values
n_mfcc = 13
datasize = len(train_path)
features = torch.zeros(datasize, n_mfcc)
transform = torchaudio.transforms.MFCC(n_mfcc=13, melkwargs={'n_mels':64, 'n_fft':512})

for i, path in enumerate(train_path):
    data, fs = librosa.load(os.path.join(root, path))
    #features[i] = torch.mean(transform(data[0]), axis=1)
    x = data.flatten()
    #x = data.to('cpu').detach().numpy().copy()
    #print(data)
    #import pdb; pdb.set_trace()
    mfccs = librosa.feature.mfcc(y=x, sr=fs)
    fig, ax = plt.subplots(ncols=1)
    img = librosa.display.specshow(mfccs, sr=fs, x_axis='time')
    fig.colorbar(img, ax=ax)
    path = path.replace('dataset/train/', '')
    path = path.replace('dataset/text/', '')
    path = path.replace('.wav', '')
    plt.savefig(f'../dataset/mfcc_img/{path}.png')
    plt.close()
