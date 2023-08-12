import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf


x, fs = sf.read('../dataset/train/george_0.wav')
mfccs = librosa.feature.mfcc(y=x, sr=fs)
fig, ax = plt.subplots(ncols=1)
img = librosa.display.specshow(mfccs, sr=fs, x_axis='time')
ax.set(title='MFCC')
ax.label_outer()
fig.colorbar(img, ax=ax)
plt.show()
