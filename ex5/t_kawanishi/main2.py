"""Mel-Frequency Cepstrum Coefficients."""
import argparse

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf







if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This is a program to adapt MFCC."
        )
    parser.add_argument("path",help="sound file path")

    #get args
    args = parser.parse_args()

    #read out sound
    s_data, fs = sf.read(args.path)

    #time
    t = np.arange(0,len(s_data)) / fs

    plt.figure()
    plt.plot(t,s_data)
    plt.show()