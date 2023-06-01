
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for cepstrum analysis."""

import argparse

import scipy.fftpack.realtransforms
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import librosa

import spec as s

class MelFilterBank:
    def __init__(self, fs, f0, degree, framesize, overlap):
        self.fs = fs
        self.f0 = f0
        self.degree = degree
        self.framesize = framesize
        self.overlap = overlap
        self.filterbank = None
        self.f_centers = None
        
    def calc_mo(self):
        return 1000 * 1 / np.log10(1000 / self.f0 + 1.0)
    
    def hz2mel(self, f):
        m0 = self.calc_mo()
        return m0 * np.log10(f / self.f0 + 1.0)
    
    def mel2hz(self, m):
        m0 = self.calc_mo()
        return self.f0 * (np.power(10, m / m0) - 1.0)
    
    def melfilterbank(self):
        fmax = self.fs // 2
        melmax = self.hz2mel(fmax)
        nummax = self.framesize // 2
        dmel = melmax / (self.degree + 1)
        mel_centers = np.arange(self.degree + 2) * dmel
        f_centers = self.mel2hz(mel_centers)
        bin = np.floor((self.framesize) * f_centers / self.fs)
        filterbank = np.zeros((self.degree, nummax+1))
        for m in range(1, self.degree + 1):
            bank_start = int(bin[m - 1])
            bank_center = int(bin[m])
            bank_finish = int(bin[m + 1])
            for k in range(bank_start, bank_center):
                filterbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(bank_center, bank_finish):
                filterbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        self.filterbank, self.f_centers = filterbank, f_centers
        return filterbank, f_centers
    
    def calc_mfcc(self, data, ncepstrum):
        N = len(data)
        step = int(self.framesize * (1 - self.overlap))
        # Calculate the number of times to do windowing
        split_time = int(N / step)
        window = np.hamming(self.framesize)
        filterbank, f_centers = self.melfilterbank()
        print(f_centers)
        mfcc = np.zeros((split_time, ncepstrum))
        mel_spectrum = np.zeros((split_time))
        for t in range(split_time):
            if t * step + self.framesize > N:
                data = np.append(data, np.zeros(t * step + self.framesize - N))
            frame = data[t * step : t * step + self.framesize] * window
            fft_frame = np.abs(np.fft.rfft(frame))
            print(20 * np.log10(fft_frame))
            comp_fbank = np.dot(fft_frame, filterbank.T)
            # 0taisaku
            comp_fbank = np.where(comp_fbank == 0, np.finfo(float).eps, comp_fbank)
            mel_spectrum = 20 * np.log10(comp_fbank)
            cepstrum = scipy.fftpack.dct(mel_spectrum, type=2, norm="ortho")
            mfcc[t, :] = cepstrum[1:ncepstrum+1]
            if t == 0:
                fig3 = plt.figure(figsize=(15, 10))
                freq = np.fft.rfftfreq(self.framesize, d=1 / self.fs)
                # k-means clustering　for two-dimensional data
                ax3 = fig3.add_subplot(111)
                ax3.plot(freq, 20 * np.log10(fft_frame), label="Spectrum")
                ax3.plot(f_centers[1:-1], mel_spectrum, c="orange", label="Mel Spectrum")
                ax3.scatter(f_centers[1:-1], mel_spectrum, c="orange")
                ax3.set_title("Original Spectrum and Mel Spectrum")
                ax3.set_xlabel("Frequency [Hz]")
                ax3.set_ylabel("Magnitude [dB]")
                plt.show()
        return mfcc
        
    def delta(self, mfcc):
        delta_mfcc = np.zeros(len(mfcc))
        for i in range(len(mfcc) - 1):
            delta_mfcc[i] = mfcc[i+1] - mfcc[i]
        return delta_mfcc

def main():
    """Estimate f0 frequency."""
    parser = argparse.ArgumentParser(description="This program estimates f0 frequency.")
    parser.add_argument(
        "-f", "--framesize", help="the size of window", default=512, type=int
    )
    parser.add_argument(
        "-o", "--overlap", help="the rate of overlap", default=0.8, type=float
    )
    parser.add_argument(
        "-t", "--tap", help="the number of taps of lifter", default=51, type=int
    )
    parser.add_argument(
        "-n", "--ncepstrum", help="upper bound on the order of the lower order cepstrum to be cut out to find the MFCC", default=12, type=int
    )
    parser.add_argument(
        "-d", "--degree", help="the number of degrees to compress the frequency domain", default=20, type=int
    )
    parser.add_argument(
        "-fo", help="frequency parameter", default=700, type=int
    )
    
    parser.add_argument("path", help="the path to the audio file")
    args = parser.parse_args()

    sound_file = args.path
    framesize = args.framesize
    overlap = args.overlap
    ncepstrum = args.ncepstrum
    degree = args.degree
    f0 = args.fo
    # Get waveform and sampling rate from the audio file
    data, samplerate = sf.read(sound_file)
    
    MFB = MelFilterBank(samplerate, f0, degree, framesize, overlap)
    mfcc = MFB.calc_mfcc(data, ncepstrum)
    filterbank, f_centers = MFB.filterbank, MFB.f_centers
    
    cmap_keyword = "jet"
    cmap = plt.get_cmap(cmap_keyword)
    
    fig0 = plt.figure(figsize=(15, 10))
    # k-means clustering　for two-dimensional data
    ax0 = fig0.add_subplot(111)
    np.set_printoptions(threshold=np.inf)
    #print(filterbank)
    f = np.arange(0, samplerate // 2, (samplerate // 2) / (framesize // 2 + 1))
    for i in range(1, filterbank.shape[0]):
        ax0.plot(f, filterbank[i, :], color=cmap((i+1)/filterbank.shape[0]))
    ax0.set_title("Mel Filter Bank")
    ax0.set_xlabel("Frequency [Hz]")
    ax0.set_ylabel("Magnitude")
    #delt = MFB.delta(mfcc)
    #deltdelt = MFB.delta(delt)
    
    fig1 = plt.figure(figsize = (15, 10))
    ax1_1 = fig1.add_subplot(2, 1, 1)
    s.draw_spectrogram(
        data,
        ax=ax1_1,
        framesize=framesize,
        y_limit=samplerate // 2,
        time=len(data) / samplerate,
        overlap=overlap,
        samplerate=samplerate,
    )
    ax1_1.set_title('Original Signal')

    ax1_2 = fig1.add_subplot(2, 1, 2)
    s.draw_spectrogram(
        mfcc,
        ax=ax1_2,
        framesize=framesize,
        y_limit=samplerate // 2,
        time=len(data) / samplerate,
        overlap=overlap,
        samplerate=samplerate,
        is_spec=True
    )
    ax1_2.set_title('MFCC')
    fig1.savefig("mfcc.png")
    
    
    #fig2 = plt.figure(figsize = (15, 10))
    #ax2_1 = fig2.add_subplot(2, 1, 1)
    #librosa.display.specshow(delt.T)
    #ax2_1.colorbar()
    #ax2_1.xlabel('time[s]')
    #ax2_1.ylabel('$\delta$MFCC')

    #ax2_2 = fig2.add_subplot(2, 1, 1)
    #librosa.display.specshow(deltdelt.T)
    #ax2_2.colorbar()
    #ax2_2.xlabel('time[s]')
    #ax2_2.ylabel('$\delta\delta$MFCC')
    
    #fig2.savefig("delta.png")
    
 

    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
