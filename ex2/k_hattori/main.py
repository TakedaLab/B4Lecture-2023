import numpy as np
import matplotlib.pyplot as plt
import myfunc

def convolution(x, h):
    y = np.zeros(len(x) + len(h) - 1)
    width = len(h)
    for i in range(len(x)):
        y[i:i+width] += x[i] * h
    return y

def filter_window(length, cut_off, samplerate):

def main():
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    h = np.array([1/4, 1/4, 1/4, 1/4])

    y = convolution(x, h)

    print(y)



if __name__ == '__main__':
    main()
    exit(1)
