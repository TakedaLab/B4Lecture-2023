"""This is a program about k-means and MFCC"""
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import numpy as py

import csvOpe as cO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a program to adapt k-means and MFCC to dataset.")
    parser.add_argument("path",help="The path of dataset.")
    parser.add_argument("-g","--group",help="The number of groups to classify")

    # read out parser
    args = parser.parse_args()

    # read out csv
    data = cO.read_csv(args.path)

    # create save image name
    i_name = re.sub(r".+\\","",args.path)
    i_name = re.sub(r"\..+","",i_name)

    #plot data
    cO.plot_data(data)

    #show graph
    plt.show()


