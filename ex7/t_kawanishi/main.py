"""To adapt GMM."""
import argparse

import matplotlib.pyplot as plt
import numpy as np

import csvOpe
import GMM


def main():
    # create parser and argument
    parser = argparse.ArgumentParser(
        description="This is a program to adapt GMM"
        )
    parser.add_argument("path",help="the path to the dataset.")
    parser.add_argument("-g","--group",default=2,help="The number of clusters")

    # read out parser
    args = parser.parse_args()

    # get file name
    f_name = csvOpe.get_fname(args.path)

    # read out csv
    data = csvOpe.read_csv(args.path)

    # 1-dim
    #TODO create method to GMM for 2-dim
    if data.shape[1] == 1:
        pass

    # 2-dim
    #TODO create method to GMM for 2-dim
    elif data.shape[1] == 2:
        pass

if __name__ == "__main__":
    main()