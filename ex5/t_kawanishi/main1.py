"""This is a program about k-means and MFCC."""
import argparse
import re

import matplotlib.pyplot as plt

import csvOpe as cO
import kMeans as km

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This is a program to adapt k-means and MFCC to dataset."
    )
    parser.add_argument("path", help="The path of dataset.")
    parser.add_argument(
        "-g", "--group", default=2, type=int,
        help="The number of groups to classify"
    )
    parser.add_argument(
        "-p", "--plus", action="store_true",
        help="whether k-means or k-means++"
    )

    # read out parser
    args = parser.parse_args()

    # read out csv
    data = cO.read_csv(args.path)

    # k-means
    label, centroid = km.k_means(data, args.group, plus_algo=args.plus)

    km.init_lbg(data,3)

    # create save image name
    i_name = re.sub(r".+\\", "", args.path)
    i_name = re.sub(r"\..+", "", i_name)

    # plot data and save image
    cO.plot(data, centroid, label, args.group)
    plt.savefig(i_name + "_groups" + str(args.group))

    # show graph
    plt.show()
