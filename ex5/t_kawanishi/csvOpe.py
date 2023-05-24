import csv

import numpy as np
import matplotlib.pyplot as plt

def plot_data(data:np.ndarray):
    """To plot data

    Args:
        data (np.ndarray): dataset for plot
    """
    # create graph
    fig = plt.figure()

    if len(data) == 2:
        # create graph and plot
        plt.plot(data[0], data[1], ".", label="dataset")
        plt.set_xlabel("X")
        plt.set_ylabel("f(X)")
        plt.legend()

    elif len(data) == 3:

        # create graph and plot
        ax = plt.subplot(projection="3d")
        ax.plot(data[0], data[1], data[2], ".", c="b", label="dataset")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("f(X)")
        ax.legend()

    else:
        raise ValueError(
            "data dimension should in 2 or 3 but " + str(len(data_array[0]))
        )

def read_csv(path:str) -> np.ndarray:
    """read out csv to matrix

    Args:
        path (str): the csv file path

    Returns:
        np.ndarray: data matrix
    """
    # read scv file and change type
    data_array = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        reader_data = next(reader)
        for line in reader:
            sub_group = []
            for value in line:
                sub_group.append(float(value))
            data_array.append(sub_group)

    # group by dimension
    data_array = np.array(data_array)
    data = data_array.T

    return data

if __name__ == "__main__":
    pass