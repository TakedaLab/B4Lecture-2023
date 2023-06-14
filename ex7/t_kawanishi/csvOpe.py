"""csv operation."""
import csv
import re

import numpy as np


def get_fname(path: str) -> str:
    """Get file name.

    Args:
        path (str): path to the dataset

    Returns:
        str: dataset file name
    """
    i_name = re.sub(r".+\\", "", path)
    i_name = re.sub(r"\..+", "", i_name)
    return i_name


def read_csv(path: str) -> np.ndarray:
    """Read out csv to matrix.

    Args:
        path (str): the csv file path

    Returns:
        np.ndarray: data matrix
    """
    # read scv file and change type
    data_array = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            sub_group = []
            for value in line:
                sub_group.append(float(value))
            data_array.append(sub_group)

    # group by dimension
    data_array = np.array(data_array)

    return data_array


if __name__ == "__main__":
    pass
