"""Main."""
import argparse
import pickle
import time

import numpy as np

import hmm


def parse_args():
    """Retrieve variables from the command prompt."""
    parser = argparse.ArgumentParser(description="Perform hmm")
    parser.add_argument(
        "--pickle_file",
        type=str,
        default="../data1.pickle",
        help="data pickle file",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="forward",
        help="choice of algorithm",
    )

    return parser.parse_args()


def main():
    """Calculate GMM main file."""
    args = parse_args()
    fname = args.pickle_file
    alg = args.algorithm
    data = pickle.load(open(fname, "rb"))
    # print(data["answer_models"])
    # print(data["output"])
    # print(data["models"]["B"])

    answer = np.array(data["answer_models"])
    outputs = np.array(data["output"])
    PI = np.array(data["models"]["PI"])
    A = np.array(data["models"]["A"])
    B = np.array(data["models"]["B"])

    if "forward" in alg:
        models_forward = hmm.forward(outputs, PI, A, B)
        time_f = time.perf_counter()
        print(alg + ":{}".format(time_f))
        correct_number = np.count_nonzero(models_forward == answer)
        acc_forward = correct_number / models_forward.shape[0] * 100
        conf_mat_forward = np.zeros([A.shape[0], A.shape[0]])
        hmm.plot_model(
            answer,
            models_forward,
            conf_mat_forward,
            alg,
            acc_forward
        )
    elif "viterbi" in alg:
        models_viterbi = hmm.viterbi(outputs, PI, A, B)
        time_v = time.perf_counter()
        print(alg + ":{}".format(time_v))
        correct_number = np.count_nonzero(models_viterbi == answer)
        acc_viterbi = correct_number / models_viterbi.shape[0] * 100
        conf_mat_forward = np.zeros([A.shape[0], A.shape[0]])
        hmm.plot_model(
            answer,
            models_viterbi,
            conf_mat_forward,
            alg,
            acc_viterbi
        )


if __name__ == "__main__":
    main()
