import time
import tensorflow as tf
import warnings
import os
from keras.datasets import mnist
from keras.utils import to_categorical


def os_path():
    warnings.filterwarnings("ignore")
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cur_dir)
    print(f"\nCurrent directory:\n{os.getcwd()}\n")


def main():
    os_path()


    sp = time.perf_counter()

    ep = time.perf_counter()
    print(f"Processing time:\n{ep - sp} sec\n")

    pass


if __name__ == "__main__":
    main()
