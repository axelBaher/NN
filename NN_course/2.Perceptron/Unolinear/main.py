import warnings
import os
import numpy as np
import matplotlib.pyplot as plt

N = 5
b = 3


def os_path():
    warnings.filterwarnings("ignore")
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cur_dir)
    print(f"\nCurrent directory:\n{os.getcwd()}\n")


def perceptron():
    x1 = np.random.random(N)
    x2 = x1 + [np.random.randint(10) / 10 for _ in range(N)] + b
    c1 = [x1, x2]

    x1 = np.random.random(N)
    x2 = x1 - [np.random.randint(10) / 10 for _ in range(N)] - 0.1 + b
    c2 = [x1, x2]

    f = [0 + b, 1 + b]

    w2 = 0.5
    w3 = -b * w2
    w = np.array([-w2, w2, w3])

    for n in range(2):
        if n == 0:
            c = c1
        else:
            c = c2
        for i in range(N):
            x = np.array([c[0][i], c[1][i], 1])
            y = np.dot(w, x)
            if y >= 0:
                print("C1")
            else:
                print("C2")
        if n == 0:
            print("\n")

    plt.scatter(c1[0], c1[1], s=10, c="red")
    plt.scatter(c2[0], c2[1], s=10, c="blue")
    plt.plot(f)
    plt.grid(True)
    plt.show()


def main():
    os_path()

    perceptron()
    pass


if __name__ == "__main__":
    main()
