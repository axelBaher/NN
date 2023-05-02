import warnings
import os
import numpy as np


def os_path():
    warnings.filterwarnings("ignore")
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cur_dir)
    print(f"\nCurrent directory:\n{os.getcwd()}\n")


def act(x):
    return 0 if x <= 0 else 1


def go(c):
    x = np.array([c[0], c[1], 1])
    w1 = [1, 1, -1.5]
    w2 = [1, 1, -0.5]
    w_hidden = np.array([w1, w2])
    w_out = np.array([-1, 1, -0.5])

    _sum = np.dot(w_hidden, x)
    out = [act(x) for x in _sum]
    out.append(1)
    out = np.array(out)

    _sum = np.dot(w_out, out)
    y = act(_sum)
    return y


def main():
    os_path()

    c1 = [(1, 0), (0, 1)]
    c2 = [(0, 0), (1, 1)]

    print(go(c1[0]), go(c1[1]))
    print(go(c2[0]), go(c2[1]))
    pass


if __name__ == "__main__":
    main()
