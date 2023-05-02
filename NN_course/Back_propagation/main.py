import warnings
import os
import numpy as np
import matplotlib.pyplot as plt


# weights init for 1-st and 2-nd layers
W1 = np.array([[-0.2, 0.3, -0.4],
               [0.1, -0.3, -0.4]])
W2 = np.array([0.2, 0.3])
# first 3 vector elements - input,
# last vector element - expected output
EPOCH = [(-1, -1, -1, -1),
         (-1, -1, 1, 1),
         (-1, 1, -1, -1),
         (-1, 1, 1, 1),
         (1, -1, -1, -1),
         (1, -1, 1, 1),
         (1, 1, -1, -1),
         (1, 1, 1, -1)]


def os_path():
    warnings.filterwarnings("ignore")
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cur_dir)
    print(f"\nCurrent directory:\n{os.getcwd()}\n")


def activation_function(x):
    # hyperbolic tangents function
    y = 2 / (1 + np.exp(-x)) - 1
    return y


def derivative(x):
    y = 0.5 * (1 + x) * (1 - x)
    return y


def nn_forward(x):
    # input to hidden layer
    sum_hidden = np.dot(W1, x)
    output_hidden = np.array([activation_function(elem) for elem in sum_hidden])

    # hidden layer to output
    sum_y = np.dot(W2, output_hidden)
    y = activation_function(sum_y)

    return y, output_hidden


def nn_train(epoch):
    global W1, W2
    # iteration step
    lambda_ = 0.01
    # number of iterations
    n = 10000
    count = len(epoch)
    for k in range(n):
        # get random input vector of values
        x = epoch[np.random.randint(0, count)]
        # pass input through neural network
        y, output_hidden = nn_forward(x[0:3])
        # error (deviation between received and expected values)
        e = y - x[-1]
        # out layer local gradient
        delta_out = e * derivative(y)
        # output layer weights correction
        W2[0] = W2[0] - lambda_ * delta_out * output_hidden[0]
        W2[1] = W2[1] - lambda_ * delta_out * output_hidden[1]
        # hidden layer local gradients
        delta_hidden = W2 * delta_out * derivative(output_hidden)
        # hidden layer weights correction
        W1[0, :] = W1[0, :] - np.array(x[0:3]) * delta_hidden[0] * lambda_
        W1[1, :] = W1[1, :] - np.array(x[0:3]) * delta_hidden[1] * lambda_


def main():
    os_path()

    nn_train(EPOCH)

    for x in EPOCH:
        y, output_hidden = nn_forward(x[0:3])
        print(f"Out value: {y} => {x[-1]}")
    pass


if __name__ == "__main__":
    main()
