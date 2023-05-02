import numpy as np
import warnings
import os


def os_path():
    warnings.filterwarnings("ignore")
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cur_dir)
    print(f"\nCurrent directory:\n{os.getcwd()}\n")


def activation_function_1(x):
    y = 0 if x < 0.5 else 1
    print(f"Activation function:\nx = {x}, y = {y}")
    return y


def run_1(attr1, attr2, attr3):
    # generate input layer
    x = np.array([attr1, attr2, attr3])
    print(f"Input layer values:\n{x}")

    # generate weights for hidden layer neurons
    w11 = [0.3, 0.3, 0.0]
    w12 = [0.4, -0.5, 1.0]
    # reshape to 2x3 matrix
    weight1 = np.array([w11, w12])
    print(f"Hidden layer weights:\n{weight1}")

    # generate weights for output layer neurons [1x2]
    weight2 = np.array([-1, 1])
    print(f"Output layer weights:\n{weight2}")

    # calculate sums of hidden layer neurons including weights
    sum_hidden = np.dot(weight1, x)
    print(f"Values on input of hidden layer neurons:\n{sum_hidden}")

    # pass sums through activation function
    out_hidden = np.array([activation_function_1(x) for x in sum_hidden])
    print(f"Values on output of hidden layer neurons:\n{out_hidden}")

    # calculate output layer value
    sum_output = np.dot(weight2, out_hidden)
    y = activation_function_1(sum_output)
    print(f"Output value:\n{y}")


def NN_1():
    run_1(attr1=1, attr2=0, attr3=1)


def activation_function_2(x):
    y = 0 if x < 0.5 else 1
    return y


def run_2(attr1, attr2, attr3):
    x = np.array([attr1, attr2, attr3])

    w11 = [0.3, 0.4, -0.6]
    w12 = [0.5, -0.1, 0.7]
    weight1 = np.array([w11, w12])

    weight2 = np.array([0.7, -0.3])

    sum1 = np.dot(weight1, x)

    res1 = np.array([activation_function_2(x) for x in sum1])

    sum2 = np.dot(weight2, sum1)

    res = activation_function_2(sum2)
    print(res)


def NN_2():
    if run_2(attr1=1, attr2=0, attr3=1):
        print("Apple")
    else:
        print("Orange")


def activation_function_3(x):
    y = 0 if x < 0.5 else 1
    return y


def run_3(attr1, attr2, attr3):
    x = np.array([attr1, attr2, attr3])

    w11 = [0.3, 0.5, -0.1]
    w12 = [0.7, 0.3, -0.7]
    w13 = [0.1, -0.3, -0.2]

    weight1 = np.array([w11, w12, w13])

    weight2 = np.array([0.4, -1, 0.8])

    sum1 = np.dot(weight1, x)

    res1 = np.array([activation_function_3(x) for x in sum1])

    sum2 = np.dot(weight2, res1)

    res = activation_function_3(sum2)

    print(res)

    if res:
        print("Like")
    else:
        print("Dislike")


def NN_3():
    run_3(attr1=1, attr2=0, attr3=1)


def main():
    os_path()

    # NN_1()
    # NN_2()
    # NN_3()
    pass


if __name__ == "__main__":
    main()
