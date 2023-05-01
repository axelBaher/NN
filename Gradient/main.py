# import numpy as np
# import warnings
# import os
# import matplotlib.pyplot as plt
# import time
#
#
# def f(x):
#     return np.sin(x) + 0.5 * x
#     # return x ** 2 - 5 * x + 5
#
# def df(x):
#     return np.cos(x) + 0.5
#     # return 2 * x - 5
#
#
# def main():
#     warnings.filterwarnings("ignore")
#     cur_dir = os.path.dirname(os.path.realpath(__file__))
#     os.chdir(cur_dir)
#     print(f"\nCurrent directory:\n{os.getcwd()}\n")
#
#     N = 40
#     x0 = 0
#     mn = 1000
#
#     x_plt = np.arange(-7.0, 7.0, 0.1)
#     f_plt = [f(x) for x in x_plt]
#
#     plt.ion()
#     fig, ax = plt.subplots()
#     ax.grid(True)
#
#     ax.plot(x_plt, f_plt)
#     point = ax.scatter(x0, f(x0), c = 'red')
#
#     for i in range(N):
#         lambda_ = 1 / min(i + 1, mn)
#         x0 = x0 - lambda_ * np.sign(df(x0))
#         point.set_offsets([x0, f(x0)])
#
#         fig.canvas.draw()
#         fig.canvas.flush_events()
#         time.sleep(0.02)
#
#     plt.ioff()
#     print(x0)
#     ax.scatter(x0, f(x0), c = 'blue')
#     plt.show()
#     pass
#
#
# if __name__ == "__main__":
#     main()