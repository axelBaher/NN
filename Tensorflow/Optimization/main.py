# import time
# start_import = time.perf_counter()
# import tensorflow as tf
# import numpy as np
# import warnings
# import os
# import matplotlib.pyplot as plt
# end_import = time.perf_counter()
#
#
# print(f"\nImport time:\n{end_import - start_import}")
#
#
# def main():
#     warnings.filterwarnings("ignore")
#     cur_dir = os.path.dirname(os.path.realpath(__file__))
#     os.chdir(cur_dir)
#     print(f"\nCurrent directory:\n{os.getcwd()}\n")
#
#     TOTAL_POINTS = 1000
#     EPOCHS = 50
#     learn_rate = 0.02
#     BATCH_SIZE = 100
#     num_steps = TOTAL_POINTS // BATCH_SIZE
#     x = tf.random.uniform(shape = [TOTAL_POINTS], minval = 0, maxval = 10)
#     noise = tf.random.normal(shape = [TOTAL_POINTS], stddev = 0.2)
#
#     k_true = 0.7
#     b_true = 2.0
#
#     y = x * k_true + b_true + noise
#
#     k = tf.Variable(0.0)
#     b = tf.Variable(0.0)
#
#     start_proc = time.perf_counter()
#
#     opt = tf.optimizers.SGD(momentum = 0.0, nesterov = False, learning_rate = learn_rate)
#
#     for _ in range(EPOCHS):
#         for n_batch in range(num_steps):
#             cur_batch = n_batch * BATCH_SIZE
#             next_batch = (n_batch + 1) * BATCH_SIZE
#             y_batch = y[cur_batch : next_batch]
#             x_batch = x[cur_batch : next_batch]
#
#             with tf.GradientTape() as tape:
#                 f = k * x_batch + b
#                 loss = tf.reduce_mean(tf.square(y_batch - f))
#
#             dk, db = tape.gradient(loss, [k, b])
#
#             opt.apply_gradients(zip([dk, db], [k, b]))
#             # k.assign_sub(learn_rate * dk)
#             # b.assign_sub(learn_rate * db)
#
#     end_proc = time.perf_counter()
#     print(f"Processing time:\n{end_proc - start_proc}\n")
#
#     print(k, b, sep = '\n')
#     y_pr = k * x + b
#     plt.scatter(x, y, s = 2)
#     plt.scatter(x, y_pr, c = 'r', s = 2)
#     plt.show()
#
#     print(f"\nOverall time:\n{(end_proc - start_proc) + (end_import - start_import)}")
#     pass
#
#
# if __name__ == "__main__":
#     main()