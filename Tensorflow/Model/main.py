# import time
# start_import = time.perf_counter()
# import tensorflow as tf
# import warnings
# import os
# end_import = time.perf_counter()
#
#
# print(f"\nImport time:\n{end_import - start_import}")
#
#
# class DenseNN(tf.Module):
#     def __init__(self, outputs):
#         super().__init__()
#         self.outputs = outputs
#         self.flag_init = False
#
#     def __call__(self, x):
#         if not self.flag_init:
#             self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev = 0.1, name = 'w')
#             self.b = tf.zeros([self.outputs], dtype = tf.float32, name = 'b')
#
#             self.w = tf.Variable(self.w)
#             self.b = tf.Variable(self.b)
#
#             self.flag_init = True
#
#         y = x @ self.w + self.b
#         return y
#
#
# def main():
#     warnings.filterwarnings("ignore")
#     cur_dir = os.path.dirname(os.path.realpath(__file__))
#     os.chdir(cur_dir)
#     print(f"\nCurrent directory:\n{os.getcwd()}\n")
#
#     model = DenseNN(3)
#
#     x_train = tf.random.uniform(minval = 0, maxval = 10, shape = (100, 2))
#     y_train = [a + b for a, b in x_train]
#
#     loss = lambda out, out_real: tf.reduce_mean(tf.square(out - out_real))
#     opt = tf.optimizers.Adam(learning_rate = 0.01)
#
#     EPOCHS = 50
#
#     start_proc = time.perf_counter()
#
#     for _ in range(EPOCHS):
#         for x, y in zip(x_train, y_train):
#             x = tf.expand_dims(x, axis = 0)
#             y = tf.constant(y, shape = (1, 1))
#
#             with tf.GradientTape() as tape:
#                 f_loss = loss(y, model(x))
#
#             grads = tape.gradient(f_loss, model.trainable_variables)
#             opt.apply_gradients(zip(grads, model.trainable_variables))
#
#         print(f"Current loss function value: {f_loss.numpy()}")
#         if (f_loss.numpy() < 10e-8):
#             break
#
#     end_proc = time.perf_counter()
#
#     print(model.trainable_variables)
#     print(model(tf.constant([[1.0, 2.0]])))
#
#     print(f"Processing time:\n{end_proc - start_proc}\n")
#
#     print(f"Overall time:\n{(end_proc - start_proc) + (end_import - start_import)}")
#     pass
#
#
# if __name__ == "__main__":
#     main()