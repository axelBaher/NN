# import time
# import tensorflow as tf
# import warnings
# import os
#
#
# def os_path():
#     warnings.filterwarnings("ignore")
#     cur_dir = os.path.dirname(os.path.realpath(__file__))
#     os.chdir(cur_dir)
#     print(f"\nCurrent directory:\n{os.getcwd()}\n")
#
#
# class DenseLayer(tf.keras.layers.Layer):
#     def __init__(self, units=1):
#         super().__init__()
#         self.units = units
#
#     def build(self, input_shape):
#         self.w = self.add_weight(shape=(input_shape[-1], self.units),
#                                  initializer="random_normal",
#                                  trainable=True)
#         self.b = self.add_weight(shape=(self.units,),
#                                  initializer="zeros",
#                                  trainable=True)
#
#     def call(self, inputs):
#         return tf.matmul(inputs, self.w) + self.b
#
#
# class NeuralNetwork(tf.keras.layers.Layer):
#     def __init__(self):
#         super().__init__()
#         self.layer_1 = DenseLayer(128)
#         self.layer_2 = DenseLayer(10)
#
#     def call(self, inputs):
#         x = self.layer_1(inputs)
#         x = tf.nn.relu(x)
#         x = self.layer_2(x)
#         x = tf.nn.softmax(x)
#         return x
#
#
# def main():
#     os_path()
#
#     model = NeuralNetwork()
#     y = model(tf.constant([[1., 2., 3.]]))
#     print(y)
#     start_proc = time.perf_counter()
#     end_proc = time.perf_counter()
#     print(f"Processing time:\n{end_proc - start_proc} sec\n")
#
#     pass
#
#
# if __name__ == "__main__":
#     main()
