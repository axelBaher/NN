# import time
# import tensorflow as tf
# import warnings
# import os
# from keras.datasets import mnist
# from keras.utils import to_categorical
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
# class NeuralNetwork(tf.keras.Model):
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
#     # model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
#     #               loss=tf.losses.categorical_crossentropy,
#     #               metrics=["accuracy"])
#     model.compile(optimizer="adam",
#                   loss="categorical_crossentropy",
#                   metrics=["accuracy"])
#
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
#     x_train = x_train / 255
#     x_test = x_test / 255
#
#     x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28 * 28])
#     x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28 * 28])
#
#     y_train = to_categorical(y_train, 10)
#     y_test_cat = to_categorical(y_test, 10)
#
#     sp = time.perf_counter()
#     model.fit(x_train, y_train, batch_size=32, epochs=5)
#     ep = time.perf_counter()
#     print(f"\nLearning time:\n{ep - sp} sec\n")
#
#     sp = time.perf_counter()
#     print(model.evaluate(x_test, y_test_cat))
#     ep = time.perf_counter()
#     print(f"\nTesting time:\n{ep - sp} sec\n")
#
#     pass
#
#
# if __name__ == "__main__":
#     main()
