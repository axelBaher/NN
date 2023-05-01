# import time
# import tensorflow as tf
# import warnings
# import os
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.utils import to_categorical
#
#
# class DenseNN(tf.Module):
#     def __init__(self, outputs, activate="relu"):
#         super().__init__()
#         self.outputs = outputs
#         self.activate = activate
#         self.flag_init = False
#
#     def __call__(self, x):
#         if not self.flag_init:
#             self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name='w')
#             self.b = tf.zeros([self.outputs], dtype=tf.float32, name='b')
#
#             self.w = tf.Variable(self.w)
#             self.b = tf.Variable(self.b)
#
#             self.flag_init = True
#
#         y = x @ self.w + self.b
#
#         if self.activate == "relu":
#             return tf.nn.relu(y)
#         elif self.activate == "softmax":
#             return tf.nn.softmax(y)
#
#         return y
#
#
# class SequentialModule(tf.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer_1 = DenseNN(128)
#         self.layer_2 = DenseNN(10, activate="softmax")
#
#     def __call__(self, x):
#         return self.layer_2(self.layer_1(x))
#
#
# def train_batch(x_batch, y_batch):
#     with tf.GradientTape() as tape:
#         f_loss = cross_entropy(y_batch, model(x_batch))
#
#     grads = tape.gradient(f_loss, model.trainable_variables)
#     opt.apply_gradients(zip(grads, model.trainable_variables))
#     return f_loss
#
#
# def main():
#     warnings.filterwarnings("ignore")
#     cur_dir = os.path.dirname(os.path.realpath(__file__))
#     os.chdir(cur_dir)
#     print(f"\nCurrent directory:\n{os.getcwd()}\n")
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
#
#     global model
#     model = SequentialModule()
#
#     global cross_entropy
#     cross_entropy = lambda y_true, y_pred: tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_pred))
#     global opt
#     opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
#
#     BATCH_SIZE = 32
#     EPOCHS = 10
#
#     train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#     train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)
#
#     start_proc = time.perf_counter()
#
#     for _ in range(EPOCHS):
#         loss = 0
#         for x_batch, y_batch in train_dataset:
#             loss += train_batch(x_batch, y_batch)
#
#         print(f"Current loss function value: {loss.numpy()}")
#
#     y = model(x_test)
#     y2 = tf.argmax(y, axis=1).numpy()
#     acc = tf.metrics.Accuracy()
#     acc.update_state(y_test, y2)
#     print(f"Accuracy: {round(acc.result().numpy() * 100, 2)}%")
#
#     end_proc = time.perf_counter()
#     print(f"Processing time:\n{end_proc - start_proc}\n")
#
#     pass
#
#
# if __name__ == "__main__":
#     main()
