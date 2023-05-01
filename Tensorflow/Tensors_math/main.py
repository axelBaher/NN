# import tensorflow as tf
# import numpy as np
# import warnings
# import os
#
#
# def main():
#     warnings.filterwarnings("ignore")
#     cur_dir = os.path.dirname(os.path.realpath(__file__))
#     os.chdir(cur_dir)
#     print(f"\nCurrent directory:\n{os.getcwd()}\n")
#
#     '''
#     const = tf.constant(1, shape = (1, 1))
#     const_cast = tf.cast(const, dtype = tf.float32)
#     const_np = const.numpy()
#     print("Const tensor:", const, const_cast, const_np, sep = '\n')
#     var = tf.Variable(7, name = "My var")
#     var_cast = tf.cast(var, dtype = tf.float32)
#     var_np = var.numpy()
#     print("Var tensor:", var, var_cast, var_np, sep = '\n')
#     '''
#
#     '''
#     var = tf.Variable(7, name = "My var")
#     var.assign(10)
#     print("Var tensor assign:", var)
#     var.assign_add(1)
#     print("Var tensor assign add:", var)
#     var.assign_sub(3)
#     print("Var tensor assign sub:", var)
#     '''
#
#     """
#     tensor_list = tf.constant(range(10)) + 10
#     tensor_list_idx = tf.gather(tensor_list, [0, 2, 4])
#     tensor_matrix = tf.constant([[1, 2, 3], [11, 12, 13]])
#     tensor_matrix_idx1 = tensor_matrix[0]
#     tensor_matrix_idx2 = tensor_matrix[0, 1]
#     tensor_matrix_idx3 = tensor_matrix[:, 1]
#     print("Tensor indexing:", tensor_list, tensor_list_idx, tensor_matrix, tensor_matrix_idx1, tensor_matrix_idx2, tensor_matrix_idx3, sep = '\n')
#     """
#
#     '''
#     reshape1 = tf.Variable(range(72))
#     reshape2 = tf.reshape(reshape1, [8, -1])
#     temp = reshape1[0]
#     temp.assign(100)
#     print("Tensor reshaping:", reshape1, reshape2, temp, sep = '\n')
#     reshape2_T = tf.transpose(reshape2, perm = [1, 0])
#     print("Tensor transpose:", reshape2_T, sep = '\n')
#     '''
#
#     '''
#     a = tf.Variable(tf.ones([5, 3]))
#     b = tf.Variable(tf.ones_like(a))
#     print(a, b, sep = '\n')
#     '''
#
#     '''
#     a = tf.Variable([1, 2, 3])
#     b = tf.Variable([9, 8, 7])
#     c = tf.tensordot(a, b, axes = 0)
#     d = tf.tensordot(a, b, axes = 1)
#     print(a, b, c, d, sep = '\n')
#     '''
#
#     '''
#     a = tf.Variable(tf.ones([3, 3], dtype = tf.dtypes.int32))
#     b = tf.Variable(tf.fill([3, 3], 2))
#     c = tf.matmul(a, b)
#     d = a @ b
#     print(a, b, c, d, sep = '\n')
#     '''
#
#     '''
#     a = tf.Variable([1, 2, 3])
#     b = tf.Variable([9, 8, 7])
#     c = tf.tensordot(a, b, axes = 0)
#     d = tf.reduce_logsumexp(tf.cast(c, dtype = float))
#     print(a, b, c, d, sep = '\n')
#     '''
#
#     '''
#     a = tf.Variable([1, 2, 3])
#     b = tf.Variable([9, 8, 7])
#     c = tf.tensordot(a, b, axes = 0)
#     d = tf.reduce_sum(c, axis = 0)
#     e = tf.reduce_sum(c, axis = 1)
#     f = tf.reduce_sum(c, axis = [0, 1])
#     print(a, b, c, d, e, f, sep = '\n')
#     '''
#
#
#     pass
#
#
# if __name__ == "__main__":
#     main()