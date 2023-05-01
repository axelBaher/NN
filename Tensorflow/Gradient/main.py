import tensorflow as tf
import numpy as np
import warnings
import os


def main():
    warnings.filterwarnings("ignore")
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cur_dir)
    print(f"\nCurrent directory:\n{os.getcwd()}\n")
    
    '''
    x = tf.Variable(0.0)
    b = tf.constant(1.5)
    c = tf.Variable(1.0, trainable = False)
    # After GradientTape.gradient() execution, all watched calculation will be free
    # If GradientTape(persisent = True), memory will not be free after GradientTape.gradient() execution
    # In that case, after using GradientTape object, we need to delete it: del tape
    with tf.GradientTape(watch_accessed_variables = False) as tape:
        tape.watch([x, b, c])
        f = (x + b) ** 2 + 2 * b + 3 * c
    df = tape.gradient(f, [x, b, c])
    print(df[0], df[1], df[2], sep = '\n')
    '''

    '''
    x = tf.Variable(1.0)
    with tf.GradientTape() as tape:
        y = [2.0, 3.0] * x ** 2
    df = tape.gradient(y, x)
    print(df)
    '''

    '''
    x = tf.Variable([1.0, 2.0])
    with tf.GradientTape() as tape:
        y = tf.reduce_sum([1.0, 2.0]) * x ** 2
    df = tape.gradient(y, x)
    print(df)
    '''

    '''
    w = tf.Variable(tf.random.normal([3, 2]))
    b = tf.Variable(tf.zeros(2, dtype = tf.float32))
    x = tf.Variable([[-2.0, 1.0, 3.0]])
    print(w, b, x, sep = '\n')
    with tf.GradientTape() as tape:
        y = x @ w + b
        loss = tf.reduce_mean(y ** 2)
    df = tape.gradient(loss, [w, b])
    print(df[0], df[1], sep = '\n')
    '''    
    pass
    

if __name__ == "__main__":
    main()