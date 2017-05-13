import Tensorflow as tf
import numpy as np

# Define the Graph

# Create Placeholders
input_dimension = [10, 10]
output_dimension = [10, 1]
X = tf.placeholder("float", input_dimension, name="X")
Y = tf.placeholder("float", output_dimension, name="Y")

# Create Trainable Variable
initial_W = np.zeros((10, 1))
W = tf.Variable(initial_W, name="W", dtype="float32")

# Define Loss Function
Loss = tf.pow(tf.add(Y, -tf.matmul(X,W)), 2, name="Loss")

# Set up the session
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    Model_Loss = sess.run(
        Loss,
        {
            X: np.random.rand(10, 10),
            Y: np.random.rand(10).reshape(-1, 1)
        })
    print(Model_Loss)
