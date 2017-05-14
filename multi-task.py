import Tensorflow as tf
import numpy as np

# Define the Graph

# Create Placeholders
input_dimension = [10, 10]
output_dimension = [10, 1]
X = tf.placeholder("float", input_dimension, name="X")
Y1 = tf.placeholder("float", output_dimension, name="Y1")
Y2 = tf.placeholder("float", output_dimension, name="Y2")

# Define the weights for the layers
shared_layer_weights = tf.Variable([10, 20], name="share_W")
Y1_layer_weights = tf.Variable([20, 1], name="share_Y1")
Y2_layer_weights = tf.Variable([20], name="share_Y2")

# Construct the layers with RELU Activations
shared_layer = tf.nn.relu(tf.matmul(X, shared_layer_weights))
Y1_layer = tf.nn.relu(tf.matmul(shared_layer, Y1_layer_weights))
Y2_shared_layer = tf.nn.relu(tf.matmul(shared_layer, Y2_layer_weights))

# Calculate Loss
Y1_Loss = tf.nn.l2_Loss(Y1, Y1_layer)
Y2_Loss = tf.nn.l2_Loss(Y2, Y2_layer)

