""" Neural Network.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import mesh_tensorflow as mtf

# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

global_step = tf.train.get_or_create_global_step()

graph = mtf.Graph()
mesh = mtf.Mesh(graph, "my_mesh")

# tf Graph input
batch_dim = mtf.Dimension("batch", batch_size)
io_dim = mtf.Dimension("io", num_input)
classes_dim = mtf.Dimension("classes", num_classes)

X = tf.placeholder("float", [batch_size, num_input])
Y = tf.placeholder("float", [batch_size, num_classes])

x = mtf.import_tf_tensor(mesh, X, [batch_dim, io_dim])
y = mtf.import_tf_tensor(mesh, Y, [batch_dim, classes_dim])

weight = mtf.get_variable(mesh, "weight", [io_dim, classes_dim])
bias = mtf.get_variable(mesh, "bias", [classes_dim])

# Create model
logits = mtf.einsum([x, weight], [batch_dim, classes_dim]) + bias
loss = mtf.layers.softmax_cross_entropy_with_logits(
    logits, y, classes_dim)
loss = mtf.reduce_mean(loss)

mesh_shape = mtf.convert_to_shape("c1:1;c2:4")
layout_rules = mtf.convert_to_layout_rules("batch:c1;io:c2")

mesh_size = mesh_shape.size
mesh_devices = ["/cpu:0"] * mesh_size
mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
    mesh_shape, layout_rules, mesh_devices)

var_grads = mtf.gradients(
    [loss], [v.outputs[0] for v in graph.trainable_variables])
optimizer = mtf.optimize.AdafactorOptimizer()
update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)

lowering = mtf.Lowering(graph, {mesh: mesh_impl})
tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
tf_update_ops.append(tf.assign_add(global_step, 1))
train_op = tf.group(tf_update_ops)

tf_logits = lowering.export_to_tf_tensor(logits)
tf_loss = lowering.export_to_tf_tensor(loss)
prediction = tf.nn.softmax(tf_logits)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        print(batch_x.size, batch_y.size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))
