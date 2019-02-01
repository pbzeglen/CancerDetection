"""
This is an example of a fully convolutional network prediction pipeline.
The data used is from the Histopathologic Cancer Detection playground challenge on Kaggle (link:https://www.kaggle.com/c/histopathologic-cancer-detection)

This module defines the model and trains it.

The model was inspired by https://arxiv.org/pdf/1412.6806.pdf
There are modifications (based on the different size of the data set)
The choice not to use batchnorm, dropout, or l2 regularization was based on the paper's findings,
(there is no fully connected layer, thus reducing complexity).

Additionally, we apply regularization with our test set (since it is available) as an unlabeled set.
We expect the model to have the same classification for an image no matter how it is rotated,
so the loss is trained to the variance in the prediction of a single image no matter how it is flipped or rotated.

Author: Peter Zeglen

"""

import tensorflow as tf
import numpy as np

# Whether to use regularization
USE_REGULARIZATION = True

# Whether to use one cycle policy (alternative is fixed learning rate = 1e-4)
USE_ONE_CYCLE = True

sess = tf.InteractiveSession()

tf.set_random_seed(0)

# x,y represent the supervised training pair
x = tf.placeholder(shape=[None, 64, 64, 3], dtype=tf.float32)
y = tf.placeholder(shape=[None], dtype=tf.float32)

# x is from the unlabeled data set: it is used to enforce rotation invariance in the model
x_test_reg = tf.placeholder(shape=[None, 64, 64, 3], dtype=tf.float32)

rate = tf.placeholder(shape=[], dtype=tf.float32)


def conv(x, w, stride=1):
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')


def weight(shape):
    return tf.random_normal(shape=shape, stddev=2 / np.sqrt(shape[0] * shape[1] * shape[2]))


def full_model_pass(x_input):
    depths_in = [32, 64, 128, 400]
    depths_out = [64, 128, 400, 1]

    w_wide = tf.get_variable("w_wide", initializer=weight([7, 7, 3, 32]))
    b_wide = tf.get_variable("b_wide", initializer=tf.zeros([32]))
    x_wide = tf.nn.relu(conv(x_input, w_wide) + b_wide)

    w_pool = tf.get_variable("w_pool", initializer=weight([5, 5, 32, 32]))
    b_pool = tf.get_variable("b_pool", initializer=tf.zeros([32]))
    x_pool = [tf.nn.relu(conv(x_wide, w_pool, stride=2) + b_pool)] # downconvolution

    for depth, depth_out in zip(depths_in, depths_out):
        with tf.variable_scope("layer_" + str(depth) + "_" + str(depth_out)):
            w5 = tf.get_variable("w5", initializer=weight([5, 5, depth, depth]))
            b5 = tf.get_variable("b5", initializer=tf.zeros([depth]))
            x5 = tf.nn.relu(conv(x_pool[-1], w5) + b5)

            w1 = tf.get_variable("w1", initializer=weight([1, 1, depth, depth]))
            b1 = tf.get_variable("b1", initializer=tf.zeros([depth]))
            x1 = tf.nn.relu(conv(x5, w1) + b1)

            w3 = tf.get_variable("w3", initializer=weight([3, 3, depth, depth_out]))
            b3 = tf.get_variable("b3", initializer=tf.zeros([depth_out]))
            x_pool.append(conv(x1, w3, stride=2) + b3) # downconvolution

    final_logits = tf.reduce_max(x_pool[-1], axis=(1, 2, 3))
    #We use reduce max based on the problem domain: a value of 1 represents an aberrant growth.
    #The maximum value, therefore, is taken as the logit probability of a positive example.

    return final_logits


# Prediction output
with tf.variable_scope("model", reuse=False):
    training_logits = full_model_pass(x)


with tf.variable_scope("model", reuse=True):
    output_for_regularization = tf.stack([
        full_model_pass(x_test_reg),
        full_model_pass(tf.reverse(x_test_reg, axis=[1])),
        full_model_pass(tf.reverse(x_test_reg, axis=[2])),
        full_model_pass(tf.transpose(x_test_reg, perm=[0, 2, 1, 3]))
    ], axis=0)

l1 = .001
#Calculate the variation in the output for the same image, rotated four ways
#These images are pulled from the test set (this loss promotes consistency)
regularization_loss = tf.reduce_mean(tf.nn.moments(output_for_regularization, axes=0)[1])

cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=training_logits, labels=y))
accuracy = tf.reduce_mean(tf.cast(tf.equal(training_logits<0, y<.5), tf.float32))

if USE_REGULARIZATION:
    step = tf.train.AdamOptimizer(rate).minimize(cross_entropy_loss)
else:
    step = tf.train.AdamOptimizer(rate).minimize(cross_entropy_loss + l1 * regularization_loss)


saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
print(tf.trainable_variables())


'''
We rely on a 1cycle policy
(phrase introduced in this paper: https://arxiv.org/abs/1803.09820,
but this fastai turtorial was relied on: https://sgugger.github.io/the-1cycle-policy.html).
Adam's parameters are adjusted in an inverted triange over the first 45 epochs, and then level off for the rest of the training.
We only adjust the learning rate in this code.
The learning rate modulates between .008 and .08, and then descends down to 1e-4 for the remaining epochs
'''


def get_learning_rate_and_momentum(batch_number, final_number=200000):
    if batch_number < 45000:
        return (.01 - .0008) * np.abs(batch_number - 22500) / 22500
    else:
        return .0008 - .0007 * (batch_number - 45000) / (final_number - 45000)


from load_data import get_batch, get_test, get_validation
for i in range(100000):
    if USE_ONE_CYCLE:
        r = get_learning_rate_and_momentum(i, 100000)
    else:
        r = 1e-4

    x_batch, y_batch = get_batch()
    x_test = get_test(5)
    step.run(feed_dict={x: x_batch, x_test_reg: x_test, y: y_batch, rate: r})

    if i % 100 == 0:
        x_val, y_val = get_validation(50)
        print(i)
        print(accuracy.eval(feed_dict={x: x_batch, y: y_batch}))


saver.save(sess, 'models/fully_conv')
