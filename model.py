import tensorflow as tf
import numpy as np

_NUM_CHANELS = 3
_IMAGE_SIZE = 224


def model(regularzier):
    X = tf.placeholder(tf.float32, [None, _IMAGE_SIZE*_IMAGE_SIZE*3], 'x-input')
    Y = tf.placeholder(tf.float32, [None, 8], 'y-input')
    image = tf.reshape(X, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANELS])

    with tf.variable_scope('conv1'):
        conv1_weights = tf.get_variable('weight', shape=[7, 7, 3, 64], initializer=tf.truncated_normal_initializer(1/2000))
        conv1_biases = tf.get_variable('biase', shape=[64], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(image, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    norm1 = tf.nn.lrn(relu1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv2'):
        conv2_weights = tf.get_variable('weight', shape=[5, 5, 64, 128], initializer=tf.truncated_normal_initializer(1e-7))
        conv2_biases = tf.get_variable('biase', shape=[128], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv3'):
        conv3_weights = tf.get_variable('weight', shape=[3, 3, 128, 128], initializer=tf.truncated_normal_initializer(1e-5))
        conv3_biases = tf.get_variable('biase', shape=[128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

    with tf.variable_scope('conv4'):
        conv4_weights = tf.get_variable('weight', shape=[3, 3, 128, 256], initializer=tf.truncated_normal_initializer(1e-5))
        conv4_biases = tf.get_variable('biase', shape=[256], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv5'):
        conv5_weights = tf.get_variable('weight', shape=[3, 3, 256, 256], initializer=tf.truncated_normal_initializer(1e-5))
        conv5_biases = tf.get_variable('biase', shape=[256], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(pool4, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))
    norm3 = tf.nn.lrn(relu5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool5 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    pool_shape = pool5.get_shape().as_list()
    node = pool_shape[1] * pool_shape[2] * pool_shape[3]

    with tf.variable_scope('fc-1'):
        reshape = tf.reshape(pool5, [-1, node])
        fc1_weights = tf.get_variable('weight', [node, 800], initializer=tf.truncated_normal_initializer(0.01))
        fc1_biases = tf.get_variable('biase', [800], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        if regularzier != None:
            tf.add_to_collection('losses', regularzier(fc1_weights))
        fc1 = tf.nn.dropout(fc1, 0.8)

    with tf.variable_scope('fc-2'):
        fc2_weights = tf.get_variable('weight', [800, 8], initializer=tf.truncated_normal_initializer(0.1))
        fc2_biases = tf.get_variable('biase', [8], initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if regularzier != None:
            tf.add_to_collection('losses', regularzier(fc2_weights))
    return X, Y, fc2