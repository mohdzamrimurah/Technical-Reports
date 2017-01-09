#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 00:17:46 2017

@author: zamri
"""

# logistic regression


import tensorflow as tf

# initialize variables / model parameters
# define the training loop operations

W = tf.Variable(tf.zeros([2, 1]), name="weights")
# array([[0.],
#        [0.]], dtype=float32)
b = tf.Variable(0., name="bias")

def combine_inputs(X):
    #return tf.matmul(X,W) + b
    return tf.sigmoid(combine_inputs(X))

def inference(X):
    return tf.matmul(X, W) + b
    # compute inference model over data X and return the result


def loss(X, Y):
    # compute loss over training data X and expected outputs Y
    #Y_predicted = inference(X)
    #return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(combine_inputs(X), Y))

def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.dirname(__file__) + "/" + file_name])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(value, record_defaults=record_defaults)
    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded,
                batch_size=batch_size,
                capacity=batch_size * 50,
                min_after_dequeue=batch_size)


def train(total_loss):
    # train / adjust model parameters according to computed total loss
    learning_rate = 0.000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):
    # evaluate the resulting trained model# Launch the graph in
    # a session, setup boilerplate
    print(sess.run(inference([[80., 25.]])))
    print(sess.run(inference([[65., 25.]])))


# saver = tf.train.Saver()

with tf.Session() as sess:

    tf.initialize_all_variables().run()
    X, Y = inputs()
    total_loss = loss(X, Y)

    train_op = train(total_loss)

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess = sess,
        coord = coord)

    # actual training loop
    training_steps = 500000
    for step in range(training_steps):
        sess.run([train_op])
        # for debugging and learning purposes,
        # see how the loss gets decremented thru
        # training steps
        if step % 10000 == 0:
            print("loss:", sess.run([total_loss]))
        #if step % 500 == 0:
        #    saver.save(sess, 'my-model',
        #               global_step=training_steps)

    evaluate(sess, X, Y)
    coord.request_stop()
    coord.join(threads)
    sess.close()
