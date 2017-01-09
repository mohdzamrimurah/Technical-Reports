#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 00:17:46 2017

@author: zamri
"""


import tensorflow as tf

# initialize variables / model parameters
# define the training loop operations

W = tf.Variable(tf.zeros([2, 1]), name="weights")
# array([[0.],
#        [0.]], dtype=float32)

b = tf.Variable(0., name="bias")


def inference(X):
    return tf.matmul(X, W) + b
    # compute inference model over data X and return the result


def loss(X, Y):
    # compute loss over training data X and expected outputs Y
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))


def inputs():
    # read / generate input training data X and expected outputs Y
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], 
                  [69, 25], [63, 28], [72, 36], [79 , 57],[75, 44]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 
                         402, 365]
    return tf.to_float(weight_age), tf.to_float(blood_fat_content)


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
