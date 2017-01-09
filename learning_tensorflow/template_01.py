#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 23:49:44 2017

@author: zamri
"""

import tensorflow as tf
# initialize variables / model parameters# define the training loop operations


def inference(X):
    # compute inference model over data X and return the result

    return None

def loss(X,Y):
    # compute loss over training data X and expected outputs Y
    return None

def inputs():
    # read / generate input training data X and expected outputs Y
    return None

def train(total_loss):
    # train / adjust model parameters according to computed total loss
    return None

def evaluate(sess, X, Y):
    # evaluate the resulting trained model# Launch the graph in 
    # a session, setup boilerplate
    return None

saver = tf.train.Saver()

with tf.Session() as sess:

    tf.initialize_all_variables().run()
    X, Y = inputs()
    total_loss = loss(X, Y)

    train_op = train(total_loss)

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess = sess, 
        coord = coord)

    # actual training loop
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])  
        # for debugging and learning purposes, 
        # see how the loss gets decremented thru 
        # training steps
        if step % 10 == 0:
            print("loss: %f", sess.run([total_loss]))
        if step % 50 == 0:
            saver.save(sess, 'my-model', 
                       global_step=training_steps)

    evaluate(sess, X, Y)
    coord.request_stop()
    coord.join(threads)
    sess.close()
