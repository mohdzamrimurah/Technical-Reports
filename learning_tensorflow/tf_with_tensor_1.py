#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 6 17:41:56 2017

@author: zamri
"""

import tensorflow as tf
a = tf.constant([5, 3, 9, 12], name="input_a")
b = tf.reduce_prod(a, name="prod_b")
c = tf.reduce_sum(a, name="sum_c")
d = tf.add(b, c, name="add_d")

sess = tf.Session()
sess.run(d)

print(sess.run(d))
output = sess.run(d)

writer = tf.summary.FileWriter('./my_graph0', sess.graph)
# tensorboard --logdir="my_graph"

writer.close()
sess.close()
