#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 6 16:27:09 2017

@author: zamri
"""

import tensorflow as tf
a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.mul(a, b, name="mul_c")
d = tf.add(a, b, name="add_d")
e = tf.add(c, d, name="add_e")

sess = tf.Session()
sess.run(e)

print(sess.run(e))
output = sess.run(e)

tf.summary.scalar('a', a)
tf.summary.scalar('b', b)
tf.summary.scalar('c', c)
tf.summary.scalar('d', d)
tf.summary.scalar('e', e)
writer = tf.summary.FileWriter('./my_graph', sess.graph)
# tensorboard --logdir="my_graph"

writer.close()
sess.close()
