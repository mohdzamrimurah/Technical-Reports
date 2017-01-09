#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 18:24:33 2017

@author: zamri
"""

import tensorflow as tf
import numpy as np
# Initialize some tensors to use in computation
a = np.array([2, 3], dtype=np.int32)
b = np.array([4, 5], dtype=np.int32)
# Use `tf.add()` to initialize an "add" Operation
# The variable `c` will be a handle to the Tensor output of
# this Op
c = tf.add(a, b)

sess = tf.Session()
# sess.run(c)

# print(sess.run(c))
output = sess.run(c)

print(c)
writer = tf.summary.FileWriter('./my_graph1', sess.graph)
# tensorboard --logdir="my_graph"

writer.close()
sess.close()
