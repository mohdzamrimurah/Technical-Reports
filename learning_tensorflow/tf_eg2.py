#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 01:16:56 2017

@author: zamri
"""

import tensorflow as tf

# 2x2 matrix	of zeros
zeros = tf.zeros([2,	2])
# vector	 of length 6 of ones
ones = tf.ones([6])
# 3x3x3	Tensor of random	uniform	 values between 0 and 10
uniform = tf.random_uniform([3, 3, 3], minval=0, maxval=10)
# 3x3x3	Tensor of normally distributed	numbers; mean 0 and standard deviation 2
normal = tf.random_normal([3, 3, 3], mean=0.0,	stddev=2.0)

# sess = tf.Session()
# sess.run([zeros, ones, uniform, normal])

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print(sess.run([zeros, ones, uniform, normal]))
