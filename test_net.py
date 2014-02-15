#!/usr/bin/env python

from net import Net
import numpy as n

basic_weights1 = [
    n.array([[ 1.         , 1.        ,  1.00112266],
             [ 1.         , 1.        ,  1.00112266]]),
    n.array([[ 1.00415709 , 1.00415709,  1.0056864 ]])
]

basic_weights2 = [
    n.array([[ 1.        ,  0.99477124,  0.99588803],
             [ 1.        ,  0.99477124,  0.99588803]]),
    n.array([[ 0.95834062,  0.95834062,  0.9536763 ]])
]

def test_basic():
    net = Net([2, 2, 1], 1, weights=1)
    err = net.train([0, 0], [1])
    for a, b in zip(basic_weights1, net.weights):
        print a
        print b
        print a == b
        n.testing.assert_array_almost_equal(a, b)
    err = net.train([0, 1], [0])
    for a, b in zip(basic_weights2, net.weights):
        print a
        print b
        print a == b
        n.testing.assert_array_almost_equal(a, b)



# vim: et sw=4 sts=4
