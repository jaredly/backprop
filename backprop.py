#!/usr/bin/env python

from learner import Learner
from numpy import e, array, zeros
import numpy as n


def f(net):
    return 1/(1 + e^(-net))

def fprime(net):
    z = f(net)
    return z * (1 - z)

# def deltaw(netz

class Net:
    def __init__(self, layers):
        self.weights = []
        for prev, next in zip(layers[:-1], layers[1:]):
            self.weights.push(n.random.rand(prev + 1, next))

    def forward(self, input):
        inputs = [input + [1]]
        outputs = []
        out = input
        for level in self.weights:
            input = n.concatenate((input, (1,)))
            weighted = input * level
            out = f(weighted)
            outputs.push(out)
            k

            

            
            

        

class BackProp:
    def __init__(self, meta, layers, target=None):
        Target.__init__(self, meta, target)

    def state(self):
        return self.weights.copy()

    def train(self, data):
        target

    def epoch(self, data):
        '''returns accuracy'''
        for i in data.index:
            results.append(self.train(




# vim: et sw=4 sts=4
