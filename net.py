#!/usr/bin/env python
from numpy import e, array, zeros
import numpy as n

log = False

def f(net):
    return 1/(1 + e**(-net))

def fprime(net):
    z = f(net)
    return z * (1 - z)

def fp(outs):
    return outs * (1 - outs)

# def deltaw(netz

class Net:
    def __init__(self, layers, rate=.05, weights=None, wrange=100):
        self.rate = rate
        self.weights = []
        for prev, next in zip(layers[:-1], layers[1:]):
            if weights is not None:
                level = n.zeros((next, prev+1))
                level += weights
            else:
                level = n.random.rand(next, prev+1) / wrange - .5/wrange
            self.weights.append(level)

    def train(self, input, target):
        input = n.array(input)
        target = n.array(target)
        outputs = self.forward(input)
        if log:
            print 'Outputs'
            for level in outputs:
                print '  ', level
        err = target - outputs[-1]
        delta = outputs[-1] * (1 - outputs[-1]) * err
        self.backward(outputs[:-1], delta)
        if log:
            print 'Weights'
            for level in self.weights:
                print '  ', level
        return (err**2).sum()

    def classify(self, input):
        return self.forward(input)[-1]

    def forward(self, input):
        '''Calculate outputs forward'''
        outputs = []
        out = n.array(input)
        for level in self.weights:
            bout = n.concatenate((out, (1,)))
            outputs.append(bout)
            # inp is the weighted outputs
            if log:
                print level, level.shape
                print bout, bout.shape
            inp = level.dot(bout)
            # out has the sigmoid applied
            out = f(inp)
        outputs.append(out)
        return outputs

    def backward(self, outputs, delta):
        '''propagate error backward'''
        if log:
            print 'Output'
            print '  ', outputs[-1]
            print '   Delta:', delta
        diff = self.rate * outputs[-1] * delta.reshape(delta.shape + (1,))
        if log:
            print 'Last weights diff', diff
        self.weights[-1] += diff
        for i in range(len(self.weights)-2, -1, -1):
            j = i + 1
            if log:
                print 'For level', i, j
                print '  Weights:', self.weights[j], self.weights[j].shape
                print '  Deltas:', delta, delta.shape
            delta = self.weights[j][:, :-1].transpose().dot(delta)
            fprime = fp(outputs[j][:-1])
            if log:
                print '  F\'', fprime
                print '  ToDelta', delta
            delta *= fprime
            if log:
                print '  Real Delta:', delta
                print '  And outputs:', outputs[i]
                print '  Onto weights:', self.weights[i]
            diff = self.weights[i] * outputs[i] * delta.reshape(delta.shape + (1,))
            if log:
                print '  Diff:', diff
            self.weights[i] += diff


# vim: et sw=4 sts=4