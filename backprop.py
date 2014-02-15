#!/usr/bin/env python

from learner import Learner
from net import Net
import numpy as n

LOG = False

class BackProp(Learner):
    def __init__(self, meta, layers=[], rate=.05, target=None):
        Learner.__init__(self, meta, target)

        inputs = len(self.meta.names()) - 1
        _, possible = self.meta[self.target]
        self.outputs = possible
        self.net = Net([inputs] + layers + [len(possible)], rate=rate)

    def state(self):
        return [x.copy() for x in self.net.weights]

    def use_best(self):
        self.net.weights = self.best_state

    def classify(self, data):
        output = self.net.classify(data)
        return self.outputs[output.argmax()]

    def train(self, data, target):
        output = n.zeros(len(self.outputs))
        output[self.outputs.index(target)] = 1
        if LOG:
            print 'training'
            print 'data', data
            print 'expected', output
            print 'weights'
            for level in self.net.weights:
                print '  ', level
        err = self.net.train(data, output)
        return err / 2

# vim: et sw=4 sts=4
