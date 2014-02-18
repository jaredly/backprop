#!/usr/bin/env python

from learner import Learner
from net import Net
import numpy as n

LOG = False

class BackProp(Learner):
    def __init__(self, meta, layers=[], rate=.05, target=None, momentum=None, trans=None, wrange=100):
        Learner.__init__(self, meta, target)

        inputs = len(self.meta.names()) - 1
        _, possible = self.meta[self.target]
        self.outputs = possible
        self.net = Net([inputs] + layers + [len(possible)], rate=rate, momentum=momentum, wrange=wrange, trans=trans)

    def state(self):
        return [x.copy() for x in self.net.weights]

    def use_state(self, state):
        self.net.weights = state

    def classify(self, data):
        output = self.net.classify(data)
        # print 'result'
        # print output
        # print 'result', output, self.outputs
        return self.outputs[output[-1].argmax()]

    def validate(self, data, real):
        output = self.net.classify(data)[-1]
        label = self.outputs[output.argmax()]
        target = n.zeros(len(self.outputs))
        target[self.outputs.index(real)] = 1
        squerr = (target - output)**2
        return label, squerr.mean()

    def train(self, data, target):
        output = n.zeros(len(self.outputs))
        # print self.outputs, target
        output[self.outputs.index(target)] = 1
        if LOG:
            print 'training'
            print 'data', data
            print 'expected', output
            print 'weights'
            for level in self.net.weights:
                print '  ', level
        err = self.net.train(data, output)
        return err

# vim: et sw=4 sts=4
