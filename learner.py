#!/usr/bin/env python

import numpy as np
from pandas import DataFrame, Series
from scipy.io.arff import loadarff
from runner import Runner

class Learner:
    def __init__(self, meta, target=None):
        if target is None:
            target = meta.names()[-1]
        self.best = None
        self.best_state = None
        self.target = target
        self.meta = meta

    @classmethod
    def fromArff(cls, fname, **args):
        data, meta = loadarff(fname)
        data = DataFrame(data)
        t = meta.names()[-1]
        l = cls(meta, target=t, **args)
        return Runner(l, meta, t), data

    # OVERRIDE THESE
    def state(self):
        '''get current state'''
        raise Exception('Override')

    def use_best(self):
        '''set state to the best so far'''
        raise Exception('Override')

    def train(self, data, target):
        '''test a line, return Error amount'''
        raise Exception('Override')

    def classify(self, line):
        raise Exception('Override')

    # don't need to override
    def epoch(self, data, targets):
        '''run an epoch on the data. Calleds "train(line)"'''
        results = []
        for i in data.index:
            results.append(self.train(data.loc[i], targets.loc[i]))
        error = sum(results)/float(len(results))
        weights = self.state()
        if self.best is None or error < self.best:
            self.best = error
            self.best_state = weights
        return error, weights

class DefaultLearner(Learner):
    def __init__(self, meta, target=None):
        Learner.__init__(self, meta, target)
        self.seen = {}
        self.most = None

    def state(self):
        return self.seen.copy()

    def use_best(self):
        self.seen = self.best_state

    def train(self, data, target):
        if self.most == target:
            return 0
        self.seen[target] = self.seen.get(target, 0) + 1
        items = self.seen.items()
        items.sort(lambda a,b: b[1] - a[1])
        self.most = items[0][0]
        return 1

    def classify(self, line):
        return self.most


# vim: et sw=4 sts=4
