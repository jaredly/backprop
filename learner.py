#!/usr/bin/env python

class Learner:
    def __init__(self, meta, target=None):
        if target is None:
            target = meta.names()[-1]
        self.best = 0
        self.best_state = None
        self.target = target
        self.meta = meta
    
    def state(self):
        raise Exception('Override')

    def use_best(self):
        raise Exception('Override')

    def train(self, data):
        '''test a line, return 1/0 for right/wrong'''
        raise Exception('Override')

    def epoch(self, data):
        results = []
        for i in tdata.index:
            results.append(self.train(tdata.loc[i]))
        accuracy = sum(results)/float(len(results))
        weights = self.state()
        if accuracy > self.best:
            self.best = accuracy
            self.best_state = weights
        return accuracy, weights

    def classify(self, line):
        raise Exception('Override')



# vim: et sw=4 sts=4
