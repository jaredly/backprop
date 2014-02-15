#!/usr/bin/env python

class Learner:
    def __init__(self, meta, target=None):
        if target is None:
            target = meta.names()[-1]
        self.best = 0
        self.best_state = None
        self.target = target
        self.meta = meta
        self.init_state()

    # OVERRIDE THESE
    def init_state(self):
        self.seen = {}
        self.most = None
    
    def state(self):
        '''get current state'''
        return self.seen.copy()

    def use_best(self):
        '''set state to the best so far'''
        self.seen = self.best_state

    def train(self, data, target):
        '''test a line, return True/False for right/wrong'''
        if self.most == target:
            return True
        self.seen[target] = self.seen.get(target, 0) + 1
        items = self.seen.items()
        items.sort(lambda a,b: b[1] - a[1])
        self.most = items[0][0]

    def classify(self, line):
        return self.most

    # don't need to override
    def epoch(self, data, targets):
        '''run an epoch on the data. Calleds "train(line)"'''
        results = []
        for i in tdata.index:
            results.append(1 if self.train(tdata.loc[i], targets.loc[i]) else 0)
        accuracy = sum(results)/float(len(results))
        weights = self.state()
        if accuracy > self.best:
            self.best = accuracy
            self.best_state = weights
        return accuracy, weights



# vim: et sw=4 sts=4
