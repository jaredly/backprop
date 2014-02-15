#!/usr/bin/env python

from utils import normalizers, denormalize, normalize
import numpy as np

class Runner:
    def __init__(self, learner, meta, target=None, stop_after=20, max_iters=1000):
        if target is None:
            target = meta.names()[-1]
        self.max_iters = max_iters
        self.stop_after = stop_after
        self.target = target
        self.norm = None
        self.meta = meta
        self.learner = learner

        self.best = None
        self.best_state = None

    def train(self, data):
        '''data is already normalized'''
        history = []
        bests = [0]
        incols = list(data.columns)
        incols.remove(self.target)
        for i in xrange(self.max_iters):
            print '.',
            # shuffle data
            ix = np.array(data.index)
            np.random.shuffle(ix)
            data = data.reindex(ix)
            # train learner
            error, weights = self.learner.epoch(data[incols], data[[self.target]])
            # update best
            history.append([error, weights])
            if self.best is None or error < self.best:
                self.best = error
                self.best_state = weights
            if self.best == 0:
                print 'Fully trained'
                return history
            bests.append(self.best)
            # check for stopping
            if len(bests) > self.stop_after:
                if bests.pop(0) == self.best:
                    print 'Done classifying; no progress in past 20 epochs'
                    # revert to the best weights
                    self.learner.use_state(self.best_state)
                    return history
        return history

    def run(self, raw, split=None):
        self.norm = normalizers(raw, self.meta)
        data = normalize(self.norm, raw)

        if not split:
            return self.train(data)

        ix = np.array(data.index)
        np.random.shuffle(ix)
        data = data.reindex(ix)

        ln = len(data.index)
        stop = int(ln * split)
        print 'Using', stop, 'for training, and', ln - stop, 'for testing'
        history = self.train(data.loc[data.index[:stop]])
        result = self.validate(data.loc[data.index[stop:]])
        return history, result

    def validate(self, norm):
        '''norm is already normalized'''
        total = len(norm.index)
        wrong = 0.0
        for i in norm.index:
            cln = self.learner.classify(norm.loc[i][:-1])
            # print most
            if cln != norm.loc[i][self.target]:
                # print 'Wrong!'
                wrong += 1
        return wrong/total, wrong




"""
class Main(BaseRunner)::
    '''Perceptron tester

    This will train a perceptron (or mutliple, if there are more than two
    possible outputs) given a dataset and meta information, assumed to come
    from an .arff file.

    When training, if no progress is made in 20 epochs it will quit.

    '''

    def __init__(self, meta, rate=.1, target=None):
        BaseRunner.__init__(self, meta, target)

        self.learner = MultiPerceptron(meta, rate, target)

    def train_perceptrons(self, data):
        accuracy = []
        weights = []
        for (truthy, falsy), p in self.perceptrons.items():
            tdata = subData(data, truthy, falsy, self.find)
            dlen = tdata.shape[0]
            # print tdata
            # print tdata.index
            results = []
            for i in tdata.index:
                results.append(p.train(tdata.loc[i])[0])
            # print results
            accuracy.append(sum(results)/float(len(results)))
            weights.append(p.weights.copy())
        total = sum(accuracy)/float(len(accuracy))
        if total > self.best:
            self.best = total
            self.best_weights = weights
        return accuracy, weights

    def validate(self, norm):
        '''norm is already normalized'''
        total = len(norm.index)
        wrong = 0.0
        for i in norm.index:
            votes = {}
            for (truthy, falsy), p in self.perceptrons.items():
                res, confidence = p.classify(norm.loc[i][:-1])
                # print res, confidence, truthy, falsy, p.weights, norm.loc[i]
                vote = truthy if res else falsy
                if vote not in votes:
                    votes[vote] = [0, 0]
                votes[vote][0] += 1
                votes[vote][1] += abs(confidence)
            most = sorted(votes.items(), (lambda (ka, va), (kb, vb): vb[0] - va[0]))
            # print most
            if most[0][0] != norm.loc[i][self.find]:
                # print 'Wrong!'
                wrong += 1
        return wrong/total, wrong

    def trainUp(self, data):
        history = []
        bests = [0]
        for i in range(100):
            # shuffle data
            ix = np.array(data.index)
            np.random.shuffle(ix)
            data = data.reindex(ix)
            # train perceptrons
            history.append(self.train_perceptrons(data))
            if self.best == 1:
                print 'Fully trained'
                return history
            bests.append(self.best)
            if len(bests) > 20:
                if bests.pop(0) == self.best:
                    print 'Done classifying; no progress in past 20 epochs'
                    # revert to the best weights
                    for w, (k, p) in zip(self.best_weights, self.perceptrons.items()):
                        p.weights = w
                    return history

    def train(self, raw, split=None):
        self.norm = normalizers(raw, self.meta)
        data = normalize(self.norm, raw)

        if not split:
            return self.trainUp(data)

        ix = np.array(data.index)
        np.random.shuffle(ix)
        data = data.reindex(ix)

        ln = len(data.index)
        stop = int(ln * split)
        print 'Using', stop, 'for training, and', ln - stop, 'for testing'
        history = self.trainUp(data.loc[data.index[:stop]])
        result = self.validate(data.loc[data.index[stop:]])
        return history, result
"""


# vim: et sw=4 sts=4
