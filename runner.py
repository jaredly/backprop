#!/usr/bin/env python

from utils import normalizers, denormalize, normalize
import numpy as np

class Runner:
    def __init__(self, learner, meta, target=None, stop_after=100, max_iters=1000):
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

    def train(self, data, validate=None):
        '''data is already normalized'''
        history = []
        bests = [0]
        incols = list(data.columns)
        incols.remove(self.target)
        validation = None
        if validate:
            ln = len(data.index)
            vl = int(ln*validate)
            print 'Using', ln-vl, 'for training', vl, 'for validation'
            validation = data.loc[data.index[:vl]]
            data = data.loc[data.index[vl:]]
            # print 'Validation'
            # print validation
            # print 'Data'
            # print data[data.columns[-1]]
        for i in xrange(self.max_iters):
            print '.',
            # shuffle data
            ix = np.array(data.index)
            np.random.shuffle(ix)
            data = data.reindex(ix)
            # train learner
            error, weights = self.learner.epoch(data[incols], data[[self.target]])
            val = 0
            verr = 0
            if validation:
                val, verr, wrong = self.validate(validation)
                # print 'val, wrong', val, wrong
            # update best
            history.append([error, val, verr, weights])
            if validation:
                error = verr
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
                    print 'Done classifying; no progress in past %s epochs' % self.stop_after
                    # revert to the best weights
                    self.learner.use_state(self.best_state)
                    return history
        self.learner.use_state(self.best_state)
        print 'Reached max', self.max_iters
        return history

    def run(self, raw, split=None, validate=None):
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
        history = self.train(data.loc[data.index[:stop]], validate=validate)
        result = self.validate(data.loc[data.index[stop:]])
        return history, result

    def validate(self, norm):
        '''norm is already normalized. Returns % wrong'''
        total = len(norm.index)
        wrong = 0.0
        terr = 0
        for i in norm.index:
            line = norm.loc[i][:-1]
            # print line
            real = norm.loc[i][self.target]
            cln, err = self.learner.validate(line, real)
            terr += err
            # print cln
            if cln != real:
                # print 'Wrong!', cln, norm.loc[i][self.target]
                wrong += 1
        return wrong/total, terr / total, wrong




# vim: et sw=4 sts=4
