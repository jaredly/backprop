#!/usr/bin/env python

from runner import Runner
from learner import Learner

from utils import subData

class Perceptron:
    def __init__(self, size, rate=.1):
        self.weights = np.zeros(size)
        self.trainingRate = rate

    def classify(self, line):
        line = np.concatenate((line, [1]))
        net = self.weights.dot(line)
        output = 1 if net > 0 else 0
        return output, net

    def train(self, line, learn=True):
        # history?
        values = line.copy()
        values[-1] = 1
        net = self.weights.dot(values)
        output = 1 if net > 0 else 0
        target = line[-1]
        if output == target:
            return 1, net
        if not learn:
            return 0, net
        self.learn(values, target, output)
        return 0, net

    def learn(self, values, target, output):
        self.weights += self.trainingRate * (target - output) * values

class MultiPerceptron(Learner):
    def __init__(self, meta, rate, target=None):
        Learner.__init__(self, meta, target)

        length = len(meta.names())
        _, possible = meta[self.target]

        self.perceptrons = {}
        for truthy, falsy in itertools.combinations(possible, 2):
            self.perceptrons[(truthy, falsy)] = Perceptron(length, rate)

    def state(self):
        return [p.weights.copy() for (_, _), p in self.perceptrons.items())]

    def use_best(self):
        for w, (k, p) in zip(self.best_state, self.perceptrons.items()):
            p.weights = w

    def epoch(self, data):
        accuracy = []
        weights = []
        for (truthy, falsy), p in self.perceptrons.items():
            tdata = subData(data, truthy, falsy, self.target)
            dlen = tdata.shape[0]
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
            self.best_state = weights
        return accuracy, weights

    def classify(self, line):
        votes = {}
        for (truthy, falsy), p in self.perceptrons.items():
            res, confidence = p.classify(line)
            # print res, confidence, truthy, falsy, p.weights, norm.loc[i]
            vote = truthy if res else falsy
            if vote not in votes:
                votes[vote] = [0, 0]
            votes[vote][0] += 1
            votes[vote][1] += abs(confidence)
        most = sorted(votes.items(), (lambda (ka, va), (kb, vb): vb[0] - va[0]))
        return most[0][0]

def fromArff(fname, rate):
    data, meta = loadarff(fname)
    data = DataFrame(data)
    t = meta.names()[:-1]
    p = MultiPerceptron(meta, rate, t)
    return Runner(p, meta, t)

# vim: et sw=4 sts=4
