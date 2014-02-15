#!/usr/bin/env python

import pytest
from pandas import DataFrame, Series
from scipy.io.arff import loadarff

from learner import DefaultLearner as Learner

@pytest.fixture()
def ex1():
    data, meta = loadarff('test.arff')
    data = DataFrame(data)
    return data, meta

@pytest.fixture()
def simple(ex1):
    return Learner(ex1[1])

def test_learner(ex1, simple):
    assert simple.train(None, 'hat')
    assert simple.train(None, 'coat')
    assert simple.train(None, 'jacket')
    simple.train(None, 'coat')
    assert simple.classify(None) == 'coat'


# vim: et sw=4 sts=4
