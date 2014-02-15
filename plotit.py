#!/usr/bin/env python
from itertools import cycle
from IPython.display import display

from runner import Runner
from perceptron import MultiPerceptron, fromArff as pFromArff

def plot_weights(weights, labels):
    styles = cycle(('-', '--', '-.'))
    n = len(weights[0])
    ncolors = 7
    i = -1
    for i in range(0, n/ncolors):
        plot(weights[:,i*ncolors:(i+1)*ncolors], ls=styles.next())
    plot(weights[:,(i+1)*ncolors:], ls=styles.next())
    legend(list(labels) + ['Bias'], loc='upper left', bbox_to_anchor=(1.01,1))

def test(fname):
    m = pFromArff(fname, .1)
    a = m.trainUp()
    print 'Done Training'
    # Plot the accuracies over epochs
    ps = m.perceptrons.items()
    for i in range(len(a[0][1])):
        figure()
        plot_weights(array([x[1][i] for x in a]), m.data.columns[:-1])
        title(ps[i][0][0] + ' vs ' + ps[i][0][1])
        xlabel('Epoch')
        ylabel('Weight')

    if len(a[0][0]) > 1:
        figure()
        plot(array([x[0] for x in a]))
        legend(list(l[0] + ' vs ' + l[1] for l, p in ps), loc='upper left', bbox_to_anchor=(1.01,1))
        ylim(-0.1, 1.1)
        title('Individual Perceptron Accuracy')
        xlabel('Epoch')
        ylabel('% Accuracy')

    figure()
    plot(array([array(x[0]).mean() for x in a]))
    title('Mean Accuracy')
    xlabel('Epoch')
    ylabel('% Accuracy')
    return m, a

def show_resuls(m, data, a):
    # Plot the accuracies over epochs
    ps = m.perceptrons.items()
    print 'Weights vs Epochs for each perceptron:'
    for i in range(len(a[0][1])):
        f = figure()
        plot_weights(array([x[1][i] for x in a]), data.columns[:-1])
        title(ps[i][0][0] + ' vs ' + ps[i][0][1])
        xlabel('Epoch')
        ylabel('Weight')
        display(f)

    print 'Accuracies:'
    if len(a[0][0]) > 1:
        f = figure()
        plot(array([x[0] for x in a]))
        legend(list(l[0] + ' vs ' + l[1] for l, p in ps), loc='upper left', bbox_to_anchor=(1.01,1))
        ylim(-0.1, 1.1)
        title('Individual Perceptron Accuracy')
        xlabel('Epoch')
        ylabel('% Accuracy')
        display(f)
    f = figure()
    plot(array([array(x[0]).mean() for x in a]))
    title('Mean Accuracy')
    xlabel('Epoch')
    ylabel('% Accuracy')
    display(f)

from pylab import *
#a = m.train(d, split=.7)
#a

def htmltitle(*args, **kwds):
    level = str(kwds.get('level', 2))
    message = ' '.join(map(str, args))
    display(HTML('<h' + level + '>' + message + '</h' + level + '>'))

def try_rates(meta, data, rates):
    allz = []
    for rate in rates:
        htmltitle('Learning Rate', rate)
        m = Main(MultiPerceptron(meta, rate), meta)
        results = m.run(data.copy(), split=None)
        print 'Best accuracy', m.best
        print 'Number of epochs:', len(results)
        show_resuls(m, data, results)
        allz.append(results)
    compare = DataFrame({
        "Rate": Series(rates),
        "Epochs": Series(tuple(len(results) for results in allz))
    }, columns=['Rate', 'Epochs'])
    htmltitle('Number of Epochs vs Learning Rate')
    display(compare)
    return allz
    
def trials(meta, data, num, split=.7):
    allz = []
    print 'Running', num, 'trials'
    accuracies = []
    for i in range(num):
        htmltitle('Trial', i+1)
        m = Main(MultiPerceptron(meta, .1), meta)
        results, accuracy = m.run(data.copy(), split=split)
        print 'Best accuracy', m.best
        htmltitle('Percent missed (of the test set):', accuracy[0], '; ', accuracy[1], ' instances', level=4)
        accuracies.append(accuracy[0])
        show_resuls(m, data, results)
        allz.append(results)
    htmltitle('Final Results')
    print 'Mean accuracy over', num, 'runs:', 1 - sum(accuracies)/float(num)
    return allz

def runthis(fname, split=.7):
    data, meta = loadarff(fname)
    print 'Running perceptron classification on', fname, 'with a testing split of', split
    
    #figsize(12, 6)
    rcParams['figure.figsize'] = (6, 3)
    return trials(meta, DataFrame(data), 5, split)
# vim: et sw=4 sts=4
