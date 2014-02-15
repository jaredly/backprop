#!/usr/bin/env python

def normalizers(data, meta):
    '''Colibrate normalizers for a data set'''
    ranges = {}
    # print data
    # print meta
    for name in meta.names()[:-1]:
        typ, rng = meta[name]
        if typ == 'numeric':
            mn = data[name].min()
            mx = data[name].max()
            ranges[name] = typ, (mn, mx - mn)
        else:
            ranges[name] = typ, data[name].unique()
    # print ranges
    return ranges

def denormalize(norms, data):
    '''Return from normalization'''
    data = data.copy()
    for name, (typ, more) in norms.items():
        if typ == 'numeric':
            mn, rng = more
            data[name] *= rng
            data[name] += mn
        else:
            total = len(more) - 1.0
            final = np.zeros(len(data[name]), str)
            for i, v in enumerate(more):
                num = i / total
                final[data[name] == num] = v
            data[name] = final
    return data

def normalize(norms, data):
    '''Normalize some data using the precalibrated normalizers'''
    data = data.copy()
    for name, (typ, more) in norms.items():
        if typ == 'numeric':
            mn, rng = more
            data[name] -= mn
            data[name] /= rng
        else:
            total = len(more) - 1.0
            final = np.zeros(len(data[name]), float)
            for i, v in enumerate(more):
                num = i / total
                final[data[name] == v] = num
            data[name] = final
    return data

def subData(data, truthy, falsy, find):
    '''Get a subset of the data aligning only with the "truthy" and "falsy"
    values (for multi-node perceptron learning)'''
    ci = list(data.columns).index(find)
    cols = data.columns[:ci] + data.columns[ci + 1:]
    # print truthy, falsy, find
    # print data
    tdata = data[(data[find] == truthy) + (data[find] == falsy)]
    targets = np.zeros(len(tdata[find]), int)
    # print tdata
    # print len(tdata[find]), 'len'
    # print targets
    targets[tdata[find] == truthy] = 1
    targets[tdata[find] == falsy] = 0
    tdata = tdata[cols]
    tdata[find] = targets
    return tdata


# vim: et sw=4 sts=4
