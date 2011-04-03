

import itertools
import numpy
def surround_iter(x,y,shift,xmin,ymin,xmax,ymax):
    if shift==0:
        yield [(x,y)]
    xx = max(x-shift,xmin) 
    yy = max(y-shift,ymin)
    while xx <= min(x+shift, xmax):
        yield (xx, yy)
        xx += 1
    xx -= 1
    yy += 1
    while yy <= min(y+shift, ymax):
        yield (xx, yy)
        yy += 1
    yy -= 1
    xx -= 1
    while xx >= max(x-shift, xmin):
        yield (xx, yy)
        xx -= 1
    xx += 1
    yy -= 1
    while yy > max(y-shift, ymin):
        yield (xx, yy)
        yy -= 1

def nonzero_indices(ndarray):
    return numpy.transpose(numpy.nonzero(numpy.asarray(ndarray)))


def all_true(boolfunc, iterator):
    for b in iterator:
        if not boolfunc(*b):
            return False
    return True

def all_false(boolfunc, iterator):
    for b in iterator:
        if boolfunc(*b):
            return False
    return True

def false_ifilter(boolfunc, iterator):
    for b in iterator:
        if not boolfunc(*b):
            yield b

def true_ifilter(boolfunc, iterator):
    for b in iterator:
        if boolfunc(*b):
            yield b

