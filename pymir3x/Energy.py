"""
energy.py
Compute energy and related quantities
Ported from https://github.com/jsawruk/pymir: 29 August 2017
"""

import numpy
from numpy.lib import stride_tricks


def energy(audio_data, window_size = 256):
    """
    Compute the energy of the given audio data, using the given windowSize
    """
    audio_length = len(audio_data)

    window = numpy.hamming(window_size)
    window.shape = (window_size, 1)

    n = audio_length - window_size  # number of windowed samples.

    # Create a view of audioData who's shape is (n, windowSize). Use stride_tricks such that each stide jumps only one item.
    p = numpy.power(audio_data, 2)
    s = stride_tricks.as_strided(p, shape=(n, window_size), strides=(audio_data.itemsize, audio_data.itemsize))
    e = numpy.dot(s, window) / window_size
    e.shape = (e.shape[0],)
    return e


def dEnergy(audio_data, window_size=256):
    """
    Compute the dEnergy differential term with windowing
    """
    e = energy(audio_data, window_size)
    diffE = numpy.diff(e)

    diffLength = len(diffE)

    window = numpy.hamming(window_size)
    window.shape = (window_size, 1)

    n = diffLength - window_size  # number of windowed samples.

    # Create a view of diffE who's shape is (n, windowSize). Use stride_tricks such that each stide jumps only one item.
    p = numpy.power(diffE, 2)
    s = stride_tricks.as_strided(p, shape=(n, window_size), strides=(diffE.itemsize, diffE.itemsize))
    dE = numpy.dot(s, window) / window_size
    dE.shape = (dE.shape[0],)
    return dE


def dLogEnergy(audio_data, window_size=256):
    """
    Compute d(log(Energy)) with windowing
    """
    e = energy(audio_data, window_size)
    logE = numpy.log(e)
    diffLogE = numpy.diff(logE)

    diffLogLength = len(diffLogE)

    window = numpy.hamming(window_size)
    window.shape = (window_size, 1)

    n = diffLogLength - window_size  # number of windowed samples.

    # Create a view of diffLogE who's shape is (n, windowSize). Use stride_tricks such that each stide jumps only one item.
    p = numpy.power(diffLogE, 2)
    s = stride_tricks.as_strided(p, shape=(n, window_size), strides=(diffLogE.itemsize, diffLogE.itemsize))
    dLogE = numpy.dot(s, window) / window_size
    dLogE.shape = (dLogE.shape[0],)
    return dLogE


def _test():
    import doctest
    doctest.testmod(verbose=True)
if __name__ == '__main__':
    _test()
