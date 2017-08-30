"""
Spectrum class
ndarray subclass for spectral data
Ported from https://github.com/jsawruk/pymir: 29 August 2017
"""

import matplotlib.pyplot as plt
import numpy
import scipy.stats.mstats
from math import sqrt
from numpy import abs
from pymir3x import MFCC, Pitch, Transforms


class Spectrum(numpy.ndarray):
    def __new__(cls, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = numpy.ndarray.__new__(cls, shape, dtype, buffer, offset, strides,
                                    order)

        obj.sampleRate = 0

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have gotten to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None:
            return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).

        self.sampleRate = getattr(obj, 'sampleRate', None)

    #####################
    # Spectrum methods
    #####################

    # Compute the spectral centroid. Characterizes the "center of gravity" of the spectrum.
    # Approximately related to timbral "brightness"
    def centroid(self):
        bin_number = 0

        numerator = 0
        denominator = 0

        for nth_bin in self:    # 'nth_bin' is used to prevent shadowing of the 'bin' keyword
            # Compute center frequency
            f = (self.sampleRate / 2.0) / len(self)
            f = f * bin_number

            numerator = numerator + (f * abs(nth_bin))
            denominator = denominator + abs(nth_bin)

            bin_number = bin_number + 1

        return (numerator * 1.0) / denominator

    # Compute the 12-ET chroma vector from this spectrum
    def chroma(self):
        return Pitch.chroma(self)

    # Compute the spectral crest factor, i.e. the ratio of the maximum of the spectrum to the
    # sum of the spectrum
    def crest(self):
        abs_spectrum = abs(self)
        spectral_sum = numpy.sum(abs_spectrum)

        max_frequency_index = numpy.argmax(abs_spectrum)
        max_spectrum = abs_spectrum[max_frequency_index]

        return max_spectrum / spectral_sum

    # Compute the spectral flatness (ratio between geometric and arithmetic means)
    def flatness(self):
        geometric_mean = scipy.stats.mstats.gmean(abs(self))
        arithmetic_mean = self.mean()

        return geometric_mean / arithmetic_mean

    # Compute the Inverse Discrete Cosine Transform (IDCT)
    def idct(self):
        return Transforms.idct(self)

    # Compute the Inverse FFT
    def ifft(self):
        return Transforms.ifft(self)

    # Compute the Mth Mel-Frequency Cepstral Coefficient
    def mfcc(self, m, num_filters=48):
        return MFCC.mfcc(self, m, num_filters)

    # Vectorized MFCC implementation
    def mfcc2(self, num_filters=32):
        return MFCC.mfcc2(self, num_filters)

    # Plot the spectrum using matplotlib
    def plot(self):
        plt.plot(abs(self))
        plt.xlim(0, len(self))
        plt.show()

    # Determine the spectral rolloff, i.e. the frequency below which 85% of the spectrum's
    # energy is located.
    def rolloff(self):
        abs_spectrum = abs(self)
        spectral_sum = numpy.sum(abs_spectrum)

        rolloff_sum = 0
        rolloff_index = 0
        for i in range(0, len(self)):
            rolloff_sum = rolloff_sum + abs_spectrum[i]
            if rolloff_sum > (0.85 * spectral_sum):
                rolloff_index = i
                break

        # Convert the index into a frequency
        frequency = rolloff_index * (self.sampleRate / 2.0) / len(self)
        return frequency

    # Compute the spectral spread
    # (basically a variance of the spectrum around the spectral centroid)
    def spread(self):
        centroid = self.centroid()

        bin_number = 0

        numerator = 0
        denominator = 0

        for nth_bin in self:
            # Compute center frequency
            f = (self.sampleRate / 2.0) / len(self)
            f = f * bin_number

            numerator = numerator + (((f - centroid) ** 2) * abs(nth_bin))
            denominator = denominator + abs(nth_bin)

            bin_number = bin_number + 1

        return sqrt((numerator * 1.0) / denominator)

    # Compute the spectral mean (first spectral moment)
    def spectral_mean(self):
        return numpy.sum(abs(self)) / len(self)

    # Compute the spectral variance (second spectral moment)
    def variance(self):
        return numpy.var(abs(self))

    # Compute the spectral skewness (third spectral moment)
    def skewness(self):
        return scipy.stats.skew(abs(self))

    # Compute the spectral kurtosis (fourth spectral moment)
    def kurtosis(self):
        return scipy.stats.kurtosis(abs(self))
