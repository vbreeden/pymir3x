"""
MFCC methods
Compute Mel-Frequency Cepstral Coefficients
Ported from https://github.com/jsawruk/pymir: 29 August 2017
"""

import math
import numpy
import scipy
from numpy import log


def mfcc2(spectrum, num_filters=32):
    """
    Alternative (and vectorized) MFCC computation from Steve Tjoa
    """
    fb = filterbank(spectrum, spectrum.sampleRate, num_filters)
    coeff = scipy.fftpack.dct(scipy.log(fb), type=2, norm='ortho')
    return coeff


def filterbank(x, fs, num_filters):
    m = 2 ** (1.0 / 6)
    f2 = 110.0
    f1 = f2 / m
    f3 = f2 * m
    fb = scipy.array(scipy.zeros(num_filters))
    for i in range(num_filters):
        fb[i] = numpy.absolute(fbwin(x, fs, f1, f2, f3))
        f1 = f2
        f2 = f3
        f3 = f3 * m

    return fb


def fbwin(x, fs, f1, f2, f3):
    n = len(x)
    b1 = int(n * f1 / fs)
    b2 = int(n * f2 / fs)
    b3 = int(n * f3 / fs)
    y = x[b2]

    for b in range(b1, b2):
        y = y + x[b] * (b - b1) / (b2 - b1)

    for b in range(b2 + 1, b3):
        y = y + x[b] * (1 - (b - b2) / (b3 - b2))

    return y


def mfcc(spectrum, m, num_filters=48):
    """
    Compute the Mth Mel-Frequency Cepstral Coefficient
    """
    outer_sum = 0

    bin_size = len(spectrum)

    if m >= num_filters:
        return 0  # This represents an error condition - the specified coefficient is greater than or equal to the number of filters. The behavior in this case is undefined.

    result = normalizationFactor(num_filters, m)

    for filter_band in range(1, num_filters + 1):
        # Compute inner sum
        inner_sum = 0
        for frequency_band in range(0, bin_size - 1):
            inner_sum = inner_sum + abs(spectrum[frequency_band] * filterParameter(bin_size, frequency_band, filter_band, spectrum.sampleRate))

        if inner_sum > 0:
            inner_sum = log(inner_sum)  # The log of 0 is undefined, so don't use it

        inner_sum = inner_sum * math.cos(((m * math.pi) / num_filters) * (filter_band - 0.5))

        outer_sum = outer_sum + inner_sum

    result = result * outer_sum

    return result


def normalizationFactor(num_filters, m):
    """
    Intermediate computation used by mfcc function.
    Computes a normalization factor
    """
    if m == 0:
        normalization_factor = math.sqrt(1.0 / num_filters)
    else:
        normalization_factor = math.sqrt(2.0 / num_filters)

    return normalization_factor


def filterParameter(bin_size, frequency_band, filter_band, sampling_rate):
    """
    Intermediate computation used by the mfcc function.
    Compute the filter parameter for the specified frequency and filter bands
    """
    filter_parameter = 0
    boundary = (frequency_band * sampling_rate) / float(bin_size)  # k * Fs / N
    prev_center_frequency = getCenterFrequency(filter_band - 1)  # fc(l - 1)
    this_center_frequency = getCenterFrequency(filter_band)  # fc(l)
    next_center_frequency = getCenterFrequency(filter_band + 1)  # fc(l + 1)

    if boundary >= 0 and boundary < prev_center_frequency:
        filter_parameter = 0

    elif boundary >= prev_center_frequency and boundary < this_center_frequency:
        filter_parameter = (boundary - prev_center_frequency) / (this_center_frequency - prev_center_frequency)
        filter_parameter = filter_parameter * getMagnitudeFactor(filter_band)

    elif boundary >= this_center_frequency and boundary < next_center_frequency:
        filter_parameter = (boundary - next_center_frequency) / (this_center_frequency - next_center_frequency)
        filter_parameter = filter_parameter * getMagnitudeFactor(filter_band)

    elif boundary >= next_center_frequency and boundary < sampling_rate:
        filter_parameter = 0

    return filter_parameter


def getMagnitudeFactor(filter_band):
    """
    Intermediate computation used by the mfcc function.
    Compute the band-dependent magnitude factor for the given filter band
    """
    magnitude_factor = 0

    if filter_band >= 1 and filter_band <= 14:
        magnitude_factor = 0.015
    elif filter_band >= 15 and filter_band <= 48:
        magnitude_factor = 2.0 / (getCenterFrequency(filter_band + 1) - getCenterFrequency(filter_band - 1))

    return magnitude_factor


def getCenterFrequency(filter_band):
    """
    Intermediate computation used by the mfcc function.
    Compute the center frequency (fc) of the specified filter band (l)
    This where the mel-frequency scaling occurs. Filters are specified so that their
    center frequencies are equally spaced on the mel scale
    """
    if filter_band == 0:
        center_frequency = 0
    elif filter_band >= 1 and filter_band <= 14:
        center_frequency = (200.0 * filter_band) / 3.0
    else:
        exponent = filter_band - 14
        center_frequency = math.pow(1.0711703, exponent)
        center_frequency = center_frequency * 1073.4

    return center_frequency
