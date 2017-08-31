"""
Compute onset times from time-domain audio data
Spectra are computed as necessary
Supported methods:
- Time-domain: energy
- Spectral: flux

Ported from https://github.com/jsawruk/pymir: 30 August 2017
"""

import numpy

from pymir3x import Energy, SpectralFlux


def onsets(audio_data, method='energy'):
    audio_onsets = []
    if method == 'energy':
        audio_onsets = onsetsByEnergy(audio_data)
    elif method == 'flux':
        audio_onsets = onsetsByFlux(audio_data)

    return audio_onsets


# Compute onsets by using dEnergy (time-domain)
def onsetsByEnergy(audio_data, frame_size=512):
    dE = Energy.dEnergy(audio_data, frame_size)
    peaks = peakPicking(dE, window_size=2048)

    return peaks


# Compute onsets by using spectral flux
def onsetsByFlux(audio_data, frame_size=1024):
    frames = audio_data.frames(frame_size)

    # Compute the spectra of each frame
    spectra = [f.spectrum() for f in frames]

    # Compute the spectral flux
    flux = SpectralFlux.spectralFlux(spectra, rectify=True)

    peaks = peakPicking(flux, window_size=10)
    peaks = [frame_size * p for p in peaks]

    return peaks


def peakPicking(audio_onsets, window_size=1024):
    peaks = peaksAboveAverage(audio_onsets, window_size)
    return peaks


# Find peaks by the following method:
# - Compute the average of all the data
# - Using a non-sliding window, find the max within each window
# - If the windowed max is above the average, add it to peaks
def peaksAboveAverage(data, window_size):
    data = numpy.array(data)

    peaks = []

    data_average = numpy.average(data)
    data_average = data_average * 1

    slide_amount = window_size / 2

    start = 0
    end = window_size
    while start < len(data):
        # print "Start: " + str(start)
        # print "End:   " + str(end)
        window_max = data[start:end].max()
        window_max_pos = data[start:end].argmax()

        if window_max > data_average:
            if (start + window_max_pos) not in peaks:
                peaks.append(start + window_max_pos)

        start = start + slide_amount
        end = end + slide_amount

    return peaks

# Reserved for future implementation:
# import sys
# from numpy import Inf, NaN, arange, asarray, isscalar
#
# def peakPicking(onsets, windowSize=1024, threshold=1.0):
#     peaks = []
#
#     peaks = peaksAboveAverage(onsets, windowSize)
#
#     # Compute a windowed (moving) average
#     # movingAverage = windowedAverage(onsets, windowSize)
#
#     # peaks = peakdet(movingAverage, 1, threshold = threshold)
#
#     # for i in range(0, len(movingAverage) - 1):
#     #	if movingAverage[i] > movingAverage[i + 1]:
#     #		peaks.append(movingAverage[i])
#     #	else:
#     #		peaks.append(0)
#     return peaks
#
# def windowedAverage(data, window_size):
#     window = numpy.repeat(1.0, window_size) / window_size
#     return numpy.convolve(data, window)[window_size - 1: -(window_size - 1)]
#
#
# def peakdet(v, delta, x=None, threshold=1):
#     """
#     Adapted from code at: https://gist.github.com/250860
#     Converted from MATLAB script at http://billauer.co.il/peakdet.html
#
#     Returns two arrays
#
#     function [maxtab, mintab]=peakdet(v, delta, x)
#     %PEAKDET Detect peaks in a vector
#     %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
#     %        maxima and minima ("peaks") in the vector V.
#     %        MAXTAB and MINTAB consists of two columns. Column 1
#     %        contains indices in V, and column 2 the found values.
#     %
#     %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
#     %        in MAXTAB and MINTAB are replaced with the corresponding
#     %        X-values.
#     %
#     %        A point is considered a maximum peak if it has the maximal
#     %        value, and was preceded (to the left) by a value lower by
#     %        DELTA.
#
#     % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
#     % This function is released to the public domain; Any use is allowed.
#
#     """
#     maxtab = []
#
#     if x is None:
#         x = arange(len(v))
#
#     v = asarray(v)
#
#     if len(v) != len(x):
#         sys.exit('Input vectors v and x must have same length')
#
#     if not isscalar(delta):
#         sys.exit('Input argument delta must be a scalar')
#
#     if delta <= 0:
#         sys.exit('Input argument delta must be positive')
#
#     mn, mx = Inf, -Inf
#     mnpos, mxpos = NaN, NaN
#
#     lookformax = True
#
#     for i in arange(len(v)):
#         this = v[i]
#         if this > mx:
#             mx = this
#             mxpos = x[i]
#         if this < mn:
#             mn = this
#             mnpos = x[i]
#
#         if lookformax:
#             if this < mx - delta and this > threshold:
#                 # maxtab.append((mxpos, mx))
#                 maxtab.append(mxpos)
#                 mn = this
#                 mnpos = x[i]
#                 lookformax = False
#         else:
#             if this > mn + delta:
#                 # mintab.append((mnpos, mn))
#                 mx = this
#                 mxpos = x[i]
#                 lookformax = True
#
#     # return array(maxtab)
#     return maxtab
