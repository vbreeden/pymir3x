"""
Transforms for converting between time and spectral domains
Includes: FFT/IFFT, DCT/IDCT, CQT
Ported from https://github.com/jsawruk/pymir: 29 August 2017
"""

import numpy
import numpy.fft
import scipy.fftpack
import pymir3x

from numpy import array, cos, pi, sqrt, zeros


# Fourier Transforms
def fft(frame):
    """
    Compute the spectrum using an FFT.
    Returns an instance of the spectrum.
    """
    fft_data = numpy.fft.rfft(frame)  # rfft only returns the real half of the FFT values, which is all we need.
    spectrum = fft_data.view(pymir3x.Spectrum)
    spectrum.sampleRate = frame.sampleRate
    return spectrum


# Inverse Fourier Transform
def ifft(spectrum):
    fft_data = numpy.fft.irfft(spectrum)
    frame = fft_data.view(pymir3x.Frame)
    frame.sampleRate = spectrum.sampleRate
    return frame


# Discrete Cosine Transform (DCT)
def dct(frame):
    dct_result = scipy.fftpack.dct(frame, type=2, norm='ortho')
    dct_spectrum = dct_result.view(pymir3x.Spectrum)
    dct_spectrum.sampleRate = frame.sampleRate
    return dct_spectrum


# Inverse Discrete Cosine Transform (IDCT)
def idct(spectrum):
    idct_result = scipy.fftpack.idct(spectrum, type=2, norm='ortho')
    idct_frame = idct_result.view(pymir3x.Frame)
    idct_frame.sampleRate = spectrum.sampleRate
    return idct_frame


# Constant Q Transform
def cqt(frame):
    frame_length = len(frame)
    y = array(zeros(frame_length))
    a = sqrt(2 / float(frame_length))
    for k in range(frame_length):
        for n in range(frame_length):
            y[k] += frame[n] * cos(pi * (2 * n + 1) * k / float(2 * frame_length))

            if k == 0:
                y[k] = y[k] * sqrt(1 / float(frame_length))
            else:
                y[k] = y[k] * a

    return y
