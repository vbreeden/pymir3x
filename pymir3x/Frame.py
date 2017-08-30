"""
Frame class
ndarray subclass for time-series data
Ported from https://github.com/jsawruk/pymir: 30 August 2017
"""

import matplotlib.pyplot as plt
import numpy
import pyaudio

from math import sqrt
from numpy.lib import stride_tricks
from pymir3x import Transforms, AudioFile


class Frame(numpy.ndarray):
    def __new__(cls, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = numpy.ndarray.__new__(cls, shape, dtype, buffer, offset, strides,
                                    order)

        obj.sampleRate = 0
        obj.channels = 1
        obj.format = pyaudio.paFloat32

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
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
        self.channels = getattr(obj, 'channels', None)
        self.format = getattr(obj, 'format', None)

        # We do not need to return anything

    #####################
    # Frame methods
    #####################

    # Compute the Constant Q Transform (CQT)
    def cqt(self):
        return Transforms.cqt(self)

    # Compute the Discrete Cosine Transform (DCT)
    def dct(self):
        return Transforms.dct(self)

    # Compute the energy of this frame
    def energy(self, window_size=256):
        frame_length = len(self)

        window = numpy.hamming(window_size)
        window.shape = (window_size, 1)

        n = frame_length - window_size  # number of windowed samples.

        # Create a view of signal who's shape is (n, windowSize).
        # Use stride_tricks such that each stride jumps only one item.
        p = numpy.power(self, 2)
        s = stride_tricks.as_strided(p, shape=(n, window_size),
                                     strides=(self.itemsize, self.itemsize))
        e = numpy.dot(s, window) / window_size
        e.shape = (e.shape[0],)
        return e

    # Decompose this frame into smaller frames of size frameSize
    def frames(self, frame_size, window_function=None):
        frames = []
        start = 0
        end = frame_size
        while start < len(self):

            if window_function is None:
                frames.append(self[start:end])
            else:
                window = window_function(frame_size)
                window.shape = (frame_size, 1)
                window = numpy.squeeze(window)
                frame = self[start:end]
                if len(frame) < len(window):
                    # Zero pad
                    frame_type = frame.__class__.__name__

                    sample_rate = frame.sampleRate
                    channels = frame.channels
                    frame_format = frame.format

                    diff = len(window) - len(frame)
                    frame = numpy.append(frame, [0] * diff)

                    if frame_type == "AudioFile":
                        frame = frame.view(AudioFile)
                    else:
                        frame = frame.view(Frame)

                    # Restore frame properties
                    frame.sampleRate = sample_rate
                    frame.channels = channels
                    frame.format = frame_format

                windowed_frame = frame * window
                frames.append(windowed_frame)

            start = start + frame_size
            end = end + frame_size

        return frames

    # Decompose into frames based on onset start time-series
    def framesFromOnsets(self, onsets):
        frames = []
        for i in range(0, len(onsets) - 1):
            frames.append(self[onsets[i]: onsets[i + 1]])

        return frames

    # Play this frame through the default playback device using pyaudio (PortAudio)
    # Note: This is a blocking operation.
    def play(self):
        # Create the stream
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format, channels=self.channels, rate=self.sampleRate,
                        output=True)

        # Write the audio data to the stream
        audio_data = self.tostring()
        stream.write(audio_data)

        # Close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

    # Plot the frame using matplotlib
    def plot(self):
        plt.plot(self)
        plt.xlim(0, len(self))
        plt.ylim(-1.5, 1.5)
        plt.show()

    # Compute the root-mean-squared amplitude
    def rms(self):
        frame_sum = 0
        for i in range(0, len(self)):
            frame_sum = frame_sum + self[i] ** 2

            frame_sum = frame_sum / (1.0 * len(self))

        return sqrt(frame_sum)

    # Compute the spectrum using an FFT. Returns an instance of Spectrum
    def spectrum(self):
        return Transforms.fft(self)

    # Compute the Zero-crossing rate (ZCR)
    def zcr(self):
        zcr = 0
        for i in range(1, len(self)):
            if (self[i - 1] * self[i]) < 0:
                zcr = zcr + 1

        return zcr / (1.0 * len(self))
