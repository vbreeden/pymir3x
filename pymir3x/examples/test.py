"""
Tests of different onset detection methods
Currently under development
Ported from https://github.com/jsawruk/pymir: 31 August 2017
"""
import numpy
import sys

from pymir3x import AudioFile

sys.path.append('..')

filename = "../audio_files/drum_loop_01.wav"

print("Opening File: " + filename)
audio_file = AudioFile.open(filename)

frames = audio_file.frames(2048, numpy.hamming)

print(len(frames))
