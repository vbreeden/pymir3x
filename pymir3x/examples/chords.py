"""
chords.py
Chord estimator from MP3 file
Ported from https://github.com/jsawruk/pymir: 31 August 2017
"""

import sys

from pymir3x import AudioFile, Pitch, Onsets

import matplotlib.pyplot as plt

sys.path.append('..')

# Load the audio
print("Loading Audio")
audio_file = AudioFile.open("../audio_files/test-stereo.mp3")

plt.plot(audio_file)
plt.show()

print("Finding onsets using Spectral Flux (spectral domain)")
o = Onsets.onsetsByFlux(audio_file)
print(o)

print("Extracting Frames")
frames = audio_file.framesFromOnsets(o)

print("Start | End  | Chord | (% match)")
print("-------------------------------")

frameIndex = 0
startIndex = 0
for frame in frames:
    spectrum = frame.spectrum()
    chroma = spectrum.chroma()
    print(chroma)

    chord, score = Pitch.getChord(chroma)

    endIndex = startIndex + len(frame)

    startTime = startIndex / frame.sampleRate
    endTime = endIndex / frame.sampleRate

    print("%.2f  | %.2f | %-4s | (%.2f)" % (startTime, endTime, chord, score))

    frameIndex = frameIndex + 1
    startIndex = startIndex + len(frame)
