"""
Tests of different onset detection methods
Currently under development
Ported from https://github.com/jsawruk/pymir: 31 August 2017
"""
import sys
import matplotlib.pyplot as plt

from pymir3x import AudioFile, Onsets

sys.path.append('..')


filename = "../audio_files/drum_loop_01.wav"

print("Opening File: " + filename)
audio_file = AudioFile.open(filename)

plt.plot(audio_file)
plt.show()

# Time-based methods
print("Finding onsets using Energy function (temporal domain)")
o = Onsets.onsetsByEnergy(audio_file)
print(o)
frames = audio_file.framesFromOnsets(o)

for i in range(0, len(frames)):
    print("Frame " + str(i))
    plt.plot(frames[i])
    plt.show()

# Spectral-based methods
print("Finding onsets using Spectral Flux (spectral domain)")
o = Onsets.onsetsByFlux(audio_file)
print(o)
frames = audio_file.framesFromOnsets(o)
for i in range(0, len(frames)):
    print("Frame " + str(i))
    plt.plot(frames[i])
    plt.show()
    frames[i].play()
