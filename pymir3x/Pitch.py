"""
Pitch functions
- Chroma
Ported from https://github.com/jsawruk/pymir: 29 August 2017
"""

import math
import numpy


# Dictionary of major and minor chords
chords = [{'name': "C", 'vector': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], 'key': 0, 'mode': 1},
          {'name': "Cm", 'vector': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], 'key': 0, 'mode': 0},
          {'name': "C#", 'vector': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], 'key': 1, 'mode': 1},
          {'name': "C#m", 'vector': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 'key': 1, 'mode': 0},
          {'name': "D", 'vector': [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],  'key': 2, 'mode': 1},
          {'name': "Dm", 'vector': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],  'key': 2, 'mode': 0},
          {'name': "Eb", 'vector': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],  'key': 3, 'mode': 1},
          {'name': "Ebm", 'vector': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],  'key': 3, 'mode': 0},
          {'name': "E", 'vector': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  'key': 4, 'mode': 1},
          {'name': "Em", 'vector': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],  'key': 4, 'mode': 0},
          {'name': "F", 'vector': [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  'key': 5, 'mode': 1},
          {'name': "Fm", 'vector': [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  'key': 5, 'mode': 0},
          {'name': "F#", 'vector': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  'key': 6, 'mode': 1},
          {'name': "F#m", 'vector': [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  'key': 6, 'mode': 0},
          {'name': "G", 'vector': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  'key': 7, 'mode': 1},
          {'name': "Gm", 'vector': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],  'key': 7, 'mode': 0},
          {'name': "Ab", 'vector': [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  'key': 8, 'mode': 1},
          {'name': "Abm", 'vector': [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],  'key': 8, 'mode': 0},
          {'name': "A", 'vector': [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  'key': 9, 'mode': 1},
          {'name': "Am", 'vector': [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  'key': 9, 'mode': 0},
          {'name': "Bb", 'vector': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],  'key': 10, 'mode': 1},
          {'name': "Bbm", 'vector': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  'key': 10, 'mode': 0},
          {'name': "B", 'vector': [0,  0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],  'key': 11, 'mode': 1},
          {'name': "Bm", 'vector': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],  'key': 11, 'mode': 0}
          ]


# Compute the 12-ET chroma vector from this spectrum
def chroma(spectrum):
    chroma_vector = [0] * 12
    for index in range(0, len(spectrum)):

        # Assign a frequency value to each bin
        f = index * (spectrum.sampleRate / 2.0) / len(spectrum)

        # Convert frequency to pitch to pitch class
        if f != 0:
            pitch = frequencyToMidi(f)
        else:
            pitch = 0
        pitch_class = pitch % 12

        chroma_vector[pitch_class] = chroma_vector[pitch_class] + abs(spectrum[index])

    # Normalize the chroma vector
    max_element = max(chroma_vector)
    chroma_vector = [c / max_element for c in chroma_vector]

    return chroma_vector


# Compute the similarity between two vectors using the cosine similarity metric
def cosineSimilarity(a, b):
    dot_product = 0
    a_magnitude = 0
    b_magnitude = 0
    for i in range(len(a)):
        dot_product += (a[i] * b[i])
        a_magnitude += math.pow(a[i], 2)
        b_magnitude += math.pow(b[i], 2)

        a_magnitude = math.sqrt(a_magnitude)
        b_magnitude = math.sqrt(b_magnitude)

    return dot_product / (a_magnitude * b_magnitude)


# Convert a given frequency in Hertz to its corresponding MIDI pitch number (60 = Middle C)
def frequencyToMidi(frequency):
    return int(round(69 + 12 * math.log(frequency / 440.0, 2)))


# Given a chroma vector, return the best chord match using naive dictionary-based method
def getChord(chroma):
    max_score = 0
    chord_name = ""
    for chord in chords:
        score = cosineSimilarity(chroma, chord['vector'])
        if score > max_score:
            max_score = score
            chord_name = chord['name']

    return chord_name, max_score


# Compute the pitch by using the naive pitch estimation method, i.e. get the pitch name for the most prominent frequency.
# Only returns MIDI pitch number
def naivePitch(spectrum):
    max_frequency_index = numpy.argmax(spectrum)
    max_frequency = max_frequency_index * (spectrum.sampleRate / 2.0) / len(spectrum)
    return frequencyToMidi(max_frequency)
