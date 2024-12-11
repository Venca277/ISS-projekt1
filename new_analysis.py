import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from numpy import correlate
from scipy.spatial.distance import jensenshannon
from fastdtw import fastdtw
from scipy.stats import pearsonr

os.chdir("/home/vaclav/github_projekty/ISS-projekt1/")

reference_file = "Audi_A5_Drive.wav"
files = ["test_m.wav", "test_j.wav", "test_i.wav", "test_g.wav"]

audio_data = []


def normal_fft(file_path):
    sample_rate, data = wavfile.read(file_path)
    fft_values = np.abs(np.fft.rfft(data))
    fft_values = fft_values / np.max(fft_values)
    return fft_values


def euclidean_distance(first_file, test_fft):
    distance = euclidean(first_file, test_fft)
    similarity = 1 / (1 + distance)
    print(f"Euclidean similarity between {reference_file} and {file}: {similarity}")
    return

def Pearson_correlation(first_file, test_fft):
    correlation, _ = pearsonr(first_file, test_fft)
    print(f"Pearson similarity between {reference_file} and {file}: {correlation}")

def Divergence(first_file, test_fft):
    reference_fft_normalized = first_file / np.sum(first_file)
    test_fft_normalized = test_fft / np.sum(test_fft)
    distance = jensenshannon(reference_fft_normalized, test_fft_normalized)
    similarity = 1 - distance  # Convert to similarity
    print(f"Divergence similarity between {reference_file} and {file}: {similarity}")


def Cross_correlation(first_file, test_fft):
    correlation = np.max(np.correlate(first_file, test_fft, mode='full'))
    normalized_correlation = correlation / (np.linalg.norm(first_file) * np.linalg.norm(test_fft))
    print(f"Cross correlation similarity between {reference_file} and {file}: {normalized_correlation}")

first_file = normal_fft(reference_file)

print("\n")
for file in files:
    test_fft = normal_fft(file)
    euclidean_distance(first_file, test_fft)

print("\n")
for file in files:
    test_fft = normal_fft(file)
    Divergence(first_file, test_fft)

print("\n")
for file in files:
    test_fft = normal_fft(file)
    Pearson_correlation(first_file, test_fft)

print("\n")
for file in files:
    test_fft = normal_fft(file)
    Cross_correlation(first_file, test_fft)






#for file in files:
#    sample_rate, data = wavfile.read(file)
#    audio_data.append(data)
#    n = len(data)
#    freq = np.fft.rfftfreq(n, d=1/sample_rate)
#    fft_values = np.abs(np.fft.rfft(data))
#    print(f"File: {file}")
#    for f, v in zip(freq, fft_values):
#        print(f"Frequency: {f:.2f} Hz, Amplitude: {v:.2f}")
#    plt.figure()
#    plt.plot(freq, fft_values)
#    plt.title(f"Frequency Spectrum of {file}")
#    plt.xlabel("Frequency (Hz)")
#    plt.ylabel("Amplitude")
#    plt.grid()
#    plt.show()

