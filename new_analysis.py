import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from numpy.correlate import correlate
from scipy.spatial.distance import jensenshannon
from dtw import accelerated_dtw
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



first_file = normal_fft(reference_file)

for file in files:
    test_fft = normal_fft(file)
    
    distance, _, _, _ = accelerated_dtw(first_file, test_fft, dist='euclidean')

    similarity = 1 / (1 + distance)
    print(f"Similarity of {reference_file} with {file} is: {similarity}")


def euclidean_distance(first_file, test_fft):
    distance = euclidean(first_file, test_fft)
    similarity = 1 / (1 + distance)
    print(f"Similarity between {reference_file} and {file}: {similarity:.2f}")

def Pearson_correlation(first_file, test_fft):
    correlation, _ = pearsonr(first_file, test_fft)
    print(f"Similarity between {reference_file} and {file}: {correlation:.2f}")

def Time_Wrapping(first_file, test_fft):
    distance, _, _, _ = accelerated_dtw(first_file, test_fft, dist='euclidean')
    similarity = 1 / (1 + distance)
    print(f"Similarity between {reference_file} and {file}: {similarity:.2f}")

def Divergence(first_file, test_fft):
    reference_fft_normalized = first_file / np.sum(first_file)
    test_fft_normalized = test_fft / np.sum(test_fft)
    distance = jensenshannon(reference_fft_normalized, test_fft_normalized)
    similarity = 1 - distance  # Convert to similarity
    print(f"Similarity between {reference_file} and {file}: {similarity:.2f}")


def Cross_correlation(first_file, test_fft):
    correlation = np.max(np.correlate(first_file, test_fft, mode='full'))
    normalized_correlation = correlation / (np.linalg.norm(first_file) * np.linalg.norm(test_fft))
    print(f"Similarity between {reference_file} and {file}: {normalized_correlation:.2f}")
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

