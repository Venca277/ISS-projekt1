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
from numpy.linalg import norm

os.chdir("/home/vaclav/github_projekty/ISS-projekt1/")

reference_files = ["Peugeot_307_Drive.wav", "Mercedes_300SE_Drive.wav", "Fiat_Panda_Drive.wav", "Audi_A5_Drive.wav"]
files = ["test_m.wav", "test_j.wav", "test_i.wav", "test_g.wav"]

audio_data = []


def normal_fft(file_path):
    sample_rate, data = wavfile.read(file_path)
    fft_values = np.abs(np.fft.rfft(data))
    fft_values = fft_values / np.max(fft_values)
    return fft_values


def euclidean_distance(first_file, test_fft, reference_file):
    distance = euclidean(first_file, test_fft)
    similarity = 1 / (1 + distance)
    print(f"Euclidean similarity between {reference_file} and {file}: {similarity}")
    return

def Pearson_correlation(first_file, test_fft, reference_file):
    correlation, _ = pearsonr(first_file, test_fft)
    normalized_correlation = (correlation + 1) / 2
    print(f"Pearson similarity between {reference_file} and {file}: {normalized_correlation}")

def Divergence(first_file, test_fft, reference_file):
    reference_fft_normalized = first_file / np.sum(first_file)
    test_fft_normalized = test_fft / np.sum(test_fft)
    distance = jensenshannon(reference_fft_normalized, test_fft_normalized)
    similarity = 1 - distance  # Convert to similarity
    print(f"Divergence similarity between {reference_file} and {file}: {similarity}")

def Cross_correlation(first_file, test_fft, reference_file):
    correlation = np.max(np.correlate(first_file, test_fft, mode='full'))
    normalized_correlation = correlation / (np.linalg.norm(first_file) * np.linalg.norm(test_fft))
    print(f"Cross correlation similarity between {reference_file} and {file}: {normalized_correlation}")

def cosine_similarity(first_file, test_fft, reference_file):
    ref = np.array(first_file)
    test = np.array(test_fft)
    similarity = np.dot(ref, test) / (norm(ref) * norm(test))
    normalized_cos_sim = (similarity + 1) / 2
    print(f"Cosine similarity between {reference_file} and {file}: {normalized_cos_sim}")

def cosine_similarity_for_sine(first_file, test_fft):
    ref = np.array(first_file)
    test = np.array(test_fft)
    similarity = np.dot(ref, test) / (norm(ref) * norm(test))
    normalized_cos_sim = (similarity + 1) / 2
    return normalized_cos_sim

def sine_similarity(first_file, test_fft, reference_file):
    cos_sim = cosine_similarity_for_sine(first_file, test_fft)
    sine_sim = np.sqrt(1 - np.square(cos_sim))
    normalized_sine_sim = 1 - sine_sim
    print(f"Sine similarity between {reference_file} and {file}: {normalized_sine_sim}")


for a in reference_files:
    first_file = normal_fft(a)
    print("\n")
    print(f"-------------------------------- {a} --------------------------------")
    for file in files:
        test_fft = normal_fft(file)
        sine_similarity(first_file, test_fft, a)

    print("\n")
    for file in files:
        test_fft = normal_fft(file)
        cosine_similarity(first_file, test_fft, a)

    print("\n")
    for file in files:
        test_fft = normal_fft(file)
        euclidean_distance(first_file, test_fft, a)

    print("\n")
    for file in files:
        test_fft = normal_fft(file)
        Divergence(first_file, test_fft, a)

    print("\n")
    for file in files:
        test_fft = normal_fft(file)
        Pearson_correlation(first_file, test_fft, a)

    print("\n")
    for file in files:
        test_fft = normal_fft(file)
        Cross_correlation(first_file, test_fft, a)
    print(f"-------------------------------- END OF {a} --------------------------------")
    print("\n")



