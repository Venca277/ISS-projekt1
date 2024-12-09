import numpy as np
import librosa
import os
print("Current working directory:", os.getcwd())

def cross_corellation(x, y):
    corellation = np.correlate(x, y, mode='full')
    norm_factor = np.sqrt(np.sum(x**2) * np.sum(y**2))
    normalized_corr = corellation / norm_factor
    return normalized_corr

def find_similarity(x, y):
    corr = cross_corellation(x,y)
    max_corr = np.max(corr)
    lag = np.argmax(corr) - (len(x)-1)
    return max_corr, lag

x, sr_x = librosa.load("/home/vaclav/github_projekty/ISS-projekt1/Audi_A5_Drive.wav", sr=None)
y, sr_y = librosa.load("/home/vaclav/github_projekty/ISS-projekt1/test_g.wav", sr=None)

min_len = min(len(x), len(y))
x = x[:min_len]
y = y[:min_len]

similarity, lag = find_similarity(x,y)
print(f"Podobnost: {similarity:.3f}, Optimální časový posun: {lag} vzorků")