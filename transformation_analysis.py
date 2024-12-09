import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def calculate_stft(file_path, n_fft=2048, hop_length=256):
    """Compute the magnitude spectrogram (STFT) for a given audio file."""
    signal, sr = librosa.load(file_path, sr=None)
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    return spectrogram, sr

def compare_spectrograms(spec1, spec2):
    """Compare two spectrograms using normalized cross-correlation."""
    min_time = min(spec1.shape[1], spec2.shape[1])  # Align along the time axis
    spec1 = spec1[:, :min_time]
    spec2 = spec2[:, :min_time]
    
    # Flatten and normalize spectrograms
    spec1_flat = spec1.flatten() / np.linalg.norm(spec1)
    spec2_flat = spec2.flatten() / np.linalg.norm(spec2)
    
    # Compute dot product as similarity measure
    similarity = np.dot(spec1_flat, spec2_flat)
    return similarity

# Known and unknown files
known_file = "/home/vaclav/github_projekty/ISS-projekt1/Mercedes_300SE_Drive.wav"
unknown_files = [
    "/home/vaclav/github_projekty/ISS-projekt1/test_m.wav",
    "/home/vaclav/github_projekty/ISS-projekt1/test_j.wav",
    "/home/vaclav/github_projekty/ISS-projekt1/test_i.wav",
    "/home/vaclav/github_projekty/ISS-projekt1/test_g.wav"
]

# Compute spectrogram for known file
known_spec, sr = calculate_stft(known_file)

# Compare with each unknown file
for i, unknown_file in enumerate(unknown_files, 1):
    unknown_spec, _ = calculate_stft(unknown_file)
    similarity = compare_spectrograms(known_spec, unknown_spec)
    print(f"Similarity with unknown{i}: {similarity:.3f}")

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(known_spec, ref=np.max),
                         sr=sr, hop_length=256, y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of Known File')
plt.show()

