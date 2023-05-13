import numpy as np
import scipy

from librosa_lite.core import mel_filter

def spectrogram(y, sr=44100, window=0.025, window_shift=0.01, nfft=512, window_fn=None, n_mel = 128, minf=None, maxf=None, pre_emphasis = 0.97):

    # pre-emphasis filter on the signal to amplify the high frequencies
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    
        # Windowing
    frame_length = int(window * sr)  # ms
    frame_step = int(window_shift * sr)  # ms

    window = np.hamming(frame_length) if window_fn is None else window_fn
    
    frames = np.array([y[i:i+frame_length] * window for i in range(0, len(y) - frame_length, frame_step)])
    magnitude_spectrum = np.abs(np.fft.rfft(frames, nfft)) ** 2
    
    filters = mel_filter(n_mel, sr, nfft, minf=minf, maxf=maxf)
    
    mel_spectrum = np.dot(magnitude_spectrum, filters.T)
    
    return mel_spectrum