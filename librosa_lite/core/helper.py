import numpy as np

def log_mel_spec(mel_spec):
    return 20 * np.log10(mel_spec + 1e-10)

def mel_to_db(mel, min_level_db = -100, max_level_db = 20):
    return np.clip((mel - min_level_db) / (max_level_db - min_level_db), 0, 1)