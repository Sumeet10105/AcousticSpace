import librosa

def load_audio(filepath, sr=16000, mono=True):
    """
    Load audio, resample to a fixed rate, convert to mono.
    Returns (waveform, sample_rate).
    """
    y, sr = librosa.load(filepath, sr=sr, mono=mono, res_type="kaiser_best")
    return y, sr