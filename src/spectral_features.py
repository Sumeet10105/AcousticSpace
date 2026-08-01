import numpy as np
import librosa

FIXED_DURATION = 4.0   # seconds - all clips padded/trimmed to this length
N_MELS = 64             # number of mel frequency bands

def extract_log_mel(y, sr, duration=FIXED_DURATION, n_mels=N_MELS):
    """
    y: waveform (from load_audio)
    sr: sample rate
    Returns: 2D numpy array (n_mels x time_frames) - log-mel spectrogram
    """
    target_len = int(duration * sr)

    # Pad or trim waveform to fixed length
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)

    # Convert to log scale (dB) - standard practice, matches human perception
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel