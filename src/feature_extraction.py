import numpy as np

def extract_rir_features(y, sr):
    """
    y: waveform (from your load_audio)
    sr: sample rate
    Returns: dict with T60 and decay slope
    """
    # 1. Compute the energy decay curve (Schroeder integration)
    energy = y ** 2
    edc = np.cumsum(energy[::-1])[::-1]
    edc = edc / (edc[0] + 1e-10)
    edc_db = 10 * np.log10(edc + 1e-10)

    # 2. Estimate T60 via T20 extrapolation
    try:
        idx_start = np.where(edc_db <= -5)[0][0]
        idx_end = np.where(edc_db <= -25)[0][0]
        t_start = idx_start / sr
        t_end = idx_end / sr
        t20 = t_end - t_start
        t60 = 3 * t20
    except IndexError:
        t60 = np.nan
        idx_start, idx_end = None, None

    # 3. Decay slope (dB/sec) via linear regression over the same region
    try:
        time_axis = np.arange(idx_start, idx_end) / sr
        decay_segment = edc_db[idx_start:idx_end]
        slope, intercept = np.polyfit(time_axis, decay_segment, 1)
    except (TypeError, ValueError):
        slope = np.nan

    # 4. Return as a dict
    return {
        "t60": t60,
        "decay_slope": slope
    }