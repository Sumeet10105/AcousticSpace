"""Room Impulse Response (RIR) and reverberation feature estimation from speech."""

import logging
import torch
import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)


def extract_rir_features(
    audio: torch.Tensor, 
    sample_rate: int = 16000,
    frame_length: int = 512,
    hop_length: int = 160
) -> torch.Tensor:
    """Estimates Room Impulse Response (RIR) reverberation features from speech.

    Features estimated:
    - RT60 (reverberation decay time estimate)
    - Early Decay Time (EDT) proxy
    - Clarity index C50/C80 proxies (energy ratio of early vs late segments)

    Args:
        audio: Audio waveform tensor of shape [channels, time].
        sample_rate: Audio sampling rate in Hz (default: 16000).
        frame_length: Frame size for analysis.
        hop_length: Hop length between analysis frames.

    Returns:
        RIR features tensor of shape [channels, feature_dim, time_steps].
    """
    channels = audio.shape[0]
    device = audio.device
    audio_np = audio.cpu().numpy()
    
    # Let's frame the audio to calculate RIR features over time
    n_samples = audio.shape[1]
    num_frames = max(1, (n_samples - frame_length) // hop_length + 1)
    
    # We will return 4 features per frame:
    # 1. RT60 estimate
    # 2. Early Decay Rate
    # 3. C50 proxy (ratio of energy in first 50ms to the rest of the frame)
    # 4. C80 proxy (ratio of energy in first 80ms to the rest of the frame)
    feature_dim = 4
    
    features_all_channels = []
    
    # Time thresholds in samples
    samples_50ms = int(0.05 * sample_rate)
    samples_80ms = int(0.08 * sample_rate)
    
    for c in range(channels):
        y = audio_np[c]
        channel_features = np.zeros((feature_dim, num_frames), dtype=np.float32)
        
        for f in range(num_frames):
            start = f * hop_length
            end = start + frame_length
            frame = y[start:end]
            
            # 1. Energy ratios (C50, C80 proxies)
            energy_total = np.sum(frame ** 2) + 1e-9
            
            # C50 energy
            c50_len = min(len(frame), samples_50ms)
            energy_50ms = np.sum(frame[:c50_len] ** 2)
            c50 = 10 * np.log10((energy_50ms + 1e-9) / (energy_total - energy_50ms + 1e-9))
            
            # C80 energy
            c80_len = min(len(frame), samples_80ms)
            energy_80ms = np.sum(frame[:c80_len] ** 2)
            c80 = 10 * np.log10((energy_80ms + 1e-9) / (energy_total - energy_80ms + 1e-9))
            
            # 2. RT60 and decay via autocorrelation
            # Autocorrelation of frame to find decay pattern
            if energy_total > 1e-7:
                corr = scipy.signal.correlate(frame, frame, mode='full')
                corr = corr[len(corr)//2:]  # Take right half
                corr = corr / (corr[0] + 1e-9)  # Normalize
                
                # Find Schroeder-like integration on autocorrelation envelope
                envelope = np.abs(corr)
                schroeder = np.cumsum(envelope[::-1])[::-1]
                schroeder = schroeder / (schroeder[0] + 1e-9)
                schroeder_db = 10 * np.log10(schroeder + 1e-9)
                
                # Estimate decay rate (slope) from 0dB to -20dB
                idx_0 = 0
                indices_below_20 = np.where(schroeder_db < -20.0)[0]
                if len(indices_below_20) > 0:
                    idx_20 = indices_below_20[0]
                    # Slope: db drop / time
                    time_delta = (idx_20 - idx_0) / sample_rate
                    decay_rate = -20.0 / (time_delta + 1e-9)  # dB per second
                    rt60 = -60.0 / (decay_rate - 1e-9)  # Time to drop 60dB
                    edt = -10.0 / (decay_rate - 1e-9)  # Early decay time
                else:
                    rt60 = 2.0  # Fallback standard room RT60
                    edt = 0.3
            else:
                c50 = -60.0
                c80 = -60.0
                rt60 = 0.0
                edt = 0.0
                
            channel_features[0, f] = float(rt60)
            channel_features[1, f] = float(edt)
            channel_features[2, f] = float(c50)
            channel_features[3, f] = float(c80)
            
        features_all_channels.append(torch.from_numpy(channel_features).float())
        
    return torch.stack(features_all_channels, dim=0).to(device)
