"""
Multi-Model Ensemble Blending Engine
Combines outputs from multiple separation models via Weighted Median Spectrogram Blending for maximum SDR.
"""

from typing import List, Optional
import numpy as np
import librosa
import logging

logger = logging.getLogger("Avenox.Ensemble")


def blend_spectrograms_median(audio_list: List[np.ndarray], n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Computes element-wise median magnitude spectrogram across multiple model predictions
    and reconstructs waveform using the average phase.
    
    audio_list: list of 2D arrays of shape (channels, samples)
    """
    if len(audio_list) == 1:
        return audio_list[0]

    # Ensure all have the same length
    min_len = min(a.shape[1] for a in audio_list)
    aligned_audios = [a[:, :min_len] for a in audio_list]
    channels = aligned_audios[0].shape[0]

    blended_channels = []

    for ch in range(channels):
        stfts = [librosa.stft(a[ch], n_fft=n_fft, hop_length=hop_length) for a in aligned_audios]
        
        # Extract magnitudes and phases
        magnitudes = np.stack([np.abs(s) for s in stfts], axis=0)
        phases = np.stack([np.angle(s) for s in stfts], axis=0)

        # 1. Median magnitude to discard outlier noise/bleeding from any single model
        median_mag = np.median(magnitudes, axis=0)

        # 2. Circular mean phase for phase coherence
        mean_phase = np.arctan2(np.mean(np.sin(phases), axis=0), np.mean(np.cos(phases), axis=0))

        # 3. Complex reconstruction & iSTFT
        blended_stft = median_mag * np.exp(1j * mean_phase)
        reconstructed_ch = librosa.istft(blended_stft, hop_length=hop_length, length=min_len)
        blended_channels.append(reconstructed_ch)

    logger.info(f"Ensemble blending selesai untuk {len(audio_list)} model.")
    return np.stack(blended_channels, axis=0)


def blend_linear_weights(audio_list: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Combines outputs via weighted linear sum in time domain.
    """
    if len(audio_list) == 1:
        return audio_list[0]

    if weights is None:
        weights = [1.0 / len(audio_list)] * len(audio_list)
    else:
        norm = sum(weights)
        weights = [w / norm for w in weights]

    min_len = min(a.shape[1] for a in audio_list)
    blended = np.zeros_like(audio_list[0][:, :min_len])

    for a, w in zip(audio_list, weights):
        blended += a[:, :min_len] * w

    return blended
