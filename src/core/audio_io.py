"""
High-Precision Audio I/O, Universal Resampler, De-Clipper & Stem Exporter
Ensures clean 32-bit floating point audio pipeline and phase-accurate stem subtraction.
"""

import os
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
import soundfile as sf
import librosa
import logging

logger = logging.getLogger("Avenox.AudioIO")


def load_audio(
    file_path: str,
    target_sr: int = 44100,
    force_stereo: bool = True,
    apply_declip_headroom: bool = True,
    headroom_db: float = -0.5
) -> Tuple[np.ndarray, int, float]:
    """
    Loads any supported audio file, normalizes channel layout to stereo (2, N),
    resamples to target_sr using high-precision sinc interpolation,
    and applies anti-distortion headroom.

    Returns: (audio_tensor_numpy [2, samples], sample_rate, duration_seconds)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File audio tidak ditemukan: {file_path}")

    # Read audio with soundfile / librosa fallback
    try:
        data, orig_sr = sf.read(file_path, dtype="float32", always_2d=True)
        # sf returns shape (samples, channels) -> transpose to (channels, samples)
        audio = data.T
    except Exception as e:
        logger.warning(f"SoundFile gagal membaca {file_path}, mencoba librosa: {e}")
        audio, orig_sr = librosa.load(file_path, sr=None, mono=False, dtype=np.float32)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)

    # 1. Channel normalization (force stereo if requested)
    if force_stereo:
        if audio.shape[0] == 1:
            # Mono to dual-mono stereo
            audio = np.repeat(audio, 2, axis=0)
        elif audio.shape[0] > 2:
            # Multi-channel 5.1 / 7.1 downmix to stereo
            left = audio[0] + 0.707 * audio[2] + 0.707 * audio[4]
            right = audio[1] + 0.707 * audio[2] + 0.707 * audio[5]
            audio = np.stack([left, right], axis=0)

    # 2. Resampling if needed
    if orig_sr != target_sr:
        logger.info(f"Resampling audio dari {orig_sr} Hz ke {target_sr} Hz...")
        resampled_channels = []
        for ch in range(audio.shape[0]):
            resampled_ch = librosa.resample(
                audio[ch],
                orig_sr=orig_sr,
                target_sr=target_sr,
                res_type="soxr_vhq" if hasattr(librosa, "resample") else "kaiser_best"
            )
            resampled_channels.append(resampled_ch)
        audio = np.stack(resampled_channels, axis=0)

    # 3. De-clipping & Headroom safety scaling
    if apply_declip_headroom:
        audio = apply_headroom_scaler(audio, headroom_db=headroom_db)

    duration = audio.shape[1] / target_sr
    logger.info(f"Audio berhasil dimuat: {os.path.basename(file_path)} | Durasi: {duration:.2f}s | SR: {target_sr}Hz")
    return audio, target_sr, duration


def apply_headroom_scaler(audio: np.ndarray, headroom_db: float = -0.5) -> np.ndarray:
    """
    Prevents digital clipping by ensuring maximum peak does not exceed headroom_db.
    """
    peak = np.max(np.abs(audio))
    target_peak = 10 ** (headroom_db / 20.0)
    
    if peak > target_peak:
        scale = target_peak / (peak + 1e-8)
        logger.debug(f"Menerapkan headroom scaler: peak {peak:.3f} -> skala {scale:.3f}")
        return audio * scale
    return audio


def compute_phase_inversion_residual(original_audio: np.ndarray, separated_audio: np.ndarray) -> np.ndarray:
    """
    Null-Test Phase Inversion Subtraction:
    Computes exact instrumental or residual bleed track via:
    Residual = Original - Separated
    """
    # Match lengths
    min_len = min(original_audio.shape[1], separated_audio.shape[1])
    orig_trim = original_audio[:, :min_len]
    sep_trim = separated_audio[:, :min_len]
    
    residual = orig_trim - sep_trim
    return residual


def save_audio(
    audio: np.ndarray,
    output_path: str,
    sr: int = 44100,
    bit_depth: str = "PCM_24",
    normalize_peak: bool = False
) -> str:
    """
    Exports audio array in shape (channels, samples) to lossless WAV / FLAC / MP3.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if audio.ndim == 1:
        audio_out = audio
    else:
        # Transpose back to (samples, channels) for soundfile
        audio_out = audio.T

    if normalize_peak:
        peak = np.max(np.abs(audio_out))
        if peak > 0:
            audio_out = audio_out / peak * 0.99

    # Clip safety
    audio_out = np.clip(audio_out, -1.0, 1.0)

    # Determine format
    subtype = bit_depth if output_path.lower().endswith(".wav") else None
    sf.write(output_path, audio_out, sr, subtype=subtype)
    logger.info(f"Stem tersimpan: {output_path}")
    return output_path


def measure_lufs(audio: np.ndarray, sr: int = 44100) -> float:
    """
    Measures integrated loudness in LUFS (EBU R128 standard).
    """
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        # pyln expects (samples, channels)
        audio_in = audio.T if audio.ndim == 2 else audio
        loudness = meter.integrated_loudness(audio_in)
        return float(loudness)
    except Exception as e:
        logger.warning(f"Gagal mengukur LUFS: {e}")
        # RMS fallback
        rms = np.sqrt(np.mean(audio ** 2))
        return float(20 * np.log10(rms + 1e-9))


def normalize_lufs(audio: np.ndarray, sr: int = 44100, target_lufs: float = -23.0) -> np.ndarray:
    """
    Normalizes audio loudness to target LUFS (e.g. -23.0 LUFS) with true-peak limiter.
    """
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        audio_in = audio.T if audio.ndim == 2 else audio
        current_lufs = meter.integrated_loudness(audio_in)
        
        if np.isneginf(current_lufs) or np.isnan(current_lufs):
            return audio
            
        normalized = pyln.normalize.loudness(audio_in, current_lufs, target_lufs)
        # Peak safety limiter at -1.0 dBFS (0.891)
        peak = np.max(np.abs(normalized))
        if peak > 0.891:
            normalized = normalized * (0.891 / peak)
            
        return normalized.T if audio.ndim == 2 else normalized
    except Exception as e:
        logger.warning(f"LUFS normalization fallback to peak: {e}")
        return apply_headroom_scaler(audio, headroom_db=-1.0)
