"""
Segmented Overlap-Add (OLA) Micro-Chunking & Crossfade Blending Engine
Allows long audio processing in bounded VRAM (4GB) with zero boundary clicks or phase distortion.
"""

import math
from typing import Callable, Optional, Tuple, List, Union
import numpy as np
import logging

logger = logging.getLogger("Avenox.Chunker")


def get_hann_window(length: int) -> np.ndarray:
    """Returns a periodic Hann window for smooth crossfading."""
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(length) / length))


def chunk_and_process_ola(
    audio: np.ndarray,
    process_fn: Callable[[np.ndarray], np.ndarray],
    sr: int = 44100,
    chunk_size_sec: float = 6.0,
    overlap: int = 2,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    stage_name: str = "Processing"
) -> np.ndarray:
    """
    Splits audio (channels, total_samples) into overlapping chunks,
    processes each chunk with process_fn, and reconstructs the full-length
    audio using Overlap-Add (OLA) with Hann window crossfading.

    Parameters:
    - audio: ndarray of shape (channels, total_samples)
    - process_fn: function taking (channels, chunk_samples) -> (channels, chunk_samples)
    - sr: sample rate (e.g., 44100)
    - chunk_size_sec: duration of each segment in seconds (default 6.0s for 4GB VRAM)
    - overlap: overlap factor (2 = 50% overlap, 4 = 75% overlap)
    - progress_callback: callback(fraction_0_to_1, status_message)

    Returns:
    - Reconstructed output array of shape (channels, total_samples)
    """
    channels, total_samples = audio.shape
    chunk_samples = int(chunk_size_sec * sr)
    
    # If audio is shorter than chunk size, process in a single pass
    if total_samples <= chunk_samples:
        if progress_callback:
            progress_callback(0.5, f"{stage_name}: Memproses track utuh...")
        out = process_fn(audio)
        if progress_callback:
            progress_callback(1.0, f"{stage_name}: Selesai.")
        return out

    # Calculate step size
    hop_samples = chunk_samples // overlap
    
    # Create window for blending
    window = get_hann_window(chunk_samples).astype(np.float32)
    # Reshape window for broadcasting: (1, chunk_samples)
    window = np.expand_dims(window, axis=0)

    # Pad audio at the end so all samples are covered
    pad_samples = (chunk_samples - (total_samples - chunk_samples) % hop_samples) % hop_samples
    padded_audio = np.pad(audio, ((0, 0), (0, pad_samples + chunk_samples)), mode="reflect")
    
    # Output buffer and normalization weight buffer
    padded_total = padded_audio.shape[1]
    output_accum = np.zeros((channels, padded_total), dtype=np.float32)
    weight_accum = np.zeros((1, padded_total), dtype=np.float32)

    # List of chunk start indices
    start_indices = list(range(0, padded_total - chunk_samples + 1, hop_samples))
    total_chunks = len(start_indices)
    
    logger.info(f"{stage_name}: Memulai OLA chunking | Total Chunks: {total_chunks} | Chunk: {chunk_size_sec}s | Overlap: {overlap}x")

    for idx, start_idx in enumerate(start_indices):
        end_idx = start_idx + chunk_samples
        chunk_in = padded_audio[:, start_idx:end_idx]

        # Execute model inference function on the chunk
        chunk_out = process_fn(chunk_in)

        # Apply Hann window to both the processed chunk and the weight accumulator
        output_accum[:, start_idx:end_idx] += chunk_out * window
        weight_accum[:, start_idx:end_idx] += window

        # Report progress
        if progress_callback:
            progress = (idx + 1) / total_chunks
            progress_callback(progress, f"{stage_name}: Chunk {idx + 1}/{total_chunks} ({int(progress * 100)}%)")

    # Normalize by accumulated weights to prevent gain distortion
    # Avoid division by zero
    weight_accum = np.maximum(weight_accum, 1e-8)
    reconstructed_padded = output_accum / weight_accum

    # Trim back to original length
    reconstructed = reconstructed_padded[:, :total_samples]
    return reconstructed
