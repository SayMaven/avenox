"""
In-App Model Downloader with Progress Reporting & Resume Support
Downloads missing checkpoints from Hugging Face Hub / Mirrors directly into models/default/.
"""

import os
import sys
import time
import requests
from typing import Optional, Callable, Dict, Any, TypedDict
import logging

logger = logging.getLogger("Avenox.Downloader")


class ModelRegistryEntry(TypedDict):
    url: str
    description: str
    expected_size_mb: int


# Official default model download links
DEFAULT_MODEL_REGISTRY: Dict[str, ModelRegistryEntry] = {
    "mel_band_roformer_vocals.ckpt": {
        "url": "https://huggingface.co/SayMaven/avenox-models/resolve/main/mel_band_roformer_vocals.ckpt",
        "description": "Mel-Band RoFormer Vocal Separation Model (Stage 1)",
        "expected_size_mb": 460
    },
    "lead_backing_complex_unet.pt": {
        "url": "https://huggingface.co/SayMaven/avenox-models/resolve/main/lead_backing_complex_unet.pt",
        "description": "Mid-Side Lead/Backing Separation Model (Stage 2)",
        "expected_size_mb": 180
    },
    "spectral_inpainter_pconv.pt": {
        "url": "https://huggingface.co/SayMaven/avenox-models/resolve/main/spectral_inpainter_pconv.pt",
        "description": "Neural STFT Spectral Inpainting Bleed Cleaner (Stage 3)",
        "expected_size_mb": 120
    },
    "silero_vad.onnx": {
        "url": "https://huggingface.co/SayMaven/avenox-models/resolve/main/silero_vad.onnx",
        "description": "Silero Voice Activity Detector ONNX (Stage 4)",
        "expected_size_mb": 5
    }
}


def download_file_with_progress(
    url: str,
    destination_path: str,
    progress_callback: Optional[Callable[[float, float, float, str], None]] = None,
    chunk_size: int = 1024 * 1024  # 1MB
) -> bool:
    """
    Downloads a file over HTTP with resume capability and progress callbacks.
    progress_callback(fraction_0_1, downloaded_mb, total_mb, speed_str)
    """
    os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)
    temp_path = destination_path + ".download"

    downloaded_bytes = 0
    if os.path.exists(temp_path):
        downloaded_bytes = os.path.getsize(temp_path)

    headers = {}
    if downloaded_bytes > 0:
        headers["Range"] = f"bytes={downloaded_bytes}-"
        logger.info(f"Melanjutkan unduhan dari byte {downloaded_bytes}...")

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        
        # Check if server supports resume (206) or starts fresh (200)
        if response.status_code == 206:
            mode = "ab"
            total_bytes = downloaded_bytes + int(response.headers.get("content-length", 0))
        elif response.status_code == 200:
            mode = "wb"
            downloaded_bytes = 0
            total_bytes = int(response.headers.get("content-length", 0))
        else:
            logger.error(f"HTTP Error saat mengunduh model: {response.status_code} {response.reason}")
            return False

        total_mb = total_bytes / (1024 * 1024)
        start_time = time.time()
        bytes_in_session = 0

        with open(temp_path, mode) as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded_bytes += len(chunk)
                bytes_in_session += len(chunk)

                elapsed = time.time() - start_time
                speed_kb_s = (bytes_in_session / 1024) / max(elapsed, 0.001)
                speed_str = f"{speed_kb_s / 1024:.2f} MB/s" if speed_kb_s > 1024 else f"{speed_kb_s:.0f} KB/s"

                fraction = downloaded_bytes / max(total_bytes, 1)
                downloaded_mb = downloaded_bytes / (1024 * 1024)

                if progress_callback:
                    progress_callback(fraction, downloaded_mb, total_mb, speed_str)

        # Rename temp file upon completion
        if os.path.exists(destination_path):
            os.remove(destination_path)
        os.rename(temp_path, destination_path)
        logger.info(f"Model berhasil diunduh ke: {destination_path}")
        return True

    except Exception as e:
        logger.error(f"Gagal mengunduh file dari {url}: {e}")
        return False


def ensure_model_exists(
    model_filename: str,
    models_dir: str = "models/default",
    progress_callback: Optional[Callable[[float, float, float, str], None]] = None
) -> str:
    """
    Checks if model exists locally in models_dir.
    If missing, automatically downloads it from the registry.
    Returns: absolute path to the ready model file.
    """
    target_path = os.path.join(models_dir, model_filename)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1024:
        return os.path.abspath(target_path)

    # Check if registered
    if model_filename in DEFAULT_MODEL_REGISTRY:
        info = DEFAULT_MODEL_REGISTRY[model_filename]
        logger.info(f"Model {model_filename} belum ada. Mengunduh otomatis dari Hugging Face ({info['expected_size_mb']} MB)...")
        success = download_file_with_progress(
            url=info["url"],
            destination_path=target_path,
            progress_callback=progress_callback
        )
        if success:
            return os.path.abspath(target_path)
        else:
            raise RuntimeError(f"Gagal mengunduh model {model_filename}. Silakan periksa koneksi internet.")
    else:
        raise FileNotFoundError(f"Model file '{model_filename}' tidak ditemukan di '{models_dir}' dan tidak ada di registry.")
