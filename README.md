# Avenox Audio Studio (Avenox-DSP)

A high-fidelity neural audio demixing, vocal deconstruction, spectral inpainting, and automated voice dataset curation suite built on PyTorch and PySide6.

Avenox is designed with an isolated, self-contained embedded runtime to eliminate dependency conflicts with other local Python environments, and is specifically optimized for memory-constrained consumer GPUs (4 GB VRAM) using segmented streaming and automatic out-of-memory recovery.

---

## Key Features

### 1. Multi-Stage Separation Pipeline
* **Stage 1 - SOTA Vocal Separation:** Mel-Band RoFormer architecture operating in the time-frequency domain with dynamic overlap-add crossfading. Includes phase-inversion subtraction to extract high-fidelity instrumental stems.
* **Stage 2 - Polyphonic Vocal Deconstruction:** Mid-Side (M/S) spatial matrix decomposition combined with a multi-channel complex U-Net to separate lead vocal tracks from stereo backing harmonies.
* **Stage 3 - Neural Spectral Inpainting:** Partial-convolution magnitude spectrogram cleaner and phase resynthesis to surgically remove instrument bleed and harmonic artifacts.
* **Stage 4 - Automated Dataset Curation:** Silero Voice Activity Detection (VAD) with dynamic phrase slicing, zero-crossing boundary alignment, and EBU R128 (-23 LUFS) loudness normalization. Integrates Faster-Whisper for automated speech-to-text transcriptions.

### 2. Isolated Embedded Runtime
* Fully self-contained Python 3.11 x64 environment located within `python_embeded/`.
* Zero contamination of system-level Python installations, Conda environments, or existing machine learning applications.
* One-click automated environment provisioning via `run_avenox.bat`.

### 3. Low-VRAM Optimization Engine (4 GB Target)
* **Sequential Model Swapping:** Each neural network is loaded only during its execution stage and immediately evacuated from VRAM via explicit cache eviction and garbage collection.
* **Micro-Chunking with Hann Windowing:** Audio streams are processed in bounded 6.0-second tiles with Overlap-Add (OLA) reconstruction to maintain constant VRAM allocation regardless of total track length.
* **Automatic OOM Recovery:** Dynamic runtime chunk downscaling and transparent CPU fallback handlers prevent application termination during memory spikes.

### 4. Studio Interface & Batch Processing
* Built on PySide6 (Qt6) with asynchronous worker threads (`QThread`) to prevent interface locking during heavy computation.
* Multi-track audio player with A/B solo/mute toggles for instant stem comparison.
* Dual-view interactive waveform and Mel-spectrogram frequency visualizer.
* Batch processing queue for multi-file and album-level workflows.
* Extensible model registry supporting custom user checkpoints (`.pt`, `.ckpt`, `.onnx`) placed in `models/custom/`.

---

## System Requirements

* **Operating System:** Windows 10 / 11 (64-bit)
* **GPU:** NVIDIA GPU with at least 4 GB VRAM (CUDA 12.1 compatible)
* **Fallback Support:** CPU execution supported for systems without dedicated NVIDIA hardware
* **Storage:** Approximately 4 GB free disk space for runtime and model checkpoints
* **Network:** Internet connection required during initial launch for automated environment provisioning

---

## Getting Started

### Installation & Launch

1. Clone the repository:
   ```cmd
   git clone https://github.com/SayMaven/avenox.git
   cd avenox
   ```

2. Launch the application:
   * **Desktop GUI Mode:** Double-click `run_avenox.bat`
   * **Command Line Interface (CLI):** Run `run_cli.bat`

Upon the first launch, `run_avenox.bat` will automatically download and configure the isolated `python_embeded` runtime, install PyTorch with CUDA support, and fetch all required dependencies. Subsequent launches execute immediately without repeating setup.

---

## Directory Structure

```text
avenox/
├── python_embeded/          # Isolated Python 3.11 runtime (created on first run)
├── configs/                 # YAML and JSON configuration files
│   ├── app_config.yaml      # Global application parameters
│   ├── model_stages.yaml    # Neural network stage parameters
│   ├── vram_profiles.yaml   # Hardware memory profiles (4GB, 8GB, 16GB)
│   ├── workflow_presets.yaml# Workflow preset definitions
│   └── custom_models.yaml   # Registry for user-defined checkpoints
├── models/
│   ├── default/             # Official model weights (auto-downloaded)
│   └── custom/              # Directory for custom user checkpoints
├── src/
│   ├── core/                # Audio DSP, VRAM management, and neural inference
│   ├── gui/                 # PySide6 desktop interface and components
│   ├── pipeline.py          # Sequential pipeline controller
│   └── main.py              # Application entry point
├── training/                # Subsystem for training and fine-tuning custom models
├── scripts/                 # Environment provisioning and maintenance scripts
├── run_avenox.bat           # Primary desktop launcher
├── run_cli.bat              # Command-line interface launcher
├── pip_install.bat          # Utility to install packages into embedded runtime
└── requirements.txt         # Dependency manifest
```

---

## License

This project is licensed under the Apache License 2.0.
