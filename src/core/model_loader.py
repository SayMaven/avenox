"""
Universal Model Loader & Registry Manager
Dynamically discovers and loads PyTorch checkpoints and ONNX runtimes for default and custom models.
"""

import os
import yaml
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger("Avenox.ModelLoader")


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Helper to safely read a YAML configuration file."""
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Gagal membaca konfigurasi YAML {file_path}: {e}")
        return {}


def get_available_models(stage_number: int) -> List[Dict[str, Any]]:
    """
    Returns a combined list of default and custom models available for a given pipeline stage (1, 2, 3, or 4).
    """
    stage_key_map = {
        1: "stage1_separation",
        2: "stage2_deconstruction",
        3: "stage3_inpainting",
        4: "stage4_curation"
    }
    
    models = []
    
    # 1. Read default models config
    stages_cfg = load_yaml_config("configs/model_stages.yaml").get("stages", {})
    stage_key = stage_key_map.get(stage_number)
    if stage_key and stage_key in stages_cfg:
        default_info = stages_cfg[stage_key]
        models.append({
            "id": default_info.get("default_model", f"default_stage_{stage_number}"),
            "name": f"[Default] {default_info.get('name', 'Default Model')}",
            "filename": default_info.get("model_filename", ""),
            "is_custom": False,
            "stage": stage_number,
            "path": os.path.join("models/default", default_info.get("model_filename", ""))
        })

    # 2. Read custom models registry
    custom_cfg = load_yaml_config("configs/custom_models.yaml").get("custom_models", [])
    for cm in custom_cfg:
        if cm.get("stage") == stage_number:
            models.append({
                "id": cm.get("id", cm.get("name")),
                "name": f"[Custom] {cm.get('name')}",
                "filename": os.path.basename(cm.get("checkpoint_file", "")),
                "is_custom": True,
                "stage": stage_number,
                "path": cm.get("checkpoint_file", ""),
                "config": cm
            })

    # 3. Scan models/custom directory for unregistered models
    custom_dir = "models/custom"
    if os.path.exists(custom_dir):
        registered_files = {m["filename"] for m in models}
        for fname in os.listdir(custom_dir):
            if fname.endswith((".pt", ".ckpt", ".onnx")) and fname not in registered_files:
                models.append({
                    "id": f"unregistered_{fname}",
                    "name": f"[Auto-Detected] {fname}",
                    "filename": fname,
                    "is_custom": True,
                    "stage": stage_number,
                    "path": os.path.join(custom_dir, fname)
                })

    return models


def load_onnx_session(model_path: str, prefer_gpu: bool = True) -> Any:
    """
    Loads an ONNX model with CUDAExecutionProvider if available, otherwise CPU.
    """
    try:
        import onnxruntime as ort
        providers = []
        if prefer_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        
        session = ort.InferenceSession(model_path, providers=providers)
        logger.info(f"ONNX Session dimuat: {os.path.basename(model_path)} (Provider: {session.get_providers()[0]})")
        return session
    except Exception as e:
        logger.error(f"Gagal memuat sesi ONNX {model_path}: {e}")
        raise e
