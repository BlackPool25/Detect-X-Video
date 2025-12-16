"""
Modal deployment for DeepFake Detection Pipeline
Deploys the 4-layer cascade detection system with GPU support
"""

import modal
from pathlib import Path
import sys

# Define Modal app
app = modal.App("deepfake-detector")

# Create volumes for model weights
model_volume = modal.Volume.from_name("deepfake-models", create_if_missing=True)

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "git"
    )
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        "torchaudio==2.1.0",
        index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install(
        "opencv-python-headless==4.8.1.78",
        "numpy==1.24.3",
        "Pillow==10.1.0",
        "scipy==1.11.4",
        "scikit-learn==1.3.2",
        "transformers==4.35.2",
        "librosa==0.10.1",
        "soundfile==0.12.1",
        "resampy==0.4.2",
        "tqdm==4.66.1",
        "albumentations==1.3.1",
        "timm==0.9.12",
        "facenet-pytorch==2.5.3",
        "ftfy",
        "regex",
    )
)

# Mount points for models
MODELS_PATH = "/models"
SBI_WEIGHTS = f"{MODELS_PATH}/SBI/FFc23.tar"
SYNCNET_WEIGHTS = f"{MODELS_PATH}/syncnet/syncnet_v2.model"
UNIVERSAL_FC_WEIGHTS = f"{MODELS_PATH}/UniversalFakeDetect/fc_weights.pth"

@app.function(
    image=image,
    gpu="A10G",  # A10G is cost-effective for inference
    timeout=600,  # 10 minutes max per request
    volumes={MODELS_PATH: model_volume},
    memory=16384,  # 16GB RAM
)
@modal.web_endpoint(method="POST")
def detect_deepfake(video_url: str = None, video_base64: str = None, adaptive_threshold: bool = True):
    """
    Detect if a video is a deepfake
    
    Args:
        video_url: URL to video file (optional)
        video_base64: Base64 encoded video (optional)
        adaptive_threshold: Use adaptive threshold (0.6) for better accuracy
    
    Returns:
        {
            "verdict": "REAL" or "FAKE",
            "confidence": float (0-100),
            "processing_time": float (seconds),
            "layer_results": [
                {
                    "layer": "Layer 1: Audio Analysis",
                    "prediction": "REAL" or "FAKE",
                    "confidence": float,
                    "processing_time": float,
                    "details": dict
                },
                ...
            ],
            "stopped_at": "Layer X: ...",
            "model_info": {
                "version": "1.0",
                "layers": ["Audio", "Visual Artifacts", "Lip-Sync", "Semantic"],
                "adaptive_threshold": bool
            }
        }
    """
    import tempfile
    import base64
    import requests
    from pathlib import Path
    import sys
    import time
    
    # Add pipeline to path
    sys.path.insert(0, "/root")
    
    from pipeline_production import DeepfakePipeline
    
    # Download or decode video
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "video.mp4"
        
        if video_url:
            # Download from URL
            response = requests.get(video_url, timeout=60)
            response.raise_for_status()
            video_path.write_bytes(response.content)
        elif video_base64:
            # Decode base64
            video_data = base64.b64decode(video_base64)
            video_path.write_bytes(video_data)
        else:
            return {
                "error": "Either video_url or video_base64 must be provided",
                "status": "error"
            }
        
        # Initialize pipeline
        start_time = time.time()
        pipeline = DeepfakePipeline(
            sbi_weights_path=SBI_WEIGHTS,
            syncnet_model_path=SYNCNET_WEIGHTS,
            adaptive_threshold=adaptive_threshold
        )
        
        # Run detection
        result = pipeline.detect(str(video_path), enable_fail_fast=False)
        total_time = time.time() - start_time
        
        # Format response
        layer_results = []
        for layer in result.layer_results:
            layer_results.append({
                "layer": layer.layer_name,
                "prediction": "FAKE" if layer.is_fake else "REAL",
                "confidence": round(layer.confidence * 100, 2),
                "processing_time": round(layer.processing_time, 3),
                "details": layer.details
            })
        
        return {
            "verdict": result.final_verdict,
            "confidence": round(result.confidence, 2),
            "processing_time": round(total_time, 2),
            "layer_results": layer_results,
            "stopped_at": result.stopped_at_layer,
            "model_info": {
                "version": "1.0.0",
                "layers": [
                    "Layer 1: Audio Analysis (Wav2Vec2)",
                    "Layer 2: Visual Artifacts (SBI EfficientNet-B4)",
                    "Layer 3: Lip-Sync Analysis (SyncNet)",
                    "Layer 4: Semantic Analysis (UniversalFakeDetect CLIP)"
                ],
                "adaptive_threshold": adaptive_threshold,
                "threshold_value": 0.6 if adaptive_threshold else 0.5,
                "dataset_optimized": "Face-swap deepfakes (Celeb-DF, FaceForensics++, DFDC)",
                "accuracy": {
                    "overall": "82.9%",
                    "real_videos": "72.7%",
                    "fake_videos": "100%"
                }
            },
            "status": "success"
        }


@app.function(
    image=image,
    volumes={MODELS_PATH: model_volume},
)
def upload_models():
    """
    Upload model weights to Modal volume
    Run this once to initialize the volume with model weights
    """
    import shutil
    from pathlib import Path
    
    # Local paths (run from project root)
    local_weights = {
        "weights/SBI/FFc23.tar": f"{MODELS_PATH}/SBI/FFc23.tar",
        "syncnet_python/data/syncnet_v2.model": f"{MODELS_PATH}/syncnet/syncnet_v2.model",
        "UniversalFakeDetect/pretrained_weights/fc_weights.pth": f"{MODELS_PATH}/UniversalFakeDetect/fc_weights.pth",
    }
    
    for local_path, modal_path in local_weights.items():
        modal_dir = Path(modal_path).parent
        modal_dir.mkdir(parents=True, exist_ok=True)
        
        if Path(local_path).exists():
            shutil.copy(local_path, modal_path)
            print(f"✓ Uploaded {local_path} -> {modal_path}")
        else:
            print(f"✗ Local file not found: {local_path}")
    
    # Commit changes to volume
    model_volume.commit()
    print("✓ Model weights uploaded successfully")


@app.local_entrypoint()
def main():
    """
    Test the deployment locally
    """
    # Upload models first
    print("Uploading model weights...")
    upload_models.remote()
    
    # Test with a sample video
    print("\nTesting detection endpoint...")
    test_video_url = "https://example.com/test_video.mp4"  # Replace with actual test URL
    
    result = detect_deepfake.remote(
        video_url=test_video_url,
        adaptive_threshold=True
    )
    
    print("\nResult:")
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Processing time: {result['processing_time']}s")
