"""
Modal Deployment for Optimized Deepfake Detection Pipeline
===========================================================
Single-layer SBI (Self-Blended Images) detector optimized for T4 GPU
Based on pipeline_optimized.py - 80% accuracy @ 1.78s per video

Features:
- T4 GPU optimization with FP16 inference
- Fast cold start with model pre-loading
- Health check endpoint
- Efficient video processing (8 frames)
"""

import modal
from typing import Dict, Optional, TypedDict

# ============================================================================
# Modal Configuration
# ============================================================================

# Create Modal stub
app = modal.App("deepfake-detector-optimized")

# Mount SBI weights
WEIGHTS_PATH = "/root/weights"

# GPU Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "opencv-python==4.9.0.80",
        "numpy==1.26.3",
        "efficientnet-pytorch==0.7.1",
        "albumentations==1.3.1",
        "tqdm==4.65.0",
        "fastapi[standard]",
        "retinaface-pytorch",  # For pre-trained face detector
    )
    .run_commands(
        # Clone SelfBlendedImages for SBI detector
        "git clone https://github.com/mapooon/SelfBlendedImages.git /root/SelfBlendedImages",
        "cd /root/SelfBlendedImages && git checkout 9fe4efe",  # Stable commit
    )
    .add_local_dir(
        local_path="./weights/SBI",
        remote_path=f"{WEIGHTS_PATH}/SBI"
    )
)

# ============================================================================
# Response Models
# ============================================================================

class DetectionResult(TypedDict):
    """Response model for deepfake detection"""
    is_fake: bool
    confidence: float  # 0-1 scale
    label: str  # "FAKE" or "REAL"
    probability_fake: float  # Raw model probability
    processing_time: float
    model_version: str
    frames_analyzed: int


# ============================================================================
# SBI Visual Detector (Layer 2 from pipeline_optimized.py)
# ============================================================================

@app.cls(
    image=image,
    gpu="T4",  # Optimized for T4 GPU
    timeout=600,
    # Keep container warm for faster subsequent requests
    scaledown_window=300,
)
class DeepfakeDetector:
    """
    Optimized single-layer deepfake detector using SBI visual artifacts
    """
    
    @modal.enter()
    def setup(self):
        """Load model once when container starts (cold start optimization)"""
        import torch
        import sys
        
        # Add SBI paths
        sys.path.insert(0, '/root/SelfBlendedImages/src')
        sys.path.insert(0, '/root/SelfBlendedImages/src/inference')
        
        from model import Detector as SBIDetector
        from retinaface.pre_trained_models import get_model as get_face_detector
        
        print("[SETUP] Loading SBI EfficientNet-B4 model...")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[SETUP] Using device: {self.device}")
        
        # Load SBI model (same as pipeline_balanced.py)
        self.sbi_model = SBIDetector().to(self.device)
        weights_path = f"{WEIGHTS_PATH}/SBI/FFc23.tar"
        print(f"[SETUP] Loading SBI weights from {weights_path}...")
        
        checkpoint = torch.load(
            weights_path,
            map_location=self.device,
            weights_only=False
        )
        self.sbi_model.load_state_dict(checkpoint['model'])
        
        # Enable FP16 for faster inference on T4
        self.sbi_model.eval().half()
        print("[SETUP] Model loaded in FP16 mode")
        
        # Load face detector (same as pipeline_balanced.py)
        print("[SETUP] Loading RetinaFace detector...")
        self.face_detector = get_face_detector("resnet50_2020-07-20", max_size=2048, device=self.device)
        self.face_detector.eval()
        
        print("[SETUP] ✓ Setup complete - ready for inference")
    
    @modal.method()
    def detect(self, video_url: str, threshold: float = 0.33) -> DetectionResult:
        """
        Detect deepfakes in a video using SBI visual artifacts
        
        Args:
            video_url: URL or path to video file
            threshold: Detection threshold (default 0.33 - empirically optimized)
        
        Returns:
            DetectionResult with prediction
        """
        import torch
        import numpy as np
        import tempfile
        import time
        import urllib.request
        from urllib.error import HTTPError, URLError
        import sys
        
        # Add SBI paths for imports
        sys.path.insert(0, '/root/SelfBlendedImages/src/inference')
        from preprocess import extract_frames
        
        start_time = time.time()

        try:
            # Download video if URL
            if video_url.startswith('http'):
                print(f"[DETECT] Downloading video from {video_url}")
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                    urllib.request.urlretrieve(video_url, tmp.name)
                    video_path = tmp.name
            else:
                video_path = video_url
            
            # Extract frames using SBI's extract_frames (same as pipeline_balanced.py)
            print("[DETECT] Extracting frames with face detection...")
            n_frames = 8
            face_list, idx_list = extract_frames(
                video_path, n_frames, self.face_detector, image_size=(380, 380)
            )
            
            if len(face_list) == 0:
                print("[DETECT] No faces detected in video")
                return {
                    "is_fake": False,
                    "confidence": 0.0,
                    "label": "REAL",
                    "probability_fake": 0.0,
                    "processing_time": time.time() - start_time,
                    "model_version": "SBI-EfficientNet-B4-Optimized-v1",
                    "frames_analyzed": 0
                }
            
            print(f"[DETECT] Analyzing {len(face_list)} face crops...")
            
            # Convert face_list to tensor (same as pipeline_balanced.py)
            faces_tensor = torch.tensor(face_list).float() / 255.0
            faces_tensor = faces_tensor.half().to(self.device)
            
            # Batch inference
            with torch.no_grad():
                if self.device == 'cuda':
                    with torch.amp.autocast('cuda'):
                        pred = self.sbi_model(faces_tensor).softmax(1)[:, 1]
                else:
                    pred = self.sbi_model(faces_tensor).softmax(1)[:, 1]
            
            # Aggregate predictions (same as pipeline_balanced.py)
            pred_list = []
            idx_img = -1
            for i in range(len(pred)):
                if idx_list[i] != idx_img:
                    pred_list.append([])
                    idx_img = idx_list[i]
                pred_list[-1].append(pred[i].item())
            
            # Get max prediction per frame and average
            pred_res = np.array([max(p) for p in pred_list])
            avg_prob = float(pred_res.mean())
            is_fake = avg_prob >= threshold
            
            processing_time = time.time() - start_time
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            print(f"[DETECT] Complete - Fake: {is_fake}, Confidence: {avg_prob:.3f}, Time: {processing_time:.2f}s")
            
            return {
                "is_fake": is_fake,
                "confidence": avg_prob if is_fake else (1 - avg_prob),
                "label": "FAKE" if is_fake else "REAL",
                "probability_fake": avg_prob,
                "processing_time": processing_time,
                "model_version": "SBI-EfficientNet-B4-Optimized-v1",
                "frames_analyzed": len(face_list)
            }

        except HTTPError as e:
            # Raise a simple, serializable exception so Modal can transport it
            raise ValueError(
                f"Failed to download video. HTTP error {e.code}: {e.reason}"
            ) from None
        except URLError as e:
            raise ValueError(
                f"Failed to download video. URL error: {getattr(e, 'reason', e)}"
            ) from None
        except Exception as e:
            # Catch-all to avoid unpicklable exceptions leaking out of this method
            raise RuntimeError(f"Deepfake detection failed: {e}") from None


# ============================================================================
# Web Endpoints
# ============================================================================

@app.function(
    image=image,
)
@modal.fastapi_endpoint(method="POST")
def detect_video(video_url: str, threshold: float = 0.33) -> Dict:
    """
    Web endpoint for video deepfake detection
    
    POST /detect-video
    Query params or JSON body:
      - video_url: "https://..."
      - threshold: 0.33
    
    We wrap errors so that clients see a JSON error instead of a generic 500.
    """
    try:
        detector = DeepfakeDetector()
        result = detector.detect.remote(video_url, threshold)
        return result
    except Exception as e:
        # Ensure the error is JSON-serializable and visible to the client
        return {
            "error": str(e),
            "error_type": e.__class__.__name__,
        }


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health() -> Dict:
    """Health check endpoint"""
    import torch
    
    return {
        "status": "healthy",
        "model": "SBI-EfficientNet-B4-Optimized-v1",
        "gpu_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


# ============================================================================
# Local Testing
# ============================================================================

@app.local_entrypoint()
def main(video_path: str = "test_video.mp4"):
    """Test locally before deployment"""
    detector = DeepfakeDetector()
    result = detector.detect.remote(video_path)
    
    print("\n" + "="*60)
    print("DETECTION RESULT")
    print("="*60)
    print(f"Label: {result.label}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Fake Probability: {result.probability_fake:.3f}")
    print(f"Processing Time: {result.processing_time:.2f}s")
    print(f"Frames Analyzed: {result.frames_analyzed}")
    print("="*60)
