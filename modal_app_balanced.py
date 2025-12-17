"""
Modal Deployment for BALANCED 3-Layer Deepfake Detection Pipeline
==================================================================
Full implementation of pipeline_balanced.py on Modal with T4 GPU

Architecture:
- Layer 1: Visual Artifacts (SBI EfficientNet-B4) - PRIMARY [1.8s]
- Layer 2: Temporal Consistency (Frame Differencing) - SECONDARY [0.5s]  
- Layer 3: Audio-Visual Sync (Lightweight) - TERTIARY [1.5s]

Target: 85-90% accuracy in <5 seconds per video
"""

import modal
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# ============================================================================
# Modal Configuration
# ============================================================================

app = modal.App("deepfake-detector-balanced-3layer")

# Mount SBI weights
WEIGHTS_PATH = "/root/weights"

# GPU Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "torchaudio==2.1.2",
        "opencv-python==4.9.0.80",
        "numpy==1.26.3",
        "efficientnet-pytorch==0.7.1",
        "albumentations==1.3.1",
        "fastapi[standard]",
        "retinaface-pytorch",
        "requests",  # For downloading videos
        "tqdm",  # For SBI preprocessing
    )
    .run_commands(
        # Clone SelfBlendedImages for SBI detector
        "git clone https://github.com/mapooon/SelfBlendedImages.git /root/SelfBlendedImages",
        "cd /root/SelfBlendedImages && git checkout 9fe4efe",
    )
    .add_local_dir(
        local_path="./weights/SBI",
        remote_path=f"{WEIGHTS_PATH}/SBI"
    )
)

# ============================================================================
# Response Models
# ============================================================================

@dataclass
class LayerResult:
    """Result from a single detection layer"""
    layer_name: str
    is_fake: bool
    confidence: float
    processing_time: float
    details: Dict

@dataclass
class PipelineResult:
    """Final pipeline result"""
    video_path: str
    final_verdict: str
    confidence: float
    stopped_at_layer: str
    layer_results: List[LayerResult]
    total_time: float


# ============================================================================
# Detection Layers
# ============================================================================

@app.cls(
    image=image,
    gpu="T4",
    timeout=600,
    min_containers=0,  # Don't keep warm to save costs
)
class BalancedDeepfakeDetector:
    """
    Balanced 3-Layer Deepfake Detector
    Fully implements pipeline_balanced.py
    """
    
    @modal.enter()
    def setup(self):
        """Load all models once when container starts"""
        import torch
        import sys
        
        print("[SETUP] Initializing BALANCED 3-Layer Pipeline...")
        
        # Add SBI paths
        sys.path.insert(0, '/root/SelfBlendedImages/src')
        sys.path.insert(0, '/root/SelfBlendedImages/src/inference')
        
        from model import Detector as SBIDetector
        from retinaface.pre_trained_models import get_model as get_face_detector
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[SETUP] Using device: {self.device}")
        
        # ===== Layer 1: Visual Artifact Detector =====
        print("[SETUP] Loading Layer 1: SBI Visual Artifact Detector (Primary)...")
        self.sbi_model = SBIDetector().to(self.device)
        weights_path = f"{WEIGHTS_PATH}/SBI/FFc23.tar"
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.sbi_model.load_state_dict(checkpoint['model'])
        self.sbi_model.eval().half()  # FP16
        
        self.face_detector = get_face_detector("resnet50_2020-07-20", max_size=2048, device=self.device)
        self.face_detector.eval()
        
        # Layer 1 parameters
        self.layer1_threshold = 0.33
        self.layer1_n_frames = 8
        print(f"  ✓ Threshold: {self.layer1_threshold}, Frames: {self.layer1_n_frames}")
        
        # ===== Layer 2: Temporal Consistency =====
        print("[SETUP] Configuring Layer 2: Temporal Consistency Detector (Secondary)...")
        self.layer2_threshold = 0.15
        self.layer2_n_frames = 12
        print(f"  ✓ Threshold: {self.layer2_threshold}, Frames: {self.layer2_n_frames}")
        
        # ===== Layer 3: Audio-Visual Sync =====
        print("[SETUP] Configuring Layer 3: Audio-Visual Sync Detector (Tertiary)...")
        self.layer3_threshold = 0.3
        print(f"  ✓ Threshold: {self.layer3_threshold}")
        
        print("[SETUP] ✓ All 3 layers ready - ready for inference")
    
    def _has_audio_stream(self, video_path: str) -> bool:
        """Check if video has audio"""
        import subprocess
        try:
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', 
                   '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                  text=True, timeout=5)
            return 'audio' in result.stdout
        except:
            return False
    
    def _layer1_detect(self, video_path: str) -> LayerResult:
        """Layer 1: Visual Artifact Detection (PRIMARY)"""
        import torch
        import numpy as np
        import time
        import sys
        
        sys.path.insert(0, '/root/SelfBlendedImages/src/inference')
        from preprocess import extract_frames
        
        start_time = time.time()
        
        try:
            face_list, idx_list = extract_frames(
                video_path, self.layer1_n_frames, self.face_detector, image_size=(380, 380)
            )
            
            if len(face_list) == 0:
                return LayerResult(
                    layer_name="Layer 1: Visual Artifacts",
                    is_fake=False, confidence=0.0,
                    processing_time=time.time() - start_time,
                    details={"error": "No faces detected"}
                )
            
            faces_tensor = torch.tensor(face_list).float() / 255.0
            faces_tensor = faces_tensor.half().to(self.device)
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                pred = self.sbi_model(faces_tensor).softmax(1)[:, 1]
            
            # Aggregate predictions
            pred_list = []
            idx_img = -1
            for i in range(len(pred)):
                if idx_list[i] != idx_img:
                    pred_list.append([])
                    idx_img = idx_list[i]
                pred_list[-1].append(pred[i].item())
            
            pred_res = np.array([max(p) for p in pred_list])
            avg_fake_prob = float(pred_res.mean())
            
            is_fake = avg_fake_prob > self.layer1_threshold
            confidence = avg_fake_prob if is_fake else (1.0 - avg_fake_prob)
            
            return LayerResult(
                layer_name="Layer 1: Visual Artifacts",
                is_fake=is_fake,
                confidence=confidence,
                processing_time=time.time() - start_time,
                details={
                    "avg_fake_probability": avg_fake_prob,
                    "max_fake_probability": float(pred_res.max()),
                    "threshold": self.layer1_threshold,
                    "frames_analyzed": len(face_list)
                }
            )
            
        except Exception as e:
            import traceback
            return LayerResult(
                layer_name="Layer 1: Visual Artifacts",
                is_fake=False, confidence=0.0,
                processing_time=time.time() - start_time,
                details={"error": str(e), "traceback": traceback.format_exc()}
            )
    
    def _layer2_detect(self, video_path: str) -> LayerResult:
        """Layer 2: Temporal Consistency (SECONDARY)"""
        import cv2
        import numpy as np
        import time
        
        start_time = time.time()
        
        try:
            # Extract frames
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames < self.layer2_n_frames:
                indices = list(range(total_frames))
            else:
                indices = np.linspace(0, total_frames - 1, self.layer2_n_frames, dtype=int)
            
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (224, 224))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(gray)
            
            cap.release()
            
            if len(frames) < 2:
                return LayerResult(
                    layer_name="Layer 2: Temporal Consistency",
                    is_fake=False, confidence=0.0,
                    processing_time=time.time() - start_time,
                    details={"error": "Not enough frames"}
                )
            
            # Compute frame differences
            diffs = []
            for i in range(len(frames) - 1):
                diff = cv2.absdiff(frames[i], frames[i+1])
                mean_diff = diff.mean() / 255.0
                diffs.append(mean_diff)
            
            diffs = np.array(diffs)
            
            # Temporal analysis
            std_diff = float(diffs.std())
            mean_diff = float(diffs.mean())
            temporal_score = std_diff / (mean_diff + 0.01)
            
            is_fake = temporal_score > self.layer2_threshold
            confidence = temporal_score if is_fake else (1.0 / (temporal_score + 1.0))
            
            return LayerResult(
                layer_name="Layer 2: Temporal Consistency",
                is_fake=is_fake,
                confidence=min(confidence, 0.8),
                processing_time=time.time() - start_time,
                details={
                    "temporal_score": temporal_score,
                    "std_diff": std_diff,
                    "mean_diff": mean_diff,
                    "threshold": self.layer2_threshold
                }
            )
            
        except Exception as e:
            import traceback
            return LayerResult(
                layer_name="Layer 2: Temporal Consistency",
                is_fake=False, confidence=0.0,
                processing_time=time.time() - start_time,
                details={"error": str(e), "traceback": traceback.format_exc()}
            )
    
    def _layer3_detect(self, video_path: str) -> LayerResult:
        """Layer 3: Audio-Visual Synchronization (TERTIARY)"""
        import subprocess
        import torchaudio
        import numpy as np
        import tempfile
        import os
        import time
        
        start_time = time.time()
        
        # Skip if no audio
        if not self._has_audio_stream(video_path):
            return LayerResult(
                layer_name="Layer 3: Audio-Visual Sync",
                is_fake=False, confidence=0.0,
                processing_time=time.time() - start_time,
                details={"skipped": "No audio stream"}
            )
        
        try:
            # Extract audio energy
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.close()
            
            cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                   '-ar', '16000', '-ac', '1', '-y', temp_audio.name]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            
            waveform, sr = torchaudio.load(temp_audio.name)
            os.unlink(temp_audio.name)
            
            # Compute energy in 100ms windows
            audio = waveform.numpy()[0]
            hop_length = int(sr * 0.1)
            energy = []
            for i in range(0, len(audio), hop_length):
                chunk = audio[i:i+hop_length]
                if len(chunk) > 0:
                    energy.append(np.sqrt(np.mean(chunk**2)))
            
            energy = np.array(energy)
            
            if len(energy) < 5:
                return LayerResult(
                    layer_name="Layer 3: Audio-Visual Sync",
                    is_fake=False, confidence=0.0,
                    processing_time=time.time() - start_time,
                    details={"error": "Audio too short"}
                )
            
            # Coefficient of variation
            energy_std = float(energy.std())
            energy_mean = float(energy.mean())
            cv = energy_std / (energy_mean + 0.001)
            
            # Too smooth suggests synthetic
            is_suspicious = (cv < 0.3) or (cv > 2.0)
            is_fake = is_suspicious and (cv < 0.3)
            confidence = 0.6 if is_fake else 0.5
            
            return LayerResult(
                layer_name="Layer 3: Audio-Visual Sync",
                is_fake=is_fake,
                confidence=confidence,
                processing_time=time.time() - start_time,
                details={
                    "coefficient_variation": cv,
                    "energy_std": energy_std,
                    "energy_mean": energy_mean,
                    "note": "Basic heuristic, low confidence"
                }
            )
            
        except Exception as e:
            import traceback
            return LayerResult(
                layer_name="Layer 3: Audio-Visual Sync",
                is_fake=False, confidence=0.0,
                processing_time=time.time() - start_time,
                details={"error": str(e), "traceback": traceback.format_exc()}
            )
    
    @modal.method()
    def detect(self, video_url: str, enable_fail_fast: bool = False) -> Dict:
        """
        Run 3-layer detection with weighted ensemble
        
        Args:
            video_url: URL or path to video file
            enable_fail_fast: Stop early if Layer 1 very confident (>0.8)
        
        Returns:
            PipelineResult as dict
        """
        import torch
        import tempfile
        import time
        import requests
        
        start_time = time.time()
        layer_results = []
        
        print(f"\n{'=' * 80}")
        print(f"Analyzing: {video_url}")
        print(f"{'=' * 80}\n")
        
        torch.cuda.empty_cache()
        
        try:
            # Download video if URL (with proper error handling)
            if video_url.startswith('http'):
                print(f"[DETECT] Downloading video from {video_url}")
                try:
                    response = requests.get(video_url, timeout=30)
                    response.raise_for_status()
                    
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                        tmp.write(response.content)
                        video_path = tmp.name
                except requests.exceptions.RequestException as e:
                    # Return error as dict (serializable)
                    return {
                        "error": f"Failed to download video: {str(e)}",
                        "error_type": "DownloadError"
                    }
            else:
                video_path = video_url
            
            # Layer 1: Visual (PRIMARY - 80% weight)
            print("► Layer 1: Visual Artifacts...")
            result1 = self._layer1_detect(video_path)
            layer_results.append(result1)
            print(f"  {'FAKE' if result1.is_fake else 'REAL'} (conf: {result1.confidence:.2%}, time: {result1.processing_time:.2f}s)")
            
            # Early exit if Layer 1 very confident
            if enable_fail_fast and result1.is_fake and result1.confidence > 0.8:
                print(f"\n⚠ High-confidence FAKE detected - stopping early")
                return self._finalize_result(video_url, layer_results, start_time, "Layer 1: Visual Artifacts")
            
            # Layer 2: Temporal (15% weight)
            print("\n► Layer 2: Temporal Consistency...")
            result2 = self._layer2_detect(video_path)
            layer_results.append(result2)
            print(f"  {'FAKE' if result2.is_fake else 'REAL'} (conf: {result2.confidence:.2%}, time: {result2.processing_time:.2f}s)")
            
            # Layer 3: Audio (5% weight)
            print("\n► Layer 3: Audio-Visual Sync...")
            result3 = self._layer3_detect(video_path)
            layer_results.append(result3)
            print(f"  {'FAKE' if result3.is_fake else 'REAL'} (conf: {result3.confidence:.2%}, time: {result3.processing_time:.2f}s)")
            
            return self._finalize_result(video_url, layer_results, start_time, "Ensemble")
            
        except Exception as e:
            import traceback
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
    
    def _finalize_result(self, video_path, layer_results, start_time, stopped_at):
        """Weighted ensemble voting"""
        import torch
        import time
        
        # Weights: Layer1=0.80, Layer2=0.15, Layer3=0.05
        weights = [0.80, 0.15, 0.05]
        
        fake_score = 0.0
        real_score = 0.0
        total_weight = 0.0
        
        for i, result in enumerate(layer_results):
            if result.confidence > 0:
                weight = weights[min(i, len(weights)-1)]
                if result.is_fake:
                    fake_score += weight * result.confidence
                else:
                    real_score += weight * result.confidence
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            fake_score /= total_weight
            real_score /= total_weight
        
        final_verdict = "FAKE" if fake_score > real_score else "REAL"
        confidence = max(fake_score, real_score)
        
        total_time = time.time() - start_time
        
        print(f"\n{'=' * 80}")
        print(f"FINAL VERDICT: {final_verdict}")
        print(f"Confidence: {confidence:.2%} (Fake: {fake_score:.2%}, Real: {real_score:.2%})")
        print(f"Total time: {total_time:.2f}s")
        print(f"{'=' * 80}\n")
        
        torch.cuda.empty_cache()
        
        # Convert to dict for serialization
        result_obj = PipelineResult(
            video_path=video_path,
            final_verdict=final_verdict,
            confidence=confidence,
            stopped_at_layer=stopped_at,
            layer_results=layer_results,
            total_time=total_time
        )
        
        # Manually convert to dict
        return {
            "video_path": result_obj.video_path,
            "final_verdict": result_obj.final_verdict,
            "confidence": result_obj.confidence,
            "stopped_at_layer": result_obj.stopped_at_layer,
            "layer_results": [
                {
                    "layer_name": lr.layer_name,
                    "is_fake": lr.is_fake,
                    "confidence": lr.confidence,
                    "processing_time": lr.processing_time,
                    "details": lr.details
                }
                for lr in result_obj.layer_results
            ],
            "total_time": result_obj.total_time
        }


# ============================================================================
# Web Endpoints
# ============================================================================

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def detect_video(video_url: str, enable_fail_fast: bool = False) -> Dict:
    """
    Web endpoint for 3-layer deepfake detection
    
    POST /detect-video
    Body (JSON):
      {
        "video_url": "https://...",
        "enable_fail_fast": false
      }
    
    Returns:
      PipelineResult with all 3 layer outputs
    """
    detector = BalancedDeepfakeDetector()
    result = detector.detect.remote(video_url, enable_fail_fast)
    return result


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health() -> Dict:
    """Health check endpoint"""
    import torch
    
    return {
        "status": "healthy",
        "model": "Balanced-3Layer-Pipeline-v1",
        "layers": [
            "Layer 1: Visual Artifacts (SBI EfficientNet-B4)",
            "Layer 2: Temporal Consistency",
            "Layer 3: Audio-Visual Sync"
        ],
        "gpu_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "weights": [0.80, 0.15, 0.05]
    }


# ============================================================================
# Local Testing
# ============================================================================

@app.local_entrypoint()
def main(video_path: str = "test_video.mp4"):
    """Test locally before deployment"""
    detector = BalancedDeepfakeDetector()
    result = detector.detect.remote(video_path)
    
    if "error" in result:
        print(f"\n❌ ERROR: {result['error']}")
        return
    
    print("\n" + "="*80)
    print("DETECTION RESULT")
    print("="*80)
    print(f"Final Verdict: {result['final_verdict']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Stopped at: {result['stopped_at_layer']}")
    print(f"Total Time: {result['total_time']:.2f}s")
    print("\nLayer Results:")
    for lr in result['layer_results']:
        print(f"\n  {lr['layer_name']}:")
        print(f"    Verdict: {'FAKE' if lr['is_fake'] else 'REAL'}")
        print(f"    Confidence: {lr['confidence']:.2%}")
        print(f"    Time: {lr['processing_time']:.2f}s")
    print("="*80)
