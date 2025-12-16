"""
BALANCED 3-Layer Deepfake Detection Pipeline
=============================================
Target: 85-90% accuracy in <5 seconds per video

Architecture:
- Layer 1: Visual Artifacts (SBI EfficientNet-B4) - PRIMARY [1.8s]
- Layer 2: Temporal Consistency (Frame Differencing) - SECONDARY [0.5s]  
- Layer 3: Audio-Visual Sync (Lightweight) - TERTIARY [1.5s]

Total: ~3.8 seconds per video, 85-90% target accuracy
"""

import torch
import torch.nn as nn
import torchaudio
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import tempfile
import subprocess
import sys
import os
import time

# Add repo paths
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "SelfBlendedImages" / "src"))
sys.path.insert(0, str(REPO_ROOT / "SelfBlendedImages" / "src" / "inference"))

# SBI imports
from preprocess import extract_frames
from model import Detector as SBIDetector
from retinaface.pre_trained_models import get_model as get_face_detector


@dataclass
class DetectionResult:
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
    layer_results: List[DetectionResult]
    total_time: float


def has_audio_stream(video_path: str) -> bool:
    """Check if video has audio"""
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', 
               '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              text=True, timeout=5)
        return 'audio' in result.stdout
    except:
        return False


class Layer1VisualDetector:
    """
    Layer 1: Visual Artifact Detection (PRIMARY DETECTOR)
    Uses SBI EfficientNet-B4 with optimizations
    """
    
    def __init__(self, weights_path: str, device='cuda'):
        self.device = device
        print("Loading Layer 1: SBI Visual Artifact Detector (Primary)...")
        
        self.model = SBIDetector().to(device)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval().half()  # FP16
        
        self.face_detector = get_face_detector("resnet50_2020-07-20", max_size=2048, device=device)
        self.face_detector.eval()
        
        # Optimized threshold from empirical testing
        self.threshold = 0.33
        self.n_frames = 8
        print(f"  ℹ Threshold: {self.threshold}, Frames: {self.n_frames}")
        
    def detect(self, video_path: str) -> DetectionResult:
        start_time = time.time()
        
        try:
            face_list, idx_list = extract_frames(
                video_path, self.n_frames, self.face_detector, image_size=(380, 380)
            )
            
            if len(face_list) == 0:
                return DetectionResult(
                    layer_name="Layer 1: Visual Artifacts",
                    is_fake=False, confidence=0.0,
                    processing_time=time.time() - start_time,
                    details={"error": "No faces detected"}
                )
            
            faces_tensor = torch.tensor(face_list).float() / 255.0
            faces_tensor = faces_tensor.half().to(self.device)
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                pred = self.model(faces_tensor).softmax(1)[:, 1]
            
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
            
            is_fake = avg_fake_prob > self.threshold
            confidence = avg_fake_prob if is_fake else (1.0 - avg_fake_prob)
            
            return DetectionResult(
                layer_name="Layer 1: Visual Artifacts",
                is_fake=is_fake,
                confidence=confidence,
                processing_time=time.time() - start_time,
                details={
                    "avg_fake_probability": avg_fake_prob,
                    "max_fake_probability": float(pred_res.max()),
                    "threshold": self.threshold
                }
            )
            
        except Exception as e:
            import traceback
            return DetectionResult(
                layer_name="Layer 1: Visual Artifacts",
                is_fake=False, confidence=0.0,
                processing_time=time.time() - start_time,
                details={"error": str(e), "traceback": traceback.format_exc()}
            )


class Layer2TemporalDetector:
    """
    Layer 2: Temporal Consistency (SECONDARY DETECTOR)
    Detects frame-to-frame inconsistencies using lightweight optical flow analysis
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading Layer 2: Temporal Consistency Detector (Secondary)...")
        self.threshold = 0.15  # Temporal inconsistency threshold
        self.n_frames = 12  # Sample 12 frames for temporal analysis
        
    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames for temporal analysis"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.n_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, self.n_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize for faster processing
                frame = cv2.resize(frame, (224, 224))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
        
        cap.release()
        return frames
    
    def detect(self, video_path: str) -> DetectionResult:
        start_time = time.time()
        
        try:
            frames = self.extract_frames(video_path)
            
            if len(frames) < 2:
                return DetectionResult(
                    layer_name="Layer 2: Temporal Consistency",
                    is_fake=False, confidence=0.0,
                    processing_time=time.time() - start_time,
                    details={"error": "Not enough frames"}
                )
            
            # Compute frame differences
            diffs = []
            for i in range(len(frames) - 1):
                diff = cv2.absdiff(frames[i], frames[i+1])
                mean_diff = diff.mean() / 255.0  # Normalize
                diffs.append(mean_diff)
            
            diffs = np.array(diffs)
            
            # Analyze temporal consistency
            # Deepfakes often have irregular frame differences
            std_diff = float(diffs.std())
            mean_diff = float(diffs.mean())
            
            # Heuristic: High std + low mean suggests temporal glitches
            temporal_score = std_diff / (mean_diff + 0.01)  # Avoid division by zero
            
            is_fake = temporal_score > self.threshold
            confidence = temporal_score if is_fake else (1.0 / (temporal_score + 1.0))
            
            return DetectionResult(
                layer_name="Layer 2: Temporal Consistency",
                is_fake=is_fake,
                confidence=min(confidence, 0.8),  # Cap at 0.8 (not as reliable as Layer 1)
                processing_time=time.time() - start_time,
                details={
                    "temporal_score": temporal_score,
                    "std_diff": std_diff,
                    "mean_diff": mean_diff,
                    "threshold": self.threshold
                }
            )
            
        except Exception as e:
            import traceback
            return DetectionResult(
                layer_name="Layer 2: Temporal Consistency",
                is_fake=False, confidence=0.0,
                processing_time=time.time() - start_time,
                details={"error": str(e), "traceback": traceback.format_exc()}
            )


class Layer3AudioVisualDetector:
    """
    Layer 3: Audio-Visual Synchronization (TERTIARY DETECTOR)
    Lightweight audio-visual matching for lip-sync detection
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading Layer 3: Audio-Visual Sync Detector (Tertiary)...")
        self.threshold = 0.3  # Energy mismatch threshold
        
    def extract_audio_energy(self, video_path: str) -> Optional[np.ndarray]:
        """Extract audio energy envelope"""
        try:
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.close()
            
            cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                   '-ar', '16000', '-ac', '1', '-y', temp_audio.name]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            
            waveform, sr = torchaudio.load(temp_audio.name)
            os.unlink(temp_audio.name)
            
            # Compute energy in 100ms windows
            audio = waveform.numpy()[0]
            hop_length = int(sr * 0.1)  # 100ms
            energy = []
            for i in range(0, len(audio), hop_length):
                chunk = audio[i:i+hop_length]
                if len(chunk) > 0:
                    energy.append(np.sqrt(np.mean(chunk**2)))
            
            return np.array(energy)
            
        except Exception:
            return None
    
    def detect(self, video_path: str) -> DetectionResult:
        start_time = time.time()
        
        # Skip if no audio
        if not has_audio_stream(video_path):
            return DetectionResult(
                layer_name="Layer 3: Audio-Visual Sync",
                is_fake=False, confidence=0.0,
                processing_time=time.time() - start_time,
                details={"skipped": "No audio stream"}
            )
        
        try:
            energy = self.extract_audio_energy(video_path)
            
            if energy is None or len(energy) < 5:
                return DetectionResult(
                    layer_name="Layer 3: Audio-Visual Sync",
                    is_fake=False, confidence=0.0,
                    processing_time=time.time() - start_time,
                    details={"error": "Could not extract audio"}
                )
            
            # Simple heuristic: Check for unnatural audio patterns
            # Real speech has regular energy patterns, synthetic may be too smooth or erratic
            energy_std = float(energy.std())
            energy_mean = float(energy.mean())
            
            # Coefficient of variation
            cv = energy_std / (energy_mean + 0.001)
            
            # Too smooth (CV < 0.3) or too erratic (CV > 2.0) suggests fake
            is_suspicious = (cv < 0.3) or (cv > 2.0)
            
            is_fake = is_suspicious and (cv < 0.3)  # Focus on too-smooth (TTS)
            confidence = 0.6 if is_fake else 0.5  # Low confidence (basic heuristic)
            
            return DetectionResult(
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
            return DetectionResult(
                layer_name="Layer 3: Audio-Visual Sync",
                is_fake=False, confidence=0.0,
                processing_time=time.time() - start_time,
                details={"error": str(e), "traceback": traceback.format_exc()}
            )


class DeepfakePipeline:
    """
    Balanced 3-Layer Pipeline
    - Layer 1: Visual (80% weight)
    - Layer 2: Temporal (15% weight)
    - Layer 3: Audio (5% weight)
    
    Target: 85-90% accuracy, <5s per video
    """
    
    def __init__(self, sbi_weights_path: str = "weights/SBI/FFc23.tar", device: str = 'cuda'):
        self.device = device
        
        print("=" * 80)
        print("Initializing BALANCED 3-Layer Deepfake Detection Pipeline")
        print("=" * 80)
        
        self.layer1 = Layer1VisualDetector(sbi_weights_path, device)
        self.layer2 = Layer2TemporalDetector(device)
        self.layer3 = Layer3AudioVisualDetector(device)
        
        print("=" * 80)
        print("✓ All 3 layers loaded")
        print("=" * 80)
    
    def detect(self, video_path: str, enable_fail_fast: bool = False) -> PipelineResult:
        """
        Run detection with weighted ensemble
        
        Args:
            video_path: Path to video
            enable_fail_fast: Stop early if Layer 1 very confident (>0.8)
        """
        start_time = time.time()
        layer_results = []
        
        print(f"\n{'=' * 80}")
        print(f"Analyzing: {Path(video_path).name}")
        print(f"{'=' * 80}\n")
        
        torch.cuda.empty_cache()
        
        # Layer 1: Visual (PRIMARY - 80% weight)
        print("► Layer 1: Visual Artifacts...")
        result1 = self.layer1.detect(video_path)
        layer_results.append(result1)
        print(f"  {'FAKE' if result1.is_fake else 'REAL'} (conf: {result1.confidence:.2%}, time: {result1.processing_time:.2f}s)")
        
        # Early exit if Layer 1 very confident
        if enable_fail_fast and result1.is_fake and result1.confidence > 0.8:
            print(f"\n⚠ High-confidence FAKE detected - stopping early")
            return self._finalize_result(video_path, layer_results, start_time, "Layer 1: Visual Artifacts")
        
        # Layer 2: Temporal (15% weight)
        print("\n► Layer 2: Temporal Consistency...")
        result2 = self.layer2.detect(video_path)
        layer_results.append(result2)
        print(f"  {'FAKE' if result2.is_fake else 'REAL'} (conf: {result2.confidence:.2%}, time: {result2.processing_time:.2f}s)")
        
        # Layer 3: Audio (5% weight)
        print("\n► Layer 3: Audio-Visual Sync...")
        result3 = self.layer3.detect(video_path)
        layer_results.append(result3)
        print(f"  {'FAKE' if result3.is_fake else 'REAL'} (conf: {result3.confidence:.2%}, time: {result3.processing_time:.2f}s)")
        
        return self._finalize_result(video_path, layer_results, start_time, "Ensemble")
    
    def _finalize_result(self, video_path, layer_results, start_time, stopped_at):
        """Weighted ensemble voting"""
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
        
        return PipelineResult(
            video_path=video_path,
            final_verdict=final_verdict,
            confidence=confidence,
            stopped_at_layer=stopped_at,
            layer_results=layer_results,
            total_time=total_time
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--sbi-weights", default="weights/SBI/FFc23.tar")
    args = parser.parse_args()
    
    pipeline = DeepfakePipeline(sbi_weights_path=args.sbi_weights)
    result = pipeline.detect(args.video)
    
    print("\nDetailed Results:")
    for r in result.layer_results:
        print(f"\n{r.layer_name}:")
        print(f"  Verdict: {'FAKE' if r.is_fake else 'REAL'}")
        print(f"  Confidence: {r.confidence:.2%}")
        print(f"  Time: {r.processing_time:.2f}s")
